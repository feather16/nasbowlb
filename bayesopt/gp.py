from copy import deepcopy

import gpytorch

from kernels import GraphKernels, Stationary
# GP model as a weighted average between the vanilla vectorial GP and the graph GP
from kernels import SumKernel
from kernels import WeisfilerLehman
from kernels import MultiscaleLaplacian
from .graph_features import FeatureExtractor
from .utils import *

import random
import math
import statistics
from typing import Optional, Union

# debug
try:
    import sys
    sys.path.append("/home/rio-hada/workspace/util")
    from debug import debug
except:
    print('# Failed to import debug')

# A vanilla GP with RBF kernel
class GP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, kernel: gpytorch.kernels, ):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        super(GP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GraphGP:
    def __init__(self, 
                 train_x: list[nx.DiGraph], 
                 train_y: torch.Tensor,
                 kernels: list[Union[WeisfilerLehman, MultiscaleLaplacian]],
                 vectorial_features: Optional[list] = None,
                 likelihood: float=1e-3,
                 weights=None,
                 vector_theta_bounds: tuple = (1e-5, 0.1),
                 graph_theta_bounds: tuple = (1e-1, 1.e1),
                 verbose: bool=False,
                 ):
        assert len(train_x) == train_y.shape[0], 'mismatch of length between train and test GP'
        assert all([isinstance(x, nx.Graph) for x in train_x]), \
            'each of the training example in train_x needs to be a networkX graph!'
        self.likelihood: float = likelihood
        self.kernels: list[Union[WeisfilerLehman, MultiscaleLaplacian]] = kernels

        self.n_kernels: int = len(kernels) # カーネルの数
        self.n_graph_kernels: int = len([i for i in kernels if isinstance(i, GraphKernels)]) # グラフカーネルの数
        self.n_vector_kernels: int = self.n_kernels - self.n_graph_kernels # ベクトルカーネルの数
        if self.n_vector_kernels > 1:
            raise NotImplementedError

        self.x: list[nx.DiGraph] = train_x[:]
        self.feature_d = None # 特徴の次元？
        self.vectorial_features = vectorial_features
        if self.n_vector_kernels > 0:
            self.x_features, self.x_features_min, self.x_features_max = \
                standardize_x(self._get_vectorial_features(self.x, vectorial_features))
        else:
            self.x_features, self.x_features_min, self.x_features_max = [None] * 3
        self.n: int = len(self.x) # 教師データサイズ
        self.y_: torch.Tensor = deepcopy(train_y) # 正規化する前のy
        self.y, self.y_mean, self.y_std = normalize_y(train_y)

        if weights is not None:
            self.fixed_weights: bool = True
            if weights is not None:
                assert len(weights) == len(
                    kernels), "the weights vector, if supplied, needs to have the same length as " \
                              "the number of kernels!"
            self.weights = weights if isinstance(weights, torch.Tensor) else torch.tensor(weights).flatten()
        else:
            self.fixed_weights: bool = False
            # Initialise the kernel weights to uniform
            self.weights: torch.Tensor = torch.tensor([1. / len(kernels)] * len(kernels), )
        self.weights = self.weights  # .double()
        self.sum_kernels: SumKernel = SumKernel(*kernels, weights=self.weights)
        self.vector_theta_bounds: tuple = vector_theta_bounds
        self.graph_theta_bounds: tuple = graph_theta_bounds
        # Verbose mode
        self.verbose: bool = verbose

        # Cache the Gram matrix inverse and its log-determinant
        self.K, self.K_i, logDetK = [None] * 3

    def _get_vectorial_features(self, x: list[nx.DiGraph], selected_features: list = None) -> torch.Tensor:
        """
        Return a list of (selected) vectorial features with the vector length being the same as the number of graphs
        in train_x.
        Return: a tensor of features with dimension N x2 S, where N is the number of training graphs and S is the number
        of features extracted
        """
        if not selected_features:
            return None
        self.feature_d = len(selected_features)
        res = torch.zeros(len(x), len(selected_features))
        i = 0
        for feature in selected_features:
            res[:, i] = torch.tensor([getattr(FeatureExtractor(x_), feature) for x_ in x])
            i += 1
        return res

    def _optimize_graph_kernels(self, h_: tuple[int, ...], lengthscale_: tuple[float, ...]):
        for k in self.sum_kernels.kernels:
            if isinstance(k, WeisfilerLehman):
                _grid_search_wl_kernel(k, h_, self.x, self.y, self.likelihood,
                                       lengthscales=lengthscale_, )
            else:
                logging.warning('Graph kernel optimisation for ' + str(k) + " not implemented yet.")

    def fit(self, 
            iters: int = 20, 
            optimizer: str = 'adam',
            wl_subtree_candidates: tuple[int, ...] = tuple(range(5)),
            wl_lengthscales: tuple[float, ...] = tuple([np.e ** i for i in range(-2, 3)]),
            optimize_lik: bool = True, 
            max_lik: float = 0.01,
            optimize_wl_layer_weights: bool = False,
            optimizer_kwargs = None) -> None:
        """

        Parameters
        ----------
        iters
        optimizer
        wl_subtree_candidates
        wl_lengthscales
        optimize_lik
        max_lik
        optimize_wl_layer_weights
        optimizer_kwargs

        Returns
        -------

        """
        # Get the node weights, if needed

        if optimizer_kwargs is None:
            optimizer_kwargs = {
                'lr': 0.1
            }
        if len(wl_subtree_candidates):
            self._optimize_graph_kernels(wl_subtree_candidates, wl_lengthscales, )

        weights: torch.Tensor = self.weights.clone()

        if (not self.fixed_weights) and len(self.kernels) > 1:
            weights.requires_grad_(True)
        # Initialise the lengthscale to be the geometric mean of the theta_vector bounds.
        theta_vector: Optional[torch.Tensor] = torch.sqrt(
            torch.tensor([self.vector_theta_bounds[0] * self.vector_theta_bounds[1]] * self.feature_d,
                         )) if \
            self.feature_d else None
        # theta_graph = torch.sqrt(torch.tensor([self.graph_theta_bounds[0] * self.graph_theta_bounds[1]])).\
        #     requires_grad_(True)
        # Only requires gradient of lengthscale if there is any vectorial input
        if self.feature_d: theta_vector.requires_grad_(True)
        # Whether to include the likelihood (jitter or noise variance) as a hyperparameter
        likelihood: torch.Tensor = torch.tensor(self.likelihood, )
        if optimize_lik:
            likelihood.requires_grad_(True)

        layer_weights = None
        if optimize_wl_layer_weights:
            for k in self.kernels:
                if isinstance(k, WeisfilerLehman):
                    layer_weights = torch.ones(k.h + 1).requires_grad_(True)
                    if layer_weights.shape[0] <= 1:
                        layer_weights = None
                    else:
                        break

        K: Optional[torch.Tensor]
        K_i: torch.Tensor
        logDetK: torch.Tensor
        optim: Union[torch.optim.Adam, torch.optim.SGD]

        # Linking the optimizer variables to the sum kernel
        optim_vars = []
        for a in [theta_vector, weights, likelihood, layer_weights]:
            if a is not None and a.is_leaf and a.requires_grad:
                optim_vars.append(a)
        nlml = None
        if len(optim_vars) == 0:  # Skip optimisation
            K = self.sum_kernels.fit_transform(weights, self.x, self.x_features, theta_vector,
                                               layer_weights=layer_weights,
                                               rebuild_model=True)
            K_i, logDetK = compute_pd_inverse(K, likelihood)
        else:
            # Select the optimizer
            assert optimizer.lower() in ['adam', 'sgd']
            if optimizer.lower() == 'adam':
                optim = torch.optim.Adam(optim_vars, **optimizer_kwargs)
            else:
                optim = torch.optim.SGD(optim_vars, **optimizer_kwargs)

            K = None
            for i in range(iters):
                optim.zero_grad()
                K = self.sum_kernels.fit_transform(weights, self.x, self.x_features,
                                                   feature_lengthscale=theta_vector,
                                                   layer_weights=layer_weights,
                                                   rebuild_model=True,
                                                   save_gram_matrix=True, )
                K_i, logDetK = compute_pd_inverse(K, likelihood)
                nlml = -compute_log_marginal_likelihood(K_i, logDetK, self.y)
                nlml.backward(create_graph=True)
                if self.verbose and i % 10 == 0:
                    print('Iteration:', i, "/", iters, 'Negative log-marginal likelihood:', nlml.item(), theta_vector,
                          weights,  # theta_graph,
                          likelihood)
                optim.step()
                with torch.no_grad():
                    weights.clamp_(0., 1.) if weights is not None and weights.is_leaf else None
                    theta_vector.clamp_(self.vector_theta_bounds[0],
                                        self.vector_theta_bounds[
                                            1]) if theta_vector is not None and theta_vector.is_leaf else None
                    likelihood.clamp_(1e-5, max_lik) if likelihood is not None and likelihood.is_leaf else None
                    layer_weights.clamp_(0., 1.) if layer_weights is not None and layer_weights.is_leaf else None
                # print('grad,', theta_graph)
            K_i, logDetK = compute_pd_inverse(K, likelihood)
        # Apply the optimal hyperparameters
        self.weights: torch.Tensor = weights.clone() / torch.sum(weights)
        self.K_i: Optional[torch.Tensor] = K_i.clone()
        self.K: Optional[torch.Tensor] = K.clone()
        self.logDetK: Optional[torch.Tensor] = logDetK.clone()
        self.likelihood: float = likelihood.item()
        self.theta_vector: Optional[torch.Tensor] = theta_vector
        self.layer_weights: Optional[torch.Tensor] = layer_weights
        self.nlml: torch.Tensor = nlml.detach().cpu() if nlml is not None else None

        for k in self.sum_kernels.kernels:
            if isinstance(k, Stationary):
                k.lengthscale = theta_vector.clamp(self.vector_theta_bounds[0], self.vector_theta_bounds[1])
            # elif isinstance(k, GraphKernels) and k.lengthscale_ is not None:
            #    k.lengthscale_ = theta_graph.clamp(self.graph_theta_bounds[0], self.graph_theta_bounds[1])
        self.sum_kernels.weights = weights.clone()
        if self.verbose:
            print('Optimisation summary: ')
            print('Optimal NLML: ', nlml)
            print('Lengthscales: ', theta_vector)
            try:
                print('Optimal h: ', self.kernels[0]._h)
            except AttributeError:
                pass
            print('Weights: ', self.weights)
            print('Lik:', self.likelihood)
            print('Optimal layer weights', layer_weights)
        # print('Graph Lengthscale', theta_graph)

    def predict(self, X_s: Union[nx.DiGraph, list[nx.DiGraph]], preserve_comp_graph: bool=False) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Kriging predictions
        
        Returns
        -------
        (mu, cov)
        mu.shape: (len(X_s),)
        cov.shape: (len(X_s), len(X_s))
        """
        if not isinstance(X_s, list):
            # Convert a single input X_s to a singleton list
            X_s = [X_s]
        if self.K_i is None or self.logDetK is None:
            raise ValueError("Inverse of Gram matrix is not instantiated. Please call the optimize function to "
                             "fit on the training data first!")
        if self.n_vector_kernels:
            X_s_features: torch.Tensor = self._get_vectorial_features(X_s, self.vectorial_features)
            X_s_features, _, _ = standardize_x(X_s_features, self.x_features_min, self.x_features_max)
        else:
            X_s_features = None

        X_s: list[nx.DiGraph]

        # Concatenate the full list
        X_all: list[nx.DiGraph] = self.x + X_s
        X_features_all: Optional[torch.Tensor]
        # print(X_s_features)
        if self.x_features is not None:
            # print(self.x_features.shape, X_s_features.shape)
            X_features_all = torch.cat([self.x_features, X_s_features])
        else:
            X_features_all = None

        # Make a copy of the sum_kernels for this step, to avoid breaking the autodiff if grad guided mutation is used
        if preserve_comp_graph:
            sum_kernel_copy: SumKernel = deepcopy(self.sum_kernels)
        else:
            sum_kernel_copy = self.sum_kernels
        K_full: torch.Tensor = sum_kernel_copy.fit_transform(self.weights, X_all, X_features_all, self.theta_vector,
                                               layer_weights=self.layer_weights,
                                               rebuild_model=True, save_gram_matrix=False)
        K_s: torch.Tensor = K_full[:len(self.x):, len(self.x):]
        K_ss: torch.Tensor = K_full[len(self.x):, len(self.x):] + self.likelihood * torch.eye(len(X_s), )

        mu_s: torch.Tensor = K_s.t() @ self.K_i @ self.y
        cov_s: torch.Tensor = K_ss - K_s.t() @ self.K_i @ K_s
        cov_s: torch.Tensor = torch.clamp(cov_s, self.likelihood, np.inf)
        mu_s: torch.Tensor = unnormalize_y(mu_s, self.y_mean, self.y_std)
        std_s: torch.Tensor = torch.sqrt(cov_s)
        std_s: torch.Tensor = unnormalize_y(std_s, None, self.y_std, True)
        cov_s: torch.Tensor = std_s ** 2
        if preserve_comp_graph:
            del sum_kernel_copy
        return mu_s, cov_s

    def reset_XY(self, train_x: list[nx.DiGraph], train_y: torch.Tensor) -> None:
        '''
        xとyおよびそれに付随する変数を初期化
        '''
        self.x: list[nx.DiGraph] = train_x
        self.n = len(self.x)
        self.y_: torch.Tensor = train_y
        self.y, self.y_mean, self.y_std = normalize_y(train_y)
        # The Gram matrix of the training data
        self.K_i, self.logDetK = None, None
        if self.n_vector_kernels > 0:
            self.x_features, self.x_features_min, self.x_features_max = \
                standardize_x(self._get_vectorial_features(self.x, self.vectorial_features))

    def dmu_dphi(self, X_s=None,
                 # compute_grad_var=False,
                 average_across_features: bool=True,
                 average_across_occurrencess: bool=False):
        print("GraphGP::dmu_dphi called.");exit()#
        """
        Compute the derivative of the GP posterior mean at the specified input location with respect to the
        *vector embedding* of the graph (e.g., if using WL-subtree, this function computes the gradient wrt
        each subtree pattern)

        The derivative is given by
        $
        \frac{\partial \mu^*}{\partial \phi ^*} = \frac{\partial K(\phi, \phi^*)}{\partial \phi ^ *}K(\phi, \phi)^{-1}
        \mathbf{y}
        $

        which derives directly from the GP posterior mean formula, and since the term $K(\phi, \phi)^{-1} and \mathbf{y}
        are both independent of the testing points (X_s, or \phi^*}, the posterior gradient is simply the matrix
        produce of the kernel gradient with the inverse Gram and the training label vector.

        Parameters
        ----------
        X_s: The locations on which the GP posterior mean derivatives should be evaluated. If left blank, the
        derivatives will be evaluated at the training points.

        compute_grad_var: bool. If true, also compute the gradient variance.

        The derivative of GP is also a GP, and thus the predictive distribution of the posterior gradient is Gaussian.
        The posterior mean is given above, and the posterior variance is:
        $
        \mathbb{V}[\frac{\partial f^*}{\partial \phi^*}]= \frac{\partial^2k(\phi^*, \phi^*)}{\partial \phi^*^2} -
        \frac{\partial k(\phi^*, \Phi)}{\partial \phi^*}K(X, X)^{-1}\frac{\partial k{(\Phi, \phi^*)}}{\partial \phi^*}
        $

        Returns
        -------
        list of K torch.Tensor of the shape N x2 D, where N is the length of the X_s list (each element of which is a
        networkx graph), K is the number of kernels in the combined kernel and D is the dimensionality of the
        feature vector (this is determined by the specific graph kernel.

        OR

        list of K torch.Tensor of shape D, if averaged_over_samples flag is enabled.
        """
        if self.K_i is None or self.logDetK is None:
            raise ValueError("Inverse of Gram matrix is not instantiated. Please call the optimize function to "
                             "fit on the training data first!")
        if self.n_vector_kernels:
            if X_s is not None:
                V_s: torch.Tensor = self._get_vectorial_features(X_s, self.vectorial_features)
                V_s, _, _ = standardize_x(V_s, self.x_features_min, self.x_features_max)
            else:
                V_s = self.x_features
                X_s = self.x[:]
        else:
            V_s = None
            X_s = X_s if X_s is not None else self.x[:]

        alpha = (self.K_i @ self.y).double().reshape(1, -1)
        dmu_dphi = []
        # dmu_dphi_var = [] if compute_grad_var else None

        Ks_handles = []
        feature_matrix = []
        for j, x_s in enumerate(X_s):
            jacob_vecs = []
            if V_s is None:
                handles = self.sum_kernels.forward_t(self.weights, [x_s], )
            else:
                handles = self.sum_kernels.forward_t(self.weights, [x_s], V_s[j])
            Ks_handles.append(handles)
            # Each handle is a 2-tuple. first element is the Gram matrix, second element is the leaf variable
            feature_vectors = []
            for handle in handles:
                k_s, y, _ = handle
                # k_s is output, leaf is input, alpha is the K_i @ y term which is constant.
                # When compute_grad_var is not required, computational graphs do not need to be saved.
                jacob_vecs.append(torch.autograd.grad(outputs=k_s, inputs=y, grad_outputs=alpha, retain_graph=False)[0])
                feature_vectors.append(y)
            feature_matrix.append(feature_vectors)
            jacob_vecs = torch.cat(jacob_vecs)
            dmu_dphi.append(jacob_vecs)
        # dmu_dphi is of shape N_t x K x D (or N_t x D if K is 1)

        # if compute_grad_var:
        #     for j, x_s in enumerate(X_s):
        #         if V_s is None:
        #             handles = self.sum_kernels.forward_t(self.weights, [x_s], gr1=[x_s])
        #         else:
        #             handles = self.sum_kernels.forward_t(self.weights, [x_s], V_s[j], gr1=[x_s], x1=V_s[j])
        #
        #         vars = []
        #         for handle in handles:
        #             k_ss, x2, x1 = handle
        #             # compute the second derivative of k_ss w.r.t the leaf variables (d^2k/dx1dx2)
        #
        #             # first compute dk_ss / dx1
        #             dkss_dx1 = gradient(k_ss[0][0], x1)
        #
        #             # Then differentiate it again wrt x2
        #             dk2ss_dx1dx2 = jacobian(dkss_dx1, x2)
        #
        #             # Extract the diagonal elements of the first term
        #             vars.append(torch.diag(dk2ss_dx1dx2))
        #
        #         for k, handle in enumerate(Ks_handles[j]):
        #             k_s, x2, _ = handle
        #             dks_dx1 = jacobian(k_s, x2)
        #             tmp = dks_dx1.t() @ self.K_i.double() @ dks_dx1
        #             # Since the first term (d^2k/dx1dx2 is the Gram matrix wrt the test samples themselves, it is
        #             # possible that there are features not seen in the training set, leading to a longer feature
        #             # vector with new unseen features concatenated at the end. Use this operation to remove these
        #             # features
        #             if vars[k].shape != tmp.shape[0]:
        #                 vars[k] = vars[k][:tmp.shape[0]]
        #             vars[k] -= torch.diag(tmp)
        #             vars[k] = vars[k].reshape(1, -1)
        #
        #         vars = torch.cat(vars).clamp_min_(1e-5)
        #         dmu_dphi_var.append(vars)

        # dmu_dphi_var is of shape N_t x K x D (or N_t x D if K is 1)
        feature_matrix = torch.cat([f[0] for f in feature_matrix])
        if average_across_features:
            dmu_dphi = torch.cat(dmu_dphi)
            # compute the weighted average of the gradient across N_t.
            # feature matrix is of shape N_t x K x D
            avg_mu, avg_var, incidences = get_grad(dmu_dphi, feature_matrix, average_across_occurrences)
            return avg_mu, avg_var, incidences
        return dmu_dphi, None, feature_matrix.sum(dim=0) if average_across_occurrences else feature_matrix

# オリジナル
class BaggingGraphGP(GraphGP):
    def __init__(self, 
                 train_x: list[nx.DiGraph], 
                 train_y: torch.Tensor,
                 kernels: list[Union[WeisfilerLehman, MultiscaleLaplacian]],
                 vectorial_features: Optional[list] = None,
                 likelihood: float=1e-3,
                 weights=None,
                 vector_theta_bounds: tuple = (1e-5, 0.1),
                 graph_theta_bounds: tuple = (1e-1, 1.e1),
                 verbose: bool=False,
                 bagging_method: str = "random_overlap",
                 train_size_max: int = 50
                 ):
        super().__init__(
            deepcopy(train_x),
            deepcopy(train_y),
            deepcopy(kernels),
            deepcopy(vectorial_features),
            likelihood,
            deepcopy(weights),
            deepcopy(vector_theta_bounds),
            deepcopy(graph_theta_bounds),
            verbose,
        )
        
        assert bagging_method in ['random_exclusive', 'random_overlap']
        self.bagging_method = bagging_method
        self.train_size_max = train_size_max
        
        self.n_children: int = self._decide_num_of_children(self.n)
        self.gp_children: list[GraphGP] = []
        
        children_train_x, children_train_y = self._separate_train_data(train_x, train_y)
            
        for i in range(self.n_children):
            self.gp_children.append(
                GraphGP(
                    deepcopy(children_train_x[i]),
                    deepcopy(children_train_y[i]),
                    deepcopy(kernels),
                    deepcopy(vectorial_features),
                    likelihood,
                    deepcopy(weights),
                    deepcopy(vector_theta_bounds),
                    deepcopy(graph_theta_bounds),
                    verbose,
                )
            )
        
    # オリジナル関数
    def _decide_num_of_children(self, data_size: int) -> int:
        if self.bagging_method == "random_exclusive":
            return math.ceil(data_size / self.train_size_max)
        elif self.bagging_method == "random_overlap":
            if data_size > self.train_size_max:
                return math.ceil(math.log(1 - 0.99) / math.log(1 - self.train_size_max / data_size))
            else:
                return 1
        else:
            raise NotImplementedError

    # オリジナル関数
    def _separate_train_data(self, train_x: list[nx.DiGraph], train_y: torch.Tensor) -> tuple[list[list[nx.DiGraph]], list[torch.Tensor]]:
        
        n = len(train_x)
        num = self.n_children
        
        gp_children_indices: list[list[int]] = [None] * num

        if self.bagging_method == "random_exclusive":
            shuffled_indices: list[int] = list(range(n))
            random.shuffle(shuffled_indices)
            
            for i in range(num):
                gp_children_indices[i] = sorted(shuffled_indices[n * i // num: n * (i + 1) // num])
    
        elif self.bagging_method == "random_overlap":
            for i in range(num):
                shuffled_indices: list[int] = list(range(n))
                random.shuffle(shuffled_indices)
                gp_children_indices[i] = sorted(shuffled_indices[:self.train_size_max])
        
        else:
            raise NotImplementedError
        
        children_train_x: list[list[nx.DiGraph]] = []
        children_train_y: list[torch.Tensor] = []
        
        for i in range(num):
            child_train_x: list[nx.DiGraph] = []
            child_train_y: list[float] = []
            for j in gp_children_indices[i]:
                child_train_x.append(deepcopy(train_x[j]))
                child_train_y.append(deepcopy(train_y[j]))
            children_train_x.append(child_train_x)
            children_train_y.append(torch.Tensor(child_train_y))
        return children_train_x, children_train_y

    def fit(self, 
            iters: int = 20, 
            optimizer: str = 'adam',
            wl_subtree_candidates: tuple[int, ...] = tuple(range(5)),
            wl_lengthscales: tuple[float, ...] = tuple([np.e ** i for i in range(-2, 3)]),
            optimize_lik: bool = True, 
            max_lik: float = 0.01,
            optimize_wl_layer_weights: bool = False,
            optimizer_kwargs = None) -> None:
        """

        Parameters
        ----------
        iters
        optimizer
        wl_subtree_candidates
        wl_lengthscales
        optimize_lik
        max_lik
        optimize_wl_layer_weights
        optimizer_kwargs

        Returns
        -------

        """
        
        super().fit(
            iters, 
            optimizer,
            deepcopy(wl_subtree_candidates),
            deepcopy(wl_lengthscales),
            optimize_lik, 
            max_lik,
            optimize_wl_layer_weights,
            deepcopy(optimizer_kwargs)
        )

        for child in self.gp_children:
            child.fit(
                iters, 
                optimizer,
                deepcopy(wl_subtree_candidates),
                deepcopy(wl_lengthscales),
                optimize_lik, 
                max_lik,
                optimize_wl_layer_weights,
                deepcopy(optimizer_kwargs)
            )
            
        new_likelihood: float = statistics.mean([child.likelihood for child in self.gp_children])
        if abs(self.likelihood - new_likelihood) > 0.00001:
            print(self.likelihood, new_likelihood)
            exit()
        self.likelihood = new_likelihood

    def predict(self, X_s: Union[nx.DiGraph, list[nx.DiGraph]], preserve_comp_graph: bool=False) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Kriging predictions
        
        Returns
        -------
        (mu, cov)
        mu.shape: (len(X_s),)
        cov.shape: (len(X_s), len(X_s))
        """
        
        if not isinstance(X_s, list):
            # Convert a single input X_s to a singleton list
            X_s = [X_s]
        
        mu_list: list[torch.Tensor] = []
        cov_list: list[torch.Tensor] = []
        mu: torch.Tensor; cov: torch.Tensor
        for child in self.gp_children:
            mu, cov = child.predict(deepcopy(X_s), preserve_comp_graph)
            mu_list.append(mu)
            cov_list.append(cov)
        
        mu0 = torch.empty([len(X_s)])
        cov0 = torch.empty([len(X_s), len(X_s)])
        for i in range(len(X_s)):
            mu0[i] = statistics.median([mu_list[k][i] for k in range(self.n_children)])
            for j in range(len(X_s)):
                cov0[i][j] = statistics.median([cov_list[k][i][j] for k in range(self.n_children)])               
                
        # デバッグ
        '''
        parent_mu, parent_cov = super().predict(deepcopy(X_s), preserve_comp_graph)
        if torch.norm(mu - parent_mu) > 0.1 or torch.norm(cov - parent_cov) > 0.1:
            print('@' * 8)
            print(f'self.n = {self.n}')
            print(f'len(self.gp_children) = {len(self.gp_children)}')
            for i, child in enumerate(self.gp_children):
                child_mu, child_cov = child.predict(deepcopy(X_s), preserve_comp_graph)
                print(f'bagg {i} mu ({child_mu.shape}) =\n{child_mu}')
                print(f'bagg {i} cov ({child_cov.shape}) =\n{child_cov}')
            print(f'bagg mu ({mu.shape}) =\n{mu}')
            print(f'true mu ({parent_mu.shape}) =\n{parent_mu}')
            print(f'bagg cov ({cov.shape}) =\n{cov}')
            print(f'true cov ({parent_cov.shape}) =\n{parent_cov}')
            exit()  
        '''

        return mu0, cov0

    def reset_XY(self, train_x: list[nx.DiGraph], train_y: torch.Tensor) -> None:
        '''
        xとyおよびそれに付随する変数を初期化
        '''
        
        super().reset_XY(deepcopy(train_x), deepcopy(train_y))
        
        old_n_children = self.n_children
        self.n_children: int = self._decide_num_of_children(self.n)
        children_train_x, children_train_y = self._separate_train_data(train_x, train_y)
        
        for i in range(self.n_children):
            if i >= old_n_children:
                for i in range(self.n_children - old_n_children):
                    self.gp_children.append(GraphGP(
                        deepcopy(children_train_x[i]),
                        deepcopy(children_train_y[i]),
                        deepcopy(self.kernels),
                        deepcopy(self.vectorial_features),
                        self.likelihood,
                        deepcopy(self.weights),
                        deepcopy(self.vector_theta_bounds),
                        deepcopy(self.graph_theta_bounds),
                        self.verbose
                    ))
            else:
                self.gp_children[i].reset_XY(deepcopy(children_train_x[i]), deepcopy(children_train_y[i]))

    def dmu_dphi(self, X_s=None,
                 # compute_grad_var=False,
                 average_across_features: bool=True,
                 average_across_occurrencess: bool=False):
        return super().dmu_dphi(X_s, average_across_features, average_across_occurrencess)

def get_grad(grad_matrix, feature_matrix, average_occurrences=False):
    """
    Average across the samples via a Monte Carlo sampling scheme. Also estimates the empirical variance.
    :param average_occurrences: if True, do a weighted summation based on the frequency distribution of the occurrence
        to compute a gradient *per each feature*. Otherwise, each different occurence (\phi_i = k) will get a different
        gradient estimate.
    """
    assert grad_matrix.shape == feature_matrix.shape
    # Prune out the all-zero columns that pop up sometimes
    valid_cols = []
    for col_idx in range(feature_matrix.size(1)):
        if not torch.all(feature_matrix[:, col_idx] == 0):
            valid_cols.append(col_idx)
    feature_matrix = feature_matrix[:, valid_cols]
    grad_matrix = grad_matrix[:, valid_cols]

    N, D = feature_matrix.shape
    if average_occurrences:
        avg_grad = torch.zeros(D)
        avg_grad_var = torch.zeros(D)
        for d in range(D):
            current_feature = feature_matrix[:, d].clone().detach()
            instances, indices, counts = torch.unique(current_feature, return_inverse=True, return_counts=True)
            weight_vector = torch.tensor([counts[i] for i in indices]).type(torch.float)
            weight_vector /= weight_vector.sum()
            mean = torch.sum(weight_vector * grad_matrix[:, d])
            # Compute the empirical variance of gradients
            variance = torch.sum(weight_vector * grad_matrix[:, d] ** 2) - mean ** 2
            avg_grad[d] = mean
            avg_grad_var[d] = variance
        return avg_grad, avg_grad_var, feature_matrix.sum(dim=0)
    else:
        # The maximum number possible occurrences -- 7 is an example, if problem occurs, maybe we can increase this
        # number. But for now, for both NAS-Bench datasets, this should be more than enough!
        max_occur = 7
        avg_grad = torch.zeros(D, max_occur)
        avg_grad_var = torch.zeros(D, max_occur)
        incidences = torch.zeros(D, max_occur)
        for d in range(D):
            current_feature = feature_matrix[:, d].clone().detach()
            instances, indices, counts = torch.unique(current_feature, return_inverse=True,
                                                      return_counts=True)
            for i, val in enumerate(instances):
                # Find index of all feature counts that are equal to the current val
                feature_at_val = grad_matrix[current_feature == val]
                avg_grad[d, int(val)] = torch.mean(feature_at_val)
                avg_grad_var[d, int(val)] = torch.var(feature_at_val)
                incidences[d, int(val)] = counts[i]
        return avg_grad, avg_grad_var, incidences


# Optimize Graph kernel
def getBack(var_grad_fn):
    print(var_grad_fn)
    for n in var_grad_fn.next_functions:
        if n[0]:
            try:
                tensor = getattr(n[0], 'variable')
                print(n[0])
                print('Tensor with grad found:', tensor)
                print(' - gradient:', tensor.grad)
                print()
            except AttributeError as e:
                getBack(n[0])


def _grid_search_wl_kernel(k: WeisfilerLehman,
                           subtree_candidates: tuple[int, ...],
                           train_x: list[nx.DiGraph],
                           train_y: torch.Tensor,
                           lik: float,
                           subtree_prior=None,
                           lengthscales: Optional[tuple[float, ...]]=None,
                           lengthscales_prior=None):
    """Optimize the *discrete hyperparameters* of Weisfeiler Lehman kernel.
    k: a Weisfeiler-Lehman kernel instance
    hyperparameter_candidate: list of candidate hyperparameter to try
    train_x: the train data (array like)
    train_y: the train label (array like)
    lik: likelihood
    lengthscale: if using RBF kernel for successive embedding, the list of lengthscale to be grid searched over
    """
    # lik = 1e-6
    assert len(train_x) == len(train_y)
    
    best_nlml = torch.tensor(np.inf)
    best_subtree_depth = None
    best_lengthscale = None
    best_K = None
    if lengthscales is not None and k.se is not None:
        candidates = [(h_, l_) for h_ in subtree_candidates for l_ in lengthscales]
    else:
        candidates = [(h_, None) for h_ in subtree_candidates]

    for i in candidates:
        if k.se is not None:
            k.change_se_params({'lengthscale': i[1]})
        k.change_kernel_params({'h': i[0]})
        K = k.fit_transform(train_x, rebuild_model=True, save_gram_matrix=True)
        # print(K)
        K_i, logDetK = compute_pd_inverse(K, lik)
        # print(train_y)
        nlml = -compute_log_marginal_likelihood(K_i, logDetK, train_y)
        # print(i, nlml)
        if nlml < best_nlml:
            best_nlml = nlml
            best_subtree_depth, best_lengthscale = i
            best_K = torch.clone(K)
    # print("h: ", best_subtree_depth, "theta: ", best_lengthscale)
    # print(best_subtree_depth)
    k.change_kernel_params({'h': best_subtree_depth})
    if k.se is not None:
        k.change_se_params({'lengthscale': best_lengthscale})
    k._gram = best_K
