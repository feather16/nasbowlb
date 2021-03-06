from __future__ import annotations

from abc import ABC

import networkx as nx
import numpy as np
import torch

import bayesopt

from typing import Union

# debug
try:
    import sys
    sys.path.append("/home/rio-hada/workspace/util")
    from debug import debug
except:
    print('# Failed to import debug')

class BaseAcquisition(ABC):
    def __init__(self,
                 gp: bayesopt.GraphGP,
                 ):
        self.gp: bayesopt.GraphGP = gp
        self.iters: int = 0

        # Storage for the current evaluation on the acquisition function
        self.next_location = None
        self.next_acq_value = None

    def propose_location(self, *args):
        """Propose new locations for subsequent sampling
        This method should be overriden by respective acquisition function implementations."""
        raise NotImplementedError

    def optimize(self):
        """This is the method that user should call for the Bayesian optimisation main loop."""
        raise NotImplementedError

    def eval(self, x):
        """Evaluate the acquisition function at point x2. This should be overridden by respective acquisition
        function implementations"""
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.eval(*args, **kwargs)


class GraphExpectedImprovement(BaseAcquisition):
    def __init__(self, surrogate_model: bayesopt.GraphGP, augmented_ei: bool=False, xi: float = 0.0, in_fill: str = 'best'):
        """
        This is the graph BO version of the expected improvement
        key differences are:
        1. The input x2 is a networkx graph instead of a vectorial input
        2. the search space (a collection of x1_graphs) is discrete, so there is no gradient-based optimisation. Instead,
        we compute the EI at all candidate points and empirically select the best position during optimisation

        augmented_ei: Using the Augmented EI heuristic modification to the standard expected improvement algorithm
        according to Huang (2006)
        xi: float: manual exploration-exploitation trade-off parameter.
        in_fill: str: the criterion to be used for in-fill for the determination of mu_star. 'best' means the empirical
        best observation so far (but could be susceptible to noise), 'posterior' means the best *posterior GP mean*
        encountered so far, and is recommended for optimisationn of more noisy functions.
        """
        super(GraphExpectedImprovement, self).__init__(surrogate_model)
        assert in_fill in ['best', 'posterior']
        self.in_fill: str = in_fill
        self.augmented_ei: bool = augmented_ei
        self.xi: float = xi
        #debug(locals(), globals(), exclude_types=['module','function','type'], colored=True);exit()

    def eval(self, x: nx.Graph, asscalar=False):
        """
        Return the negative expected improvement at the query point x2
        """
        from torch.distributions import Normal
        try:
            mu: torch.Tensor # shape: (1,)
            cov: torch.Tensor # shape: (1, 1)
            mu, cov = self.gp.predict(x)
        except:
            return -1.  # in case of error. return ei of -1
        std: torch.Tensor = torch.sqrt(torch.diag(cov)) # shape: (1,)
        mu_star: torch.Tensor = self._get_incumbent() # shape: (1,)
        gauss = Normal(torch.zeros(1, device=mu.device), torch.ones(1, device=mu.device))
        u: torch.Tensor = (mu - mu_star - self.xi) / std # shape: (1,)
        ucdf: torch.Tensor = gauss.cdf(u) # shape: (1,)
        updf: torch.Tensor = torch.exp(gauss.log_prob(u)) # shape: (1,)
        ei: torch.Tensor = std * updf + (mu - mu_star - self.xi) * ucdf # shape: (1,)
        if self.augmented_ei:
            sigma_n = self.gp.likelihood
            ei *= (1. - torch.sqrt(torch.tensor(sigma_n, device=mu.device)) / torch.sqrt(sigma_n + torch.diag(cov)))
        if asscalar:
            ei = ei.detach().numpy().item()
        return ei

    def _get_incumbent(self, ):
        """
        Get the incumbent
        """
        if self.in_fill == 'max':
            return torch.max(self.gp.y_)
        else:
            x = self.gp.x
            mu_train, _ = self.gp.predict(x)
            incumbent_idx = torch.argmax(mu_train)
            return self.gp.y_[incumbent_idx]

    def propose_location(self, candidates: list[nx.DiGraph], top_n: int=5, return_distinct: bool=True):
        """
        top_n: return the top n candidates wrt the acquisition function.
        """
        # selected_idx = [i for i in self.candidate_idx if self.evaluated[i] is False]
        # eis = torch.tensor([self.eval(self.candidates[c]) for c in selected_idx])
        # print(eis)
        self.iters += 1
        eis: Union[np.ndarray, torch.Tensor] # dtype=float
        eis_: np.ndarray # dtype=float
        unique_idx: np.ndarray # dtype=int
        if return_distinct:
            eis = torch.Tensor(([self.eval(c) for c in candidates])).detach().numpy() # eis = np.array([self.eval(c) for c in candidates]) # RuntimeError: Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.
            eis_, unique_idx = np.unique(eis, return_index=True)
            try:
                i = np.argpartition(eis_, -top_n)[-top_n:]
                indices = np.array([unique_idx[j] for j in i])
            except ValueError:
                eis = torch.tensor([self.eval(c) for c in candidates])
                values, indices = eis.topk(top_n)
        else:
            eis = torch.tensor([self.eval(c) for c in candidates])
            values, indices = eis.topk(top_n)
        xs: tuple[nx.DiGraph, ...] = tuple([candidates[int(i)] for i in indices]) # ?????????: top_n
        #debug(locals(), globals(), exclude_types=['module','function','type'], colored=True);exit()
        return xs, eis, indices

    def optimize(self):
        raise ValueError("The kernel invoked does not have hyperparameters to optimise over!")


class GraphUpperConfidentBound(GraphExpectedImprovement):
    """
    Graph version of the upper confidence bound acquisition function
    """

    def __init__(self, surrogate_model: bayesopt.GraphGP, beta=None):
        """Same as graphEI with the difference that a beta coefficient is asked for, as per standard GP-UCB acquisition
        """
        super(GraphUpperConfidentBound, self).__init__(surrogate_model, )
        self.beta = beta

    def eval(self, x: nx.Graph, asscalar=False):
        mu: torch.Tensor
        cov: torch.Tensor
        mu, cov = self.gp.predict(x)
        std: torch.Tensor = torch.sqrt(torch.diag(cov))
        if self.beta is None:
            self.beta = 3. * torch.sqrt(0.5 * torch.log(torch.tensor(2. * self.iters + 1.)))
        acq = mu + self.beta * std
        if asscalar:
            acq = acq.detach().numpy().item()
        return acq
