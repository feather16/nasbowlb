#include "wl_kernel_impl.hpp"

#include <iostream>
#include <string>
#include <unordered_map>
#include <algorithm>

using namespace std;

void natsbench_add_cell_labels(vector<string>& cell_labels, const int h){
    const string LABEL_MAX_S = "g";
    const vector<vector<int>> NEXT_NODES = {
        {1, 2, 4},
        {3, 5},
        {6},
        {6},
        {7},
        {7},
        {7}
    };
    const int N_NODES = NEXT_NODES.size() + 1;
    constexpr int N_NEXT_NODES[] = {3, 2, 1, 1, 1, 1, 1};
    const int s_index = h > 0 ? (N_NODES - 1) * h + 1 : 0;

    for(int i = 0; i < N_NODES - 1; i++){
        vector<string> nb_labels;
        nb_labels.reserve(N_NEXT_NODES[i]);
        for(const int j : NEXT_NODES[i]){
            nb_labels.push_back(j != N_NODES - 1 ? cell_labels[s_index + j] : LABEL_MAX_S);
        }
        sort(nb_labels.begin(), nb_labels.end());
        string tmp = cell_labels[s_index + i] + (char)('0' + N_NEXT_NODES[i]) + "-";
        for(int k = 0; k < (int)nb_labels.size(); k++){
            tmp += nb_labels[k];
        }
        cell_labels.push_back(tmp);
    }
}

std::vector<std::string> cell_to_wl_kernel_vector(const std::vector<int>& cell, const int H){
    constexpr int N_NODES = 8;
    vector<string> cell_labels;
    cell_labels.reserve((N_NODES - 1) * (H + 1) + 1);
    for(int i = 0; i < N_NODES; i++){
        cell_labels.push_back({(char)(cell[i] + 'a')});
    }
    for(int h = 0; h < H; h++){
        natsbench_add_cell_labels(cell_labels, h);
    }
    return cell_labels;
}

std::unordered_map<std::string, int> cell_to_wl_kernel_counter(const std::vector<int>& cell, const int H){
    vector<string> cell_labels = cell_to_wl_kernel_vector(cell, H);
    unordered_map<string, int> ret;
    for(const string& cell_label : cell_labels){
        const auto itr = ret.find(cell_label);
        if(itr != ret.end()){ // ある場合
            itr->second++;
        }
        else{ // ない場合
            ret[cell_label] = 1;
        }
    }
    return ret;
}

int natsbench_wl_kernel_from_wl_counters(
        const std::unordered_map<std::string, int>& counter1, 
        const std::unordered_map<std::string, int>& counter2){
    int ret = 0;
    for(auto itr = counter1.begin(); itr != counter1.end(); ++itr){
        const auto itr2 = counter2.find(itr->first);
        if(itr2 != counter2.end()){ // ある場合
            ret += itr->second * itr2->second;
        }
    }
    return ret;
}

// C++で書かれたWLカーネル
int natsbench_wl_kernel(
        const std::vector<int>& cell1, 
        const std::vector<int>& cell2,
        const int H
        ){
    vector<string> cell_labels2 = cell_to_wl_kernel_vector(cell2, H);

    int ret = 0;
    unordered_map<string, int> counter1 = cell_to_wl_kernel_counter(cell1, H);
    for(const string& cell_label : cell_labels2){
        const auto itr = counter1.find(cell_label);
        if(itr != counter1.end()){ // ある場合
            ret += itr->second;
        }
    }

    return ret;
}

// 遅いので不採用
// バグがあるかも
std::vector<int> natsbench_wl_kernel_vector(
        const std::vector<int>& cell1, 
        const std::vector<std::vector<int>>& cells, 
        const int H){
    const int NUM_CELLS = cells.size();
    vector<vector<string>> cells_labels(NUM_CELLS);
    for(int i = 0; i < NUM_CELLS; i++){
        cells_labels[i] = cell_to_wl_kernel_vector(cells[i], H);
    }

    vector<int> ret(NUM_CELLS);
    unordered_map<string, int> counter1 = cell_to_wl_kernel_counter(cell1, H);
    for(int c = 0; c < NUM_CELLS; c++){
        for(const string& cell_label : cells_labels[c]){
            const auto itr = counter1.find(cell_label);
            if(itr != counter1.end()){ // ある場合
                ret[c] += itr->second;
            }
        }
    }

    return ret;
}