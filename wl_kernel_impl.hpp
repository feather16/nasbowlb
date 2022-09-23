#ifndef __GUARD
#define __GUARD

#include <vector>
#include <string>
#include <unordered_map>

std::vector<std::string> cell_to_wl_kernel_vector(const std::vector<int>&, const int);
std::unordered_map<std::string, int> cell_to_wl_kernel_counter(const std::vector<int>&, const int);
int natsbench_wl_kernel_from_wl_counters(const std::unordered_map<std::string, int>&, const std::unordered_map<std::string, int>&);
int natsbench_wl_kernel(const std::vector<int>&, const std::vector<int>&, const int);
//std::vector<int> natsbench_wl_kernel_vector(const std::vector<int>&, const std::vector<std::vector<int>>&, const int);

#endif