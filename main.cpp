#include <iostream>
#include <algorithm>
#include "solver_algebrick.hpp"

using namespace std;

int main()
{
   gen_rand_matrix<double>("matrix_out1.txt", 10, 11);
    Solver_Algebrick<double> a("matrix_out1.txt");
    std::vector<std::thread> threads;
    threads.push_back(std::thread([&](){ return a.init_matrix(); }));
    threads.push_back(std::thread([&](){ return a.print_matrix(); }));
   threads.push_back(std::thread([&](){ return a.triang_matrix(); }));
//    threads.push_back(std::thread([&](){ return a.print_matrix(); }));
//    threads.push_back(std::thread([&](){ return a.solver_gauss(); }));
    threads.push_back(std::thread([&](){ return a.solver_LU(); }));
    threads.push_back(std::thread([&](){ return a.print_vector(); }));
    std::reverse(threads.begin(), threads.end());
    for(std::thread& th : threads) {
        if(th.joinable())
            th.join();
    }
    return 0;
}
