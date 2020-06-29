#include <iostream>
#include "solver_algebrick.hpp"

using namespace std;

int main()
{
    gen_rand_matrix<double>("matrix_out.txt", 50, 51);
    Solver_Algebrick<double> a("matrix_out.txt");
    std::vector<std::thread> threads;
    threads.push_back(std::thread([&](){ return a.init_matrix(); }));
    threads.push_back(std::thread([&](){ return a.print_matrix(); }));
    threads.push_back(std::thread([&](){ return a.triang_matrix(); }));
    threads.push_back(std::thread([&](){ return a.print_matrix(); }));
    threads.push_back(std::thread([&](){ return a.solver(); }));
    for(std::thread& th : threads) {
        if(th.joinable())
            th.join();
    }
    return 0;
}
