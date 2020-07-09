#include <iostream>
#include <algorithm>
#include "solver_algebrick.hpp"
#include "meta_solver_algebrick.hpp"

#include <boost/coroutine/all.hpp>
#include <boost/bind.hpp>


int main()
{
//    gen_rand_matrix<int>("matrix_out1.txt", 10, 11);
    Solver_Algebrick<double> a("matrix_out1.txt");
    std::vector<std::thread> threads;
    threads.push_back(std::thread([&](){ return a.solve_Meta_LU();}));
    threads.push_back(std::thread([&](){ return a.solver_LU(); }));
    threads.push_back(std::thread([&](){ return a.solver_gauss(); }));
    for(std::thread& th : threads) {
        if(th.joinable())
            th.join();
    }
    // solve_Meta_LU("matrix_out1.txt");
    return 0;
}
