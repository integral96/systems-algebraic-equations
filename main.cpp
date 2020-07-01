#include <iostream>
#include <algorithm>
#include "solver_algebrick.hpp"
#include <boost/coroutine/all.hpp>
#include <boost/bind.hpp>

template<typename T>
using pull_type = typename boost::coroutines::asymmetric_coroutine<T>::pull_type;

template<typename T>
using push_type = typename boost::coroutines::asymmetric_coroutine<T>::push_type;

int main()
{
//    gen_rand_matrix<int>("matrix_out1.txt", 10, 11);
    Solver_Algebrick<double> a("matrix.txt");
    std::vector<std::thread> threads;
    threads.push_back(std::thread([&](){ return a.solver_gauss(); }));
    threads.push_back(std::thread([&](){ return a.solver_LU(); }));
    for(std::thread& th : threads) {
        if(th.joinable())
            th.join();
    }
    return 0;
}
