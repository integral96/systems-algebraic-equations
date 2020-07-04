#pragma once

#include <memory>
#include <thread>
#include <mutex>
#include <array>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <ctime>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/spirit/include/karma.hpp>
#include <boost/assign/list_inserter.hpp>
#include <boost/random.hpp>
#include <boost/current_function.hpp>

namespace matrix = boost::numeric::ublas;
namespace krm  = boost::spirit::karma;

template<typename T> struct is_int : boost::mpl::false_ {};
template<> struct is_int<int> : boost::mpl::true_ {};

template<typename T>
void gen_rand_matrix(const std::string& name, size_t N, size_t M) {
    matrix::matrix<T> A(N, M);
    std::time_t now = std::time(0);
    boost::random::mt19937 gen{static_cast<std::uint32_t>(now)};
    if constexpr(is_int<T>::value) {
        boost::random::uniform_int_distribution<> dist{1, 20};
        for(size_t i = 0; i < N; ++i)
            for(size_t j = 0; j < M; ++j)
                A(i, j) = dist(gen);
    } else {
        boost::random::uniform_real_distribution<> dist{1, 10};
        for(size_t i = 0; i < N; ++i)
            for(size_t j = 0; j < M; ++j)
                A(i, j) = dist(gen);
    }
    std::ofstream(name) << N << "\n" << M << "\n" << krm::format_delimited(krm::columns(A.size2()) [krm::auto_], '\t', A.data()) << "\n";
}


template<typename T>
std::ostream& operator << (std::ostream& out, const std::shared_ptr<matrix::matrix<T>>& A) {
    out << std::setprecision(1)  << krm::format_delimited(krm::columns(A->size2()) [krm::auto_], '\t', A->data()) << std::endl;
    return out;
}

template <typename T>
class Solver_Algebrick
{
private:
    std::shared_ptr<matrix::matrix<T>> A;
    std::shared_ptr<matrix::matrix<T>> L, U;

    std::unique_ptr<std::vector<T>> x_1;
    std::unique_ptr<std::vector<T>> v;

    std::weak_ptr<matrix::matrix<T>> weak_matrix;

    std::ifstream file;
    size_t N{}, M{};

    std::mutex mutex;
public:
    Solver_Algebrick(const std::string& name) : file(name) {
        if(!file)
        {
            std::cerr << "Error opening matrix file.\n";
            return;
        }
        file >> N >> M;
        if(N < 1 || M < 1)
        {
            std::cerr << "Matrix sizes are out of bounds.\n";
            return;
        }
        x_1 = std::make_unique<std::vector<T>>(N);
        v = std::make_unique<std::vector<T>>(N);

        A = std::make_shared<matrix::matrix<T>>(N, M);
        U = std::make_shared<matrix::matrix<T>>(N, N);
        L = std::make_shared<matrix::matrix<T>>(N, N);
        init_matrix();
        std::cout << "Матрица А: \n" << A << std::endl;
        
    }
    ~Solver_Algebrick() {
        std::cout << BOOST_CURRENT_FUNCTION << std::endl;
        std::cout << A.use_count() << std::endl;
    }
private: 
    void weak_ptr_info(const std::weak_ptr<matrix::matrix<T>>& p) {
        std::cout << std::boolalpha << "weak_ptr info: " << p.expired() << ". count: " << p.use_count() << std::endl;
        if(const auto sp (p.lock()); sp) {
            std::cout << krm::format_delimited(krm::columns(sp->size2()) [krm::auto_], '\t', sp->data()) << std::endl;
        } else std::cout << "FALSE" << std::endl;
    }
    void inline init_matrix() {
        for(size_t i = 0; i < A->size1(); ++i)
            for(size_t j = 0; j < A->size2(); ++j) file >> A->at_element(i, j);
    }
    const std::shared_ptr<matrix::matrix<T>> triang_matrix(const std::shared_ptr<matrix::matrix<T>>& A) const {
        auto A_copy = A;
        T tmp;
        for(size_t j = 0; j < A_copy->size1() - 1; ++j) {
            tmp = T(0);
            for(size_t i = j + 1; i < A_copy->size2() - 1; ++i) {
                tmp = A_copy->at_element(i, j)/A_copy->at_element(j, j);
                for(size_t k = 0; k < A_copy->size2(); ++k)
                    A_copy->at_element(i, k) -= A_copy->at_element(j, k) * (tmp);
            }
        }
        // weak_matrix = A_copy;
        // weak_ptr_info(weak_matrix);
        return A_copy;     
    }
public:
    void solver_gauss() {
        std::lock_guard<std::mutex> l(mutex);
        T s;
        auto A_ptr = triang_matrix(A);
        std::cout << "============================" << std::endl;
        std::cout << "Метод Гауса: \n" << A_ptr << std::endl;
        T* x = new T[N];
        for(int i = A_ptr->size1() - 1; i >= 0; --i) {
            s = T(0);
            for(size_t j = i + 1; j < A_ptr->size2() - 1; j++) {
                s += A_ptr->at_element(i, j) * x[j];
            }
            x[i] = (A_ptr->at_element(i, A_ptr->size2() - 1) - s)/A_ptr->at_element(i, i);
        }
        std::cout << "Корни: \n"  << std::endl;
        for(unsigned j = 0; j < N; ++j) std::cout << "x[" << j + 1 << "] = " << x[j] << std::endl;
        delete [] x;
    }
    void solver_LU() {
        std::lock_guard<std::mutex> l(mutex);
        for(size_t i = 0; i < N; ++i) L->at_element(i, i) = T(1);
        for(size_t i = 0; i < N; ++i) {
            U_Row(i);
            if(i < N - 1) L_Col(i);
        }
        std::cout << "============================" << std::endl;
        std::cout << std::setprecision(1) << "LU разложение\nU =\n" << U << std::endl;
        std::cout << std::setprecision(1) << "L =\n" << L << std::endl;
        LV_B();
        UX_V();
        int i = 1;
        std::cout << "Корни: \n"  << std::endl;
        for(const auto& t : *x_1) std::cout << "x[" << i ++ << "] = " << t << std::endl;
        // weak_matrix = A;
        // weak_ptr_info(weak_matrix);
    }

private:
    void U_Row(size_t i) {
        T s;
        for(size_t j = i; j < N; ++j) {
            s = T(0);
            for(size_t k = 0; k < N - 1; ++k) {
                s += U->at_element(k, j)*L->at_element(i, k);
            }
            U->at_element(i, j) = A->at_element(i, j) - s;
        }
    }
    void L_Col(size_t j) {
        T s;
        for(size_t i = j + 1; i < N; ++i) {
            s = T(0);
            for(size_t k = 0; k <= j; k++) {
                s += U->at_element(k, j)*L->at_element(i, k);
            }
            L->at_element(i, j) = (A->at_element(i, j) - s)/U->at_element(j, j);
            // std::cout << s << " ";
        }
    }
    void LV_B() {
        T s;
        for(size_t i = 0; i < N; ++i) {
            s = T(0);
            for(size_t j = 0; j <= i; j++) {
                s += L->at_element(i, j)*v->at(j);
            }
            v->at(i) = A->at_element(i, M - 1) - s;
        }
    }
    void UX_V() {
        T s;
        for(int i = N - 1; i >= 0; i--) {
            s = T(0);
            for(size_t j = i + 1; j < N; j++) {
                s += U->at_element(i, j)*x_1->at(j);
            }
            x_1->at(i) = (v->at(i) - s)/U->at_element(i, i);
        }
    }
};


