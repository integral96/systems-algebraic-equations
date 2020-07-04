#pragma once

#include <iomanip>
#include <fstream>
#include <iostream>

#include <boost/mpl/if.hpp>
#include <boost/type_traits.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/spirit/include/karma.hpp>

namespace matrix = boost::numeric::ublas;
namespace krm  = boost::spirit::karma;


//template<typename T>
//std::ostream& operator << (std::ostream& out, const std::shared_ptr<matrix::matrix<T>>& A) {
//    out << std::setprecision(1)  << krm::format_delimited(krm::columns(A->size2()) [krm::auto_], '\t', A->data()) << std::endl;
//    return out;
//}


//void U_Row(size_t i) {
//    T s;
//    for(size_t j = i; j < N; ++j) {
//        s = T(0);
//        for(size_t k = 0; k < N - 1; ++k) {
//            s += U->at_element(k, j)*L->at_element(i, k);
//        }
//        U->at_element(i, j) = A->at_element(i, j) - s;
//    }
//}
//void L_Col(size_t j) {
//    T s;
//    for(size_t i = j + 1; i < N; ++i) {
//        s = T(0);
//        for(size_t k = 0; k <= j; k++) {
//            s += U->at_element(k, j)*L->at_element(i, k);
//        }
//        L->at_element(i, j) = (A->at_element(i, j) - s)/U->at_element(j, j);
//        // std::cout << s << " ";
//    }
//}

template<size_t N, typename T>
struct multy_tmp {
    typedef multy_tmp<N - 1, T> sub_type;
    multy_tmp(const T& val) : value(val), sub(val) {}

    T value;
    sub_type sub;
};

template<typename T>
struct multy_tmp<0, T> {
    multy_tmp(const T& val) {}
};

template<size_t I, size_t N, size_t J, size_t M>
struct loopback
{
    static constexpr size_t next_I = I;
    static constexpr size_t next_J = J + 1;
};
template<size_t I, size_t N, size_t M>
struct loopback<I, N, M, M>
{
    static constexpr size_t next_I = I + 1;
    static constexpr size_t next_J = 0;
};

template<bool PRED> struct is_LU : boost::mpl::false_ {};
template<> struct is_LU<true> : boost::mpl::true_ {};

template<size_t I, size_t N, size_t J, size_t M, bool PRED>
struct mult_block {
    typedef loopback<I, N, J, M> loop_type;
    typedef mult_block<loop_type::next_I, N, loop_type::next_J, M, PRED> next_type;
    template<typename T, typename Matrix>
    void operator ()(T& tmp, Matrix& U, Matrix& L, size_t i, size_t j, size_t k) {
        tmp.value += U(k, j + J) * L(i + I, k);
        next_type()(tmp.sub, U, L, i, j, k);
    }
    template<typename T, typename Matrix>
    void update(const T& tmp, Matrix& U, Matrix& L, const Matrix& A, size_t i, size_t j) {
        if constexpr(is_LU<PRED>::value) {
            U(i + I, j + J) = A(i + I, j + J) - tmp.value;
            next_type().update(tmp.sub, U, L, A, i, j);
        } else {
            L(i + I, j + J) = (A(i + I, j + J) - tmp.value)/U(j + J, j + J);
            next_type().update(tmp.sub, U, L, A, i, j);
        }
    }
};
template<size_t I, size_t N, size_t M, bool PRED>
struct mult_block<I, N, M, M, PRED> {
    typedef mult_block<I + 1, N, 0, M, PRED> next_type;

    template<typename T, typename Matrix>
    void operator ()(T& tmp, Matrix& U, Matrix& L, size_t i, size_t j, size_t k) {
        tmp.value += U(k, j + M) * L(i + I, k);
        next_type()(tmp.sub, U, L, i, j, k);
    }
    template<typename T, typename Matrix>
    void update(const T& tmp, Matrix& U, Matrix& L, const Matrix& A, size_t i, size_t j) {
        if constexpr(is_LU<PRED>::value) {
            U(i + I, j + M) = A(i + I, j + M) - tmp.value;
            next_type().update(tmp.sub, U, L, A, i, j);
        } else {
            L(i + I, j + M) = (A(i + I, j + M) - tmp.value)/U(j + M, j + M);
            next_type().update(tmp.sub, U, L, A, i, j);
        }
    }
};
template<size_t N, size_t M, bool PRED>
struct mult_block<N, N, M, M, PRED> {

    template<typename T, typename Matrix>
    void operator ()(T& tmp, Matrix& U, Matrix& L, size_t i, size_t j, size_t k) {
        tmp.value += U(k, j + M) * L(i + N, k);
    }
    template<typename T, typename Matrix>
    void update(const T& tmp, Matrix& U, Matrix& L, const Matrix& A, size_t i, size_t j) {
        if constexpr(is_LU<PRED>::value) {
            U(i + N, j + M) = A(i + N, j + M) - tmp.value;
        } else {
            L(i + N, j + M) = (A(i + N, j + M) - tmp.value)/U(j + M, j + M);
        }
    }
};

template<size_t I, size_t N, size_t M, typename Matrix>
void U_Row(const Matrix& A, Matrix& L, Matrix& U) {
    typedef typename Matrix::value_type value_type;
    mult_block<I, N - 1, 0, M - 1, true> block;
    for (size_t j = I; j < N; j += M)
    {
        multy_tmp<N*M, value_type> tmp(value_type{});
        for (size_t k = 0; k < N - 1; ++k)
        {
            block(tmp, U, L, I, k, j);
        }
        block.update(tmp, U, L, A, I, j);
    }
}
template<size_t J, size_t N, size_t M, typename Matrix>
void L_Col(const Matrix& A, Matrix& L, Matrix& U) {
    typedef typename Matrix::value_type value_type;
    mult_block<0, N - 1, J, M - 1, false> block;
    for (size_t i = J + 1; i < N; i += M)
    {
        multy_tmp<N*M, value_type> tmp(value_type{});
        for (size_t k = 0; k <= J; ++k)
        {
            block(tmp, U, L, i, k, J);
        }
        block.update(tmp, U, L, A, i, J);
    }
}

/*!
 * read Matrix
 */
void solve_Meta_LU(const std::string& name) {
    size_t N{}, M{};
    std::ifstream file(name);

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
    matrix::matrix<double> A(N, M), L(N, N), U(N, N);

    for(size_t i = 0; i < A.size1(); ++i)
        for(size_t j = 0; j < A.size2(); ++j) file >> A(i, j);

    for(size_t i = 0; i < N; ++i) L(i, i) = 1.;
    std::cout << krm::format_delimited(krm::columns(A.size2()) [krm::auto_], '\t', A.data()) << std::endl;
    U_Row<2, 10, 11, matrix::matrix<double>>(A, L, U);
    L_Col<2, 10, 11, matrix::matrix<double>>(A, L, U);
    std::cout << krm::format_delimited(krm::columns(L.size2()) [krm::auto_], '\t', L.data())  << std::endl
              << krm::format_delimited(krm::columns(U.size2()) [krm::auto_], '\t', U.data()) << std::endl;

}
