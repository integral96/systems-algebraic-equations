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



template<bool PRED> struct is_LU : boost::mpl::false_ {};
template<> struct is_LU<true> : boost::mpl::true_ {};

template<size_t N, typename T>
struct multy_tmp {
    typedef multy_tmp<N - 1, T> sub_type;
    multy_tmp(const T& val) : value(val), sub(val) {}

    T value;
    sub_type sub;
};

template<typename T>
struct multy_tmp<0, T> {
    multy_tmp(const T& val)  {}
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



template<size_t I, size_t N, size_t J, size_t M>
struct mult_block {
    typedef loopback<I, N, J, M> loop_type;
    typedef mult_block<loop_type::next_I, N, loop_type::next_J, M> next_type;
    template<typename T, typename Matrix>
    void operator ()(T& tmp, Matrix& U, Matrix& L, size_t i, size_t k, size_t j) {
        tmp.value += U(k, j + J) * L(i + I, k);
        next_type()(tmp.sub, U, L, i, k, j);
    }
    template<bool PRED, typename T, typename Matrix>
    void update(const T& tmp, Matrix& U, Matrix& L, const Matrix& A, size_t i, size_t j) {
        if constexpr(is_LU<PRED>::value) {
            U(i + I, j + J) = A(i + I, j + J) - tmp.value;
            next_type().template update<PRED>(tmp.sub, U, L, A, i, j);
        } else {
            L(i + I, j + J) = (A(i + I, j + J) - tmp.value)/U(j + J, j + J);
            next_type().template update<PRED>(tmp.sub, U, L, A, i, j);
        }
    }
};

template<size_t I, size_t N, size_t M>
struct mult_block<I, N, M, M> {
    typedef loopback<I, N, I, M> loop_type;
    typedef mult_block<loop_type::next_I + 1, N, loop_type::next_J, M> next_type;

    template<typename T, typename Matrix>
    void operator ()(T& tmp, Matrix& U, Matrix& L, size_t i, size_t k, size_t j) {
        tmp.value += U(k, j + M) * L(i + I, k);
        next_type()(tmp.sub, U, L, i, k, j);
    }
    template<bool PRED, typename T, typename Matrix>
    void update(const T& tmp, Matrix& U, Matrix& L, const Matrix& A, size_t i, size_t j) {
        if constexpr(is_LU<PRED>::value) {
            U(i + I, j + M) = A(i + I, j + M) - tmp.value;
            next_type().template update<PRED>(tmp.sub, U, L, A, i, j);
        } else {
            L(i + I, j + M) = (A(i + I, j + M) - tmp.value)/U(j + M, j + M);
            next_type().template update<PRED>(tmp.sub, U, L, A, i, j);
        }
    }
};
template<size_t N, size_t J, size_t M>
struct mult_block<N, N, J, M> {
    typedef loopback<0, N, J, M> loop_type;
    typedef mult_block<loop_type::next_I, N, loop_type::next_J + 1, M> next_type;

    template<typename T, typename Matrix>
    void operator ()(T& tmp, Matrix& U, Matrix& L, size_t i, size_t k, size_t j) {
        tmp.value += U(k, j + J) * L(i + N, k);
        next_type()(tmp.sub, U, L, i, k, j);
    }
    template<bool PRED, typename T, typename Matrix>
    void update(const T& tmp, Matrix& U, Matrix& L, const Matrix& A, size_t i, size_t j) {
        if constexpr(is_LU<PRED>::value) {
            U(i + N, j + J) = A(i + N, j + J) - tmp.value;
            next_type().template update<PRED>(tmp.sub, U, L, A, i, j);
        } else {
            L(i + N, j + J) = (A(i + N, j + J) - tmp.value)/U(j + J, j + J);
            next_type().template update<PRED>(tmp.sub, U, L, A, i, j);
        }
    }
};
template<size_t N, size_t M>
struct mult_block<N, N, M, M> {

    template<typename T, typename Matrix>
    void operator ()(T& tmp, Matrix& U, Matrix& L, size_t i, size_t k, size_t j) {
        tmp.value += U(k, j + M) * L(i + N, k);
    }
    template<bool PRED, typename T, typename Matrix>
    void update(const T& tmp, Matrix& U, Matrix& L, const Matrix& A, size_t i, size_t j) {
        if constexpr(is_LU<PRED>::value) {
            U(i + N, j + M) = A(i + N, j + M) - tmp.value;
        } else {
            L(i + N, j + M) = (A(i + N, j + M) - tmp.value)/U(j + M, j + M);
        }
    }
};

template<size_t N, size_t M, typename Matrix>
inline void LU_solv(const Matrix& A, Matrix& L, Matrix& U) {
    typedef typename Matrix::value_type value_type;
    size_t S = A.size1();
    mult_block<0, N - 1, 0, M - 1> block;
    for (size_t i = 0; i < S; i += N) {
        for (size_t j = i; j < S; j += M)
        {
            multy_tmp<N*M, value_type> tmp(value_type(0));
            for (size_t k = 0; k < S; ++k)
            {
                block(tmp, U, L, i, k, j);
            }
            block.template update<true>(tmp, U, L, A, i, j);
        }
        if(i < S - 1) {
            for (size_t j = i; j < S; j += M)
            {
                multy_tmp<N*M, value_type> tmp(value_type(0));
                for (size_t k = 0; k < i; ++k)
                {
                    block(tmp, U, L, j, k, i);
                }
                block.template update<false>(tmp, U, L, A, j, i);
            }
        }
    }
}

/*!
 * read Matrix
 */

