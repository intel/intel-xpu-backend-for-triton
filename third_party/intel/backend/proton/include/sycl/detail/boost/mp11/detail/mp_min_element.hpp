  // -*- C++ -*-
  //===----------------------------------------------------------------------===//
  // Modifications Copyright Intel Corporation 2022
  // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
  //===----------------------------------------------------------------------===//
  // Auto-generated from boost/mp11 sources https://github.com/boostorg/mp11

#ifndef SYCL_DETAIL_BOOST_MP11_DETAIL_MP_MIN_ELEMENT_HPP_INCLUDED
#define SYCL_DETAIL_BOOST_MP11_DETAIL_MP_MIN_ELEMENT_HPP_INCLUDED

//  Copyright 2015-2017 Peter Dimov.
//
//  Distributed under the Boost Software License, Version 1.0.
//
//  See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt

#include <sycl/detail/boost/mp11/detail/mp_fold.hpp>
#include <sycl/detail/boost/mp11/list.hpp>
#include <sycl/detail/boost/mp11/utility.hpp>
namespace sycl
{
inline namespace _V1
{
namespace detail
{
namespace boost
{
namespace mp11
{

// mp_min_element<L, P>
namespace detail
{

template<template<class...> class P> struct select_min
{
    template<class T1, class T2> using fn = mp_if<P<T1, T2>, T1, T2>;
};

} // namespace detail

template<class L, template<class...> class P> using mp_min_element = mp_fold_q<mp_rest<L>, mp_first<L>, detail::select_min<P>>;
template<class L, class Q> using mp_min_element_q = mp_min_element<L, Q::template fn>;

// mp_max_element<L, P>
namespace detail
{

template<template<class...> class P> struct select_max
{
    template<class T1, class T2> using fn = mp_if<P<T2, T1>, T1, T2>;
};

} // namespace detail

template<class L, template<class...> class P> using mp_max_element = mp_fold_q<mp_rest<L>, mp_first<L>, detail::select_max<P>>;
template<class L, class Q> using mp_max_element_q = mp_max_element<L, Q::template fn>;

} // namespace mp11
} // namespace boost
} // namespace detail
} // namespace _V1
} // namespace sycl

#endif // #ifndef SYCL_DETAIL_BOOST_MP11_DETAIL_MP_MIN_ELEMENT_HPP_INCLUDED
