  // -*- C++ -*-
  //===----------------------------------------------------------------------===//
  // Modifications Copyright Intel Corporation 2022
  // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
  //===----------------------------------------------------------------------===//
  // Auto-generated from boost/mp11 sources https://github.com/boostorg/mp11

#ifndef SYCL_DETAIL_BOOST_MP11_DETAIL_MP_IS_VALUE_LIST_HPP_INCLUDED
#define SYCL_DETAIL_BOOST_MP11_DETAIL_MP_IS_VALUE_LIST_HPP_INCLUDED

// Copyright 2023 Peter Dimov
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <sycl/detail/boost/mp11/integral.hpp>
#include <sycl/detail/boost/mp11/detail/config.hpp>
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

// mp_is_value_list<L>
namespace detail
{

template<class L> struct mp_is_value_list_impl
{
    using type = mp_false;
};

#if defined(SYCL_DETAIL_BOOST_MP11_HAS_TEMPLATE_AUTO)

template<template<auto...> class L, auto... A> struct mp_is_value_list_impl<L<A...>>
{
    using type = mp_true;
};

#endif

} // namespace detail

template<class L> using mp_is_value_list = typename detail::mp_is_value_list_impl<L>::type;

} // namespace mp11
} // namespace boost
} // namespace detail
} // namespace _V1
} // namespace sycl

#endif // #ifndef SYCL_DETAIL_BOOST_MP11_DETAIL_MP_IS_VALUE_LIST_HPP_INCLUDED
