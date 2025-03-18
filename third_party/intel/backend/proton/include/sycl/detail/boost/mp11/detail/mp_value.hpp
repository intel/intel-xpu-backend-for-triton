  // -*- C++ -*-
  //===----------------------------------------------------------------------===//
  // Modifications Copyright Intel Corporation 2022
  // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
  //===----------------------------------------------------------------------===//
  // Auto-generated from boost/mp11 sources https://github.com/boostorg/mp11

#ifndef SYCL_DETAIL_BOOST_MP11_DETAIL_MP_VALUE_HPP_INCLUDED
#define SYCL_DETAIL_BOOST_MP11_DETAIL_MP_VALUE_HPP_INCLUDED

// Copyright 2023 Peter Dimov
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <sycl/detail/boost/mp11/detail/config.hpp>
#include <type_traits>

#if defined(SYCL_DETAIL_BOOST_MP11_HAS_TEMPLATE_AUTO)
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

template<auto A> using mp_value = std::integral_constant<decltype(A), A>;

} // namespace mp11
} // namespace boost
} // namespace detail
} // namespace _V1
} // namespace sycl

#endif // #if defined(SYCL_DETAIL_BOOST_MP11_HAS_TEMPLATE_AUTO)

#endif // #ifndef SYCL_DETAIL_BOOST_MP11_DETAIL_MP_VALUE_HPP_INCLUDED
