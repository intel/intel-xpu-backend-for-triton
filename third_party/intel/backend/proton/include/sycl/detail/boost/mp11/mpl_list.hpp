  // -*- C++ -*-
  //===----------------------------------------------------------------------===//
  // Modifications Copyright Intel Corporation 2022
  // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
  //===----------------------------------------------------------------------===//
  // Auto-generated from boost/mp11 sources https://github.com/boostorg/mp11

#ifndef SYCL_DETAIL_BOOST_MP11_MPL_LIST_HPP_INCLUDED
#define SYCL_DETAIL_BOOST_MP11_MPL_LIST_HPP_INCLUDED

// Copyright 2017, 2019 Peter Dimov.
//
// Distributed under the Boost Software License, Version 1.0.
//
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt

#include <sycl/detail/boost/mp11/detail/mpl_common.hpp>
namespace sycl
{
inline namespace _V1
{
namespace detail
{
namespace boost
{
namespace mpl
{

template< typename Sequence > struct sequence_tag;

template<class... T> struct sequence_tag<mp11::mp_list<T...>>
{
    using type = aux::mp11_tag;
};

} // namespace mpl
} // namespace boost
} // namespace detail
} // namespace _V1
} // namespace sycl

#endif // #ifndef SYCL_DETAIL_BOOST_MP11_MPL_LIST_HPP_INCLUDED
