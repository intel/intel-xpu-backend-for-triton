  // -*- C++ -*-
  //===----------------------------------------------------------------------===//
  // Modifications Copyright Intel Corporation 2022
  // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
  //===----------------------------------------------------------------------===//
  // Auto-generated from boost/mp11 sources https://github.com/boostorg/mp11

#ifndef SYCL_DETAIL_BOOST_MP11_DETAIL_MP_MAP_FIND_HPP_INCLUDED
#define SYCL_DETAIL_BOOST_MP11_DETAIL_MP_MAP_FIND_HPP_INCLUDED

//  Copyright 2015 Peter Dimov.
//
//  Distributed under the Boost Software License, Version 1.0.
//
//  See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt

#include <sycl/detail/boost/mp11/utility.hpp>
#include <sycl/detail/boost/mp11/detail/config.hpp>

#if SYCL_DETAIL_BOOST_MP11_WORKAROUND( SYCL_DETAIL_BOOST_MP11_MSVC, < 1930 )

// not exactly good practice, but...
namespace std
{
    template<class... _Types> class tuple;
}

#endif
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

// mp_map_find
namespace detail
{

#if !SYCL_DETAIL_BOOST_MP11_WORKAROUND( SYCL_DETAIL_BOOST_MP11_MSVC, < 1930 )

template<class T> using mpmf_wrap = mp_identity<T>;
template<class T> using mpmf_unwrap = typename T::type;

#else

template<class... T> struct mpmf_tuple {};

template<class T> struct mpmf_wrap_impl
{
    using type = mp_identity<T>;
};

template<class... T> struct mpmf_wrap_impl< std::tuple<T...> >
{
    using type = mp_identity< mpmf_tuple<T...> >;
};

template<class T> using mpmf_wrap = typename mpmf_wrap_impl<T>::type;

template<class T> struct mpmf_unwrap_impl
{
    using type = typename T::type;
};

template<class... T> struct mpmf_unwrap_impl< mp_identity< mpmf_tuple<T...> > >
{
    using type = std::tuple<T...>;
};

template<class T> using mpmf_unwrap = typename mpmf_unwrap_impl<T>::type;

#endif // #if !SYCL_DETAIL_BOOST_MP11_WORKAROUND( SYCL_DETAIL_BOOST_MP11_MSVC, < 1930 )

template<class M, class K> struct mp_map_find_impl;

template<template<class...> class M, class... T, class K> struct mp_map_find_impl<M<T...>, K>
{
    using U = mp_inherit<mpmf_wrap<T>...>;

    template<template<class...> class L, class... U> static mp_identity<L<K, U...>> f( mp_identity<L<K, U...>>* );
    static mp_identity<void> f( ... );

    using type = mpmf_unwrap< decltype( f( static_cast<U*>(0) ) ) >;
};

} // namespace detail

template<class M, class K> using mp_map_find = typename detail::mp_map_find_impl<M, K>::type;

} // namespace mp11
} // namespace boost
} // namespace detail
} // namespace _V1
} // namespace sycl

#endif // #ifndef SYCL_DETAIL_BOOST_MP11_DETAIL_MP_MAP_FIND_HPP_INCLUDED
