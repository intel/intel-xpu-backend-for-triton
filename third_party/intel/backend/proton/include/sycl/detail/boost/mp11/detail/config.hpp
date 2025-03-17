  // -*- C++ -*-
  //===----------------------------------------------------------------------===//
  // Modifications Copyright Intel Corporation 2022
  // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
  //===----------------------------------------------------------------------===//
  // Auto-generated from boost/mp11 sources https://github.com/boostorg/mp11

#ifndef SYCL_DETAIL_BOOST_MP11_DETAIL_CONFIG_HPP_INCLUDED
#define SYCL_DETAIL_BOOST_MP11_DETAIL_CONFIG_HPP_INCLUDED

// Copyright 2016, 2018, 2019 Peter Dimov.
//
// Distributed under the Boost Software License, Version 1.0.
//
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt

// SYCL_DETAIL_BOOST_MP11_WORKAROUND

#if defined( SYCL_DETAIL_BOOST_STRICT_CONFIG ) || defined( SYCL_DETAIL_BOOST_MP11_NO_WORKAROUNDS )

# define SYCL_DETAIL_BOOST_MP11_WORKAROUND( symbol, test ) 0

#else

# define SYCL_DETAIL_BOOST_MP11_WORKAROUND( symbol, test ) ((symbol) != 0 && ((symbol) test))

#endif

//

#define SYCL_DETAIL_BOOST_MP11_CUDA 0
#define SYCL_DETAIL_BOOST_MP11_CLANG 0
#define SYCL_DETAIL_BOOST_MP11_INTEL 0
#define SYCL_DETAIL_BOOST_MP11_GCC 0
#define SYCL_DETAIL_BOOST_MP11_MSVC 0

#define SYCL_DETAIL_BOOST_MP11_CONSTEXPR constexpr

#if defined( __CUDACC__ )

// nvcc

# undef SYCL_DETAIL_BOOST_MP11_CUDA
# define SYCL_DETAIL_BOOST_MP11_CUDA (__CUDACC_VER_MAJOR__ * 1000000 + __CUDACC_VER_MINOR__ * 10000 + __CUDACC_VER_BUILD__)

// CUDA (8.0) has no constexpr support in msvc mode:
# if defined(_MSC_VER) && (SYCL_DETAIL_BOOST_MP11_CUDA < 9000000)

#  define SYCL_DETAIL_BOOST_MP11_NO_CONSTEXPR

#  undef SYCL_DETAIL_BOOST_MP11_CONSTEXPR
#  define SYCL_DETAIL_BOOST_MP11_CONSTEXPR

# endif

#endif

#if defined(__clang__)

// Clang

# undef SYCL_DETAIL_BOOST_MP11_CLANG
# define SYCL_DETAIL_BOOST_MP11_CLANG (__clang_major__ * 100 + __clang_minor__)

# if defined(__has_cpp_attribute)
#  if __has_cpp_attribute(fallthrough) && __cplusplus >= 201406L // Clang 3.9+ in c++1z mode
#   define SYCL_DETAIL_BOOST_MP11_HAS_FOLD_EXPRESSIONS
#  endif
# endif

#if SYCL_DETAIL_BOOST_MP11_CLANG < 400 && __cplusplus >= 201402L \
   && defined( __GLIBCXX__ ) && !__has_include(<shared_mutex>)

// Clang pre-4 in C++14 mode, libstdc++ pre-4.9, ::gets is not defined,
// but Clang tries to import it into std

   extern "C" char *gets (char *__s);
#endif

#elif defined(__INTEL_COMPILER)

// Intel C++

# undef SYCL_DETAIL_BOOST_MP11_INTEL
# define SYCL_DETAIL_BOOST_MP11_INTEL __INTEL_COMPILER

#elif defined(__GNUC__)

// g++

# undef SYCL_DETAIL_BOOST_MP11_GCC
# define SYCL_DETAIL_BOOST_MP11_GCC (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)

#elif defined(_MSC_VER)

// MS Visual C++

# undef SYCL_DETAIL_BOOST_MP11_MSVC
# define SYCL_DETAIL_BOOST_MP11_MSVC _MSC_VER

# if SYCL_DETAIL_BOOST_MP11_WORKAROUND( SYCL_DETAIL_BOOST_MP11_MSVC, < 1920 )
#  define SYCL_DETAIL_BOOST_MP11_NO_CONSTEXPR
# endif

#if _MSC_FULL_VER < 190024210 // 2015u3
#  undef SYCL_DETAIL_BOOST_MP11_CONSTEXPR
#  define SYCL_DETAIL_BOOST_MP11_CONSTEXPR
#endif

#endif

// SYCL_DETAIL_BOOST_MP11_HAS_CXX14_CONSTEXPR

#if !defined(SYCL_DETAIL_BOOST_MP11_NO_CONSTEXPR) && defined(__cpp_constexpr) && __cpp_constexpr >= 201304
#  define SYCL_DETAIL_BOOST_MP11_HAS_CXX14_CONSTEXPR
#endif

// SYCL_DETAIL_BOOST_MP11_HAS_FOLD_EXPRESSIONS

#if !defined(SYCL_DETAIL_BOOST_MP11_HAS_FOLD_EXPRESSIONS) && defined(__cpp_fold_expressions) && __cpp_fold_expressions >= 201603
#  define SYCL_DETAIL_BOOST_MP11_HAS_FOLD_EXPRESSIONS
#endif

// SYCL_DETAIL_BOOST_MP11_HAS_TYPE_PACK_ELEMENT

#if defined(__has_builtin)
# if __has_builtin(__type_pack_element)
#  define SYCL_DETAIL_BOOST_MP11_HAS_TYPE_PACK_ELEMENT
# endif
#endif

// SYCL_DETAIL_BOOST_MP11_HAS_TEMPLATE_AUTO

#if defined(__cpp_nontype_template_parameter_auto) && __cpp_nontype_template_parameter_auto >= 201606L
# define SYCL_DETAIL_BOOST_MP11_HAS_TEMPLATE_AUTO
#endif

#if SYCL_DETAIL_BOOST_MP11_WORKAROUND( SYCL_DETAIL_BOOST_MP11_MSVC, < 1920 )
// mp_value<0> is bool, mp_value<-1L> is int, etc
# undef SYCL_DETAIL_BOOST_MP11_HAS_TEMPLATE_AUTO
#endif

// SYCL_DETAIL_BOOST_MP11_DEPRECATED(msg)

#if SYCL_DETAIL_BOOST_MP11_WORKAROUND( SYCL_DETAIL_BOOST_MP11_CLANG, < 304 )
#  define SYCL_DETAIL_BOOST_MP11_DEPRECATED(msg)
#elif defined(__GNUC__) || defined(__clang__)
#  define SYCL_DETAIL_BOOST_MP11_DEPRECATED(msg) __attribute__((deprecated(msg)))
#elif defined(_MSC_VER) && _MSC_VER >= 1900
#  define SYCL_DETAIL_BOOST_MP11_DEPRECATED(msg) [[deprecated(msg)]]
#else
#  define SYCL_DETAIL_BOOST_MP11_DEPRECATED(msg)
#endif

#endif // #ifndef SYCL_DETAIL_BOOST_MP11_DETAIL_CONFIG_HPP_INCLUDED
