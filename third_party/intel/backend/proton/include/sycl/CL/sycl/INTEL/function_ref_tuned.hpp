//==----------- function_ref_tuned.hpp --- SYCL Function pointers ----------==//
//
// Copyright (C) 2022 Intel Corporation
//
// This software and the related documents are Intel copyrighted materials, and
// your use of them is governed by the express license under which they were
// provided to you ("License"). Unless the License provides otherwise, you may not
// use, modify, copy, publish, distribute, disclose or transmit this software or
// the related documents without Intel's prior written permission.
//
// This software and the related documents are provided as is, with no express
// or implied warranties, other than those that are expressly stated in the
// License.
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is not included by by any other file because it uses
// C++17 features, while the SYCL library is being built with C++14
//
//===----------------------------------------------------------------------===//
//
// This is a preview extension implementation, intended to provide early
// access to a feature for review and community feedback.
//
// Because the interfaces defined by this header file are not final and are
// subject to change they are not intended to be used by shipping software
// products. If you are interested in using this feature in your software
// product, please let us know!
//
//===----------------------------------------------------------------------===//

#pragma once

#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace sycl {
inline namespace _V1 {
namespace INTEL {

// Tag types to declare SIMD signature
// TODO: do we need another one nested namespace? 'simd', 'tag' or something
// else?
/// These three are used to describe an access pattern for each function
/// argument
struct uniform;
struct linear;
struct varying;

/// These two are used to describe whether the function should be compiled with
/// possbility to call it under non-uniform (from sub-group point of view)
/// context (like under control-flow) or not.
struct masked;
struct unmasked;

// TODO: we should probably get a better name for this class
/// This is just a container for list of integers, which is useful in C++
/// templates and constant expressions.
template <int...> struct int_list {};

namespace detail {

template <class...> struct variant_list {};

// return index of first matching type
template <class T, class U, class... Us> static constexpr int find() {
  if constexpr (std::is_same_v<T, U>)
    return 0;
  else
    return find<T, Us...>() + 1;
}
// return index of first matching value
template <auto T, auto U, auto... Us> static constexpr int find() {
  if constexpr (T == U)
    return 0;
  else
    return find<T, Us...>() + 1;
}
// return whether any type matches
template <class T, class... U>
constexpr bool contains_type_v = (std::is_same_v<T, U> || ...);
// return whether any value matches
template <auto T, auto... U>
constexpr bool contains_value_v = ((T == U) || ...);

template <auto F, int S, class... T>
constexpr auto make_function_ref_tuned_impl() noexcept {
#if __SYCL_DEVICE_ONLY__
  return std::array{
      __builtin_generate_SIMD_variant(F, S, std::add_pointer_t<T>())...};
#else
  return std::array<decltype(F), sizeof...(T)>{};
#endif
}
template <class T, size_t N, size_t M>
constexpr auto flatten(const std::array<std::array<T, N>, M> &c) {
  std::array<T, N * M> a;
  for (auto it = a.begin(); auto &e : c)
    it = std::copy(e.begin(), e.end(), it);
  return a;
}

template <auto F, class... T, int... S>
constexpr auto make_function_ref_tuned_impl(int_list<S...>) noexcept {
  return flatten(std::array{make_function_ref_tuned_impl<F, S, T...>()...});
}
} // namespace detail

template <typename F, class L, typename... T> class function_ref_tuned;

/// Wrapper object, which enables creation of one or more versions of a
/// function, which are optimized for particular calling contexts. Those
/// variants are stored as function pointers and compiler automatically chooses
/// the most suitable variant for each call site.
///
/// \tparam Ret Return type of the function.
/// \tparam S Template parameter pack, that holds a list of sub-group sizes
/// compiler should be aware of whlie generating various versions of the
/// function.
/// \tparam Args Template parameter pack, that contains types of function
/// arguments.
/// \tparam T Template parameter pack, that contains description of each
/// variant, which needs to be created.
template <typename Ret, int... S, class... Args, typename... T>
class function_ref_tuned<Ret(Args...), int_list<S...>, T...> {
  using F = Ret(Args...);
  template <int I> using int_constant = std::integral_constant<int, I>;

public:
  // Copy+Move constructor/assignment & destructor are all implicit default
  // (constexpr+noexcept) Default constructor
  constexpr function_ref_tuned() = default;
  constexpr function_ref_tuned(std::nullptr_t) noexcept : ptrs{} {}
  // Conversion constructor. Enabled if source contains all required variants
  // and sg-sizes
  template <
      typename... OT, int... OS,
      typename = std::enable_if_t<(detail::contains_type_v<T, OT...> && ...)>,
      typename = std::enable_if_t<(detail::contains_value_v<S, OS...> && ...)>>
  constexpr function_ref_tuned(
      const function_ref_tuned<F, int_list<OS...>, OT...> &o) noexcept {
    constexpr std::array va{detail::find<T, OT...>()...};
    constexpr std::array sa{detail::find<S, OS...>()...};
    for (int i = 0; auto s : sa)
      for (auto v : va)
        ptrs[i++] = o.ptrs[v + s * va.size()];
  }
  // Call operator·
  Ret operator()(Args &&... args) const {
#if __SYCL_DEVICE_ONLY__
    return __builtin_call_SIMD_variant(detail::variant_list<T...>(),
                                       int_list<S...>(), ptrs.data(),
                                       std::forward<decltype(args)>(args)...);
#else
    return Ret{};
#endif
  }
  // Conversion to bool. Class invariant that either none or all ptrs are null.
  constexpr explicit operator bool() const noexcept { return ptrs[0]; }

private:
  constexpr function_ref_tuned(std::array<F *, sizeof...(T) * sizeof...(S)> p)
      : ptrs(p) {}
  template <auto, int, class...>
  friend constexpr auto make_function_ref_tuned() noexcept;
  template <auto, class, class...>
  friend constexpr auto make_function_ref_tuned() noexcept;
  std::array<F *, sizeof...(T) * sizeof...(S)> ptrs;
  template <class F, class L, class... O> friend class function_ref_tuned;
};

/// Helper function to create a \c function_ref_tuned object
///
/// \tparam F Function, address of which is taken
/// \tparam L List of sub-group sizes, which should be taken into account by the
/// compiler, while handling this function
/// \tparam T Pack of template arguments, describing various variants of
/// function that should be created by the compiler
template <auto F, class L, class... T>
constexpr auto make_function_ref_tuned() noexcept {
  return function_ref_tuned<std::remove_pointer_t<decltype(F)>, L, T...>(
      detail::make_function_ref_tuned_impl<F, T...>(L()));
}

/// Helper function to create a \c function_ref_tuned object
///
/// \tparam S sub-group size, which is required to be used for this function.
/// The rest of template parameters is the same as for the other overload
/// of this function template.
template <auto F, int S, class... T>
constexpr auto make_function_ref_tuned() noexcept {
  return function_ref_tuned<std::remove_pointer_t<decltype(F)>, int_list<S>,
                            T...>(
      detail::make_function_ref_tuned_impl<F, S, T...>());
}

} // namespace INTEL
} // namespace _V1
} // namespace sycl
