// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#define ENABLE_SYCL_EXT_ONEAPI_INVOKE_SIMD
// clang-format off
#ifdef ENABLE_SYCL_EXT_ONEAPI_INVOKE_SIMD
// Stub implemenation the "Data-Parallel Types" section of the
// " C++ Extensions for Parallelism Version 2":
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2019/n4808.pdf.
//
// This is mostly a copy-paste from
// https://github.com/intel/llvm/blob/sycl/libcxx/include/experimental/simd
// Changes are marked with "SYCL/invoke_simd".
// Primary usage for now is to implement the invoke_simd spec.
//===----------------------------------------------------------------------===//
#endif // ENABLE_SYCL_EXT_ONEAPI_INVOKE_SIMD

#pragma once // Added for ENABLE_SYCL_EXT_ONEAPI_INVOKE_SIMD
// Removed for ENABLE_SYCL_EXT_ONEAPI_INVOKE_SIMD {
//#ifndef _LIBCPP_EXPERIMENTAL_SIMD
//#define _LIBCPP_EXPERIMENTAL_SIMD
// } Removed for ENABLE_SYCL_EXT_ONEAPI_INVOKE_SIMD

/*
    experimental/simd synopsis

namespace std::experimental {

inline namespace parallelism_v2 {

namespace simd_abi {

struct scalar {};
template <int N> struct fixed_size {};
template <typename T> inline constexpr int max_fixed_size = implementation-defined;
template <typename T> using compatible = implementation-defined;
template <typename T> using native = implementation-defined;

} // simd_abi

struct element_aligned_tag {};
struct vector_aligned_tag {};
template <size_t> struct overaligned_tag {};
inline constexpr element_aligned_tag element_aligned{};
inline constexpr vector_aligned_tag vector_aligned{};
template <size_t N> inline constexpr overaligned_tag<N> overaligned{};

// traits [simd.traits]
template <class T> struct is_abi_tag;
template <class T> inline constexpr bool is_abi_tag_v = is_abi_tag<T>::value;

template <class T> struct is_simd;
template <class T> inline constexpr bool is_simd_v = is_simd<T>::value;

template <class T> struct is_simd_mask;
template <class T> inline constexpr bool is_simd_mask_v = is_simd_mask<T>::value;

template <class T> struct is_simd_flag_type;
template <class T> inline constexpr bool is_simd_flag_type_v = is_simd_flag_type<T>::value;

template <class T, size_t N> struct abi_for_size { using type = see below; };
template <class T, size_t N> using abi_for_size_t = typename abi_for_size<T, N>::type;

template <class T, class Abi = simd_abi::compatible<T>> struct simd_size;
template <class T, class Abi = simd_abi::compatible<T>>
inline constexpr size_t simd_size_v = simd_size<T, Abi>::value;

template <class T, class U = typename T::value_type> struct memory_alignment;
template <class T, class U = typename T::value_type>
inline constexpr size_t memory_alignment_v = memory_alignment<T, U>::value;

// class template simd [simd.class]
template <class T, class Abi = simd_abi::compatible<T>> class simd;
template <class T> using native_simd = simd<T, simd_abi::native<T>>;
template <class T, int N> using fixed_size_simd = simd<T, simd_abi::fixed_size<N>>;

// class template simd_mask [simd.mask.class]
template <class T, class Abi = simd_abi::compatible<T>> class simd_mask;
template <class T> using native_simd_mask = simd_mask<T, simd_abi::native<T>>;
template <class T, int N> using fixed_size_simd_mask = simd_mask<T, simd_abi::fixed_size<N>>;

// casts [simd.casts]
template <class T, class U, class Abi> see below simd_cast(const simd<U, Abi>&);
template <class T, class U, class Abi> see below static_simd_cast(const simd<U, Abi>&);

template <class T, class Abi>
fixed_size_simd<T, simd_size_v<T, Abi>> to_fixed_size(const simd<T, Abi>&) noexcept;
template <class T, class Abi>
fixed_size_simd_mask<T, simd_size_v<T, Abi>> to_fixed_size(const simd_mask<T, Abi>&) noexcept;
template <class T, size_t N> native_simd<T> to_native(const fixed_size_simd<T, N>&) noexcept;
template <class T, size_t N>
native_simd_mask<T> to_native(const fixed_size_simd_mask<T, N>> &) noexcept;
template <class T, size_t N> simd<T> to_compatible(const fixed_size_simd<T, N>&) noexcept;
template <class T, size_t N> simd_mask<T> to_compatible(const fixed_size_simd_mask<T, N>&) noexcept;

template <size_t... Sizes, class T, class Abi>
tuple<simd<T, abi_for_size_t<Sizes>>...> split(const simd<T, Abi>&);
template <size_t... Sizes, class T, class Abi>
tuple<simd_mask<T, abi_for_size_t<Sizes>>...> split(const simd_mask<T, Abi>&);
template <class V, class Abi>
array<V, simd_size_v<typename V::value_type, Abi> / V::size()> split(
const simd<typename V::value_type, Abi>&);
template <class V, class Abi>
array<V, simd_size_v<typename V::value_type, Abi> / V::size()> split(
const simd_mask<typename V::value_type, Abi>&);

template <class T, class... Abis>
simd<T, abi_for_size_t<T, (simd_size_v<T, Abis> + ...)>> concat(const simd<T, Abis>&...);
template <class T, class... Abis>
simd_mask<T, abi_for_size_t<T, (simd_size_v<T, Abis> + ...)>> concat(const simd_mask<T, Abis>&...);

// reductions [simd.mask.reductions]
template <class T, class Abi> bool all_of(const simd_mask<T, Abi>&) noexcept;
template <class T, class Abi> bool any_of(const simd_mask<T, Abi>&) noexcept;
template <class T, class Abi> bool none_of(const simd_mask<T, Abi>&) noexcept;
template <class T, class Abi> bool some_of(const simd_mask<T, Abi>&) noexcept;
template <class T, class Abi> int popcount(const simd_mask<T, Abi>&) noexcept;
template <class T, class Abi> int find_first_set(const simd_mask<T, Abi>&);
template <class T, class Abi> int find_last_set(const simd_mask<T, Abi>&);

bool all_of(see below) noexcept;
bool any_of(see below) noexcept;
bool none_of(see below) noexcept;
bool some_of(see below) noexcept;
int popcount(see below) noexcept;
int find_first_set(see below) noexcept;
int find_last_set(see below) noexcept;

// masked assignment [simd.whereexpr]
template <class M, class T> class const_where_expression;
template <class M, class T> class where_expression;

// masked assignment [simd.mask.where]
template <class T> struct nodeduce { using type = T; }; // exposition only

template <class T> using nodeduce_t = typename nodeduce<T>::type; // exposition only

template <class T, class Abi>
where_expression<simd_mask<T, Abi>, simd<T, Abi>>
where(const typename simd<T, Abi>::mask_type&, simd<T, Abi>&) noexcept;

template <class T, class Abi>
const_where_expression<simd_mask<T, Abi>, const simd<T, Abi>>
where(const typename simd<T, Abi>::mask_type&, const simd<T, Abi>&) noexcept;

template <class T, class Abi>
where_expression<simd_mask<T, Abi>, simd_mask<T, Abi>>
where(const nodeduce_t<simd_mask<T, Abi>>&, simd_mask<T, Abi>&) noexcept;

template <class T, class Abi>
const_where_expression<simd_mask<T, Abi>, const simd_mask<T, Abi>>
where(const nodeduce_t<simd_mask<T, Abi>>&, const simd_mask<T, Abi>&) noexcept;

template <class T> where_expression<bool, T> where(see below k, T& d) noexcept;

template <class T>
const_where_expression<bool, const T> where(see below k, const T& d) noexcept;

// reductions [simd.reductions]
template <class T, class Abi, class BinaryOperation = std::plus<>>
T reduce(const simd<T, Abi>&, BinaryOperation = BinaryOperation());

template <class M, class V, class BinaryOperation>
typename V::value_type reduce(const const_where_expression<M, V>& x,
typename V::value_type neutral_element, BinaryOperation binary_op);

template <class M, class V>
typename V::value_type reduce(const const_where_expression<M, V>& x, plus<> binary_op = plus<>());

template <class M, class V>
typename V::value_type reduce(const const_where_expression<M, V>& x, multiplies<> binary_op);

template <class M, class V>
typename V::value_type reduce(const const_where_expression<M, V>& x, bit_and<> binary_op);

template <class M, class V>
typename V::value_type reduce(const const_where_expression<M, V>& x, bit_or<> binary_op);

template <class M, class V>
typename V::value_type reduce(const const_where_expression<M, V>& x, bit_xor<> binary_op);

template <class T, class Abi> T hmin(const simd<T, Abi>&);
template <class M, class V> T hmin(const const_where_expression<M, V>&);
template <class T, class Abi> T hmax(const simd<T, Abi>&);
template <class M, class V> T hmax(const const_where_expression<M, V>&);

// algorithms [simd.alg]
template <class T, class Abi> simd<T, Abi> min(const simd<T, Abi>&, const simd<T, Abi>&) noexcept;

template <class T, class Abi> simd<T, Abi> max(const simd<T, Abi>&, const simd<T, Abi>&) noexcept;

template <class T, class Abi>
std::pair<simd<T, Abi>, simd<T, Abi>> minmax(const simd<T, Abi>&, const simd<T, Abi>&) noexcept;

template <class T, class Abi>
simd<T, Abi> clamp(const simd<T, Abi>& v, const simd<T, Abi>& lo, const simd<T, Abi>& hi);

// [simd.whereexpr]
template <class M, class T>
class const_where_expression {
  const M& mask; // exposition only
  T& data; // exposition only
public:
  const_where_expression(const const_where_expression&) = delete;
  const_where_expression& operator=(const const_where_expression&) = delete;
  remove_const_t<T> operator-() const &&;
  template <class U, class Flags> void copy_to(U* mem, Flags f) const &&;
};

template <class M, class T>
class where_expression : public const_where_expression<M, T> {
public:
  where_expression(const where_expression&) = delete;
  where_expression& operator=(const where_expression&) = delete;
  template <class U> void operator=(U&& x);
  template <class U> void operator+=(U&& x);
  template <class U> void operator-=(U&& x);
  template <class U> void operator*=(U&& x);
  template <class U> void operator/=(U&& x);
  template <class U> void operator%=(U&& x);
  template <class U> void operator&=(U&& x);
  template <class U> void operator|=(U&& x);
  template <class U> void operator^=(U&& x);
  template <class U> void operator<<=(U&& x);
  template <class U> void operator>>=(U&& x);
  void operator++();
  void operator++(int);
  void operator--();
  void operator--(int);
  template <class U, class Flags> void copy_from(const U* mem, Flags);
};

// [simd.class]
template <class T, class Abi> class simd {
public:
  using value_type = T;
  using reference = see below;
  using mask_type = simd_mask<T, Abi>;

  using abi_type = Abi;
  static constexpr size_t size() noexcept;
  simd() = default;

  // implicit type conversion constructor
  template <class U> simd(const simd<U, simd_abi::fixed_size<size()>>&);

  // implicit broadcast constructor (see below for constraints)
  template <class U> simd(U&& value);

  // generator constructor (see below for constraints)
  template <class G> explicit simd(G&& gen);

  // load constructor
  template <class U, class Flags> simd(const U* mem, Flags f);

  // loads [simd.load]
  template <class U, class Flags> void copy_from(const U* mem, Flags f);

  // stores [simd.store]
  template <class U, class Flags> void copy_to(U* mem, Flags f) const;

  // scalar access [simd.subscr]
  reference operator[](size_t);
  value_type operator[](size_t) const;

  // unary operators [simd.unary]
  simd& operator++();
  simd operator++(int);
  simd& operator--();
  simd operator--(int);
  mask_type operator!() const;
  simd operator~() const; // see below
  simd operator+() const;
  simd operator-() const;

  // binary operators [simd.binary]
  friend simd operator+ (const simd&, const simd&);
  friend simd operator- (const simd&, const simd&);
  friend simd operator* (const simd&, const simd&);
  friend simd operator/ (const simd&, const simd&);
  friend simd operator% (const simd&, const simd&);
  friend simd operator& (const simd&, const simd&);
  friend simd operator| (const simd&, const simd&);
  friend simd operator^ (const simd&, const simd&);
  friend simd operator<<(const simd&, const simd&);
  friend simd operator>>(const simd&, const simd&);
  friend simd operator<<(const simd&, int);
  friend simd operator>>(const simd&, int);

  // compound assignment [simd.cassign]
  friend simd& operator+= (simd&, const simd&);
  friend simd& operator-= (simd&, const simd&);
  friend simd& operator*= (simd&, const simd&);
  friend simd& operator/= (simd&, const simd&);
  friend simd& operator%= (simd&, const simd&);

  friend simd& operator&= (simd&, const simd&);
  friend simd& operator|= (simd&, const simd&);
  friend simd& operator^= (simd&, const simd&);
  friend simd& operator<<=(simd&, const simd&);
  friend simd& operator>>=(simd&, const simd&);
  friend simd& operator<<=(simd&, int);
  friend simd& operator>>=(simd&, int);

  // compares [simd.comparison]
  friend mask_type operator==(const simd&, const simd&);
  friend mask_type operator!=(const simd&, const simd&);
  friend mask_type operator>=(const simd&, const simd&);
  friend mask_type operator<=(const simd&, const simd&);
  friend mask_type operator> (const simd&, const simd&);
  friend mask_type operator< (const simd&, const simd&);
};

// [simd.math]
template <class Abi> using scharv = simd<signed char, Abi>; // exposition only
template <class Abi> using shortv = simd<short, Abi>; // exposition only
template <class Abi> using intv = simd<int, Abi>; // exposition only
template <class Abi> using longv = simd<long int, Abi>; // exposition only
template <class Abi> using llongv = simd<long long int, Abi>; // exposition only
template <class Abi> using floatv = simd<float, Abi>; // exposition only
template <class Abi> using doublev = simd<double, Abi>; // exposition only
template <class Abi> using ldoublev = simd<long double, Abi>; // exposition only
template <class T, class V> using samesize = fixed_size_simd<T, V::size()>; // exposition only

template <class Abi> floatv<Abi> acos(floatv<Abi> x);
template <class Abi> doublev<Abi> acos(doublev<Abi> x);
template <class Abi> ldoublev<Abi> acos(ldoublev<Abi> x);

template <class Abi> floatv<Abi> asin(floatv<Abi> x);
template <class Abi> doublev<Abi> asin(doublev<Abi> x);
template <class Abi> ldoublev<Abi> asin(ldoublev<Abi> x);

template <class Abi> floatv<Abi> atan(floatv<Abi> x);
template <class Abi> doublev<Abi> atan(doublev<Abi> x);
template <class Abi> ldoublev<Abi> atan(ldoublev<Abi> x);

template <class Abi> floatv<Abi> atan2(floatv<Abi> y, floatv<Abi> x);
template <class Abi> doublev<Abi> atan2(doublev<Abi> y, doublev<Abi> x);
template <class Abi> ldoublev<Abi> atan2(ldoublev<Abi> y, ldoublev<Abi> x);

template <class Abi> floatv<Abi> cos(floatv<Abi> x);
template <class Abi> doublev<Abi> cos(doublev<Abi> x);
template <class Abi> ldoublev<Abi> cos(ldoublev<Abi> x);

template <class Abi> floatv<Abi> sin(floatv<Abi> x);
template <class Abi> doublev<Abi> sin(doublev<Abi> x);
template <class Abi> ldoublev<Abi> sin(ldoublev<Abi> x);

template <class Abi> floatv<Abi> tan(floatv<Abi> x);
template <class Abi> doublev<Abi> tan(doublev<Abi> x);
template <class Abi> ldoublev<Abi> tan(ldoublev<Abi> x);

template <class Abi> floatv<Abi> acosh(floatv<Abi> x);
template <class Abi> doublev<Abi> acosh(doublev<Abi> x);
template <class Abi> ldoublev<Abi> acosh(ldoublev<Abi> x);

template <class Abi> floatv<Abi> asinh(floatv<Abi> x);
template <class Abi> doublev<Abi> asinh(doublev<Abi> x);
template <class Abi> ldoublev<Abi> asinh(ldoublev<Abi> x);

template <class Abi> floatv<Abi> atanh(floatv<Abi> x);
template <class Abi> doublev<Abi> atanh(doublev<Abi> x);
template <class Abi> ldoublev<Abi> atanh(ldoublev<Abi> x);

template <class Abi> floatv<Abi> cosh(floatv<Abi> x);
template <class Abi> doublev<Abi> cosh(doublev<Abi> x);
template <class Abi> ldoublev<Abi> cosh(ldoublev<Abi> x);

template <class Abi> floatv<Abi> sinh(floatv<Abi> x);
template <class Abi> doublev<Abi> sinh(doublev<Abi> x);
template <class Abi> ldoublev<Abi> sinh(ldoublev<Abi> x);

template <class Abi> floatv<Abi> tanh(floatv<Abi> x);
template <class Abi> doublev<Abi> tanh(doublev<Abi> x);
template <class Abi> ldoublev<Abi> tanh(ldoublev<Abi> x);

template <class Abi> floatv<Abi> exp(floatv<Abi> x);
template <class Abi> doublev<Abi> exp(doublev<Abi> x);
template <class Abi> ldoublev<Abi> exp(ldoublev<Abi> x);

template <class Abi> floatv<Abi> exp2(floatv<Abi> x);
template <class Abi> doublev<Abi> exp2(doublev<Abi> x);
template <class Abi> ldoublev<Abi> exp2(ldoublev<Abi> x);

template <class Abi> floatv<Abi> expm1(floatv<Abi> x);
template <class Abi> doublev<Abi> expm1(doublev<Abi> x);
template <class Abi> ldoublev<Abi> expm1(ldoublev<Abi> x);

template <class Abi> floatv<Abi> frexp(floatv<Abi> value, samesize<int, floatv<Abi>>* exp);
template <class Abi> doublev<Abi> frexp(doublev<Abi> value, samesize<int, doublev<Abi>>* exp);
template <class Abi> ldoublev<Abi> frexp(ldoublev<Abi> value, samesize<int, ldoublev<Abi>>* exp);

template <class Abi> samesize<int, floatv<Abi>> ilogb(floatv<Abi> x);
template <class Abi> samesize<int, doublev<Abi>> ilogb(doublev<Abi> x);
template <class Abi> samesize<int, ldoublev<Abi>> ilogb(ldoublev<Abi> x);

template <class Abi> floatv<Abi> ldexp(floatv<Abi> x, samesize<int, floatv<Abi>> exp);
template <class Abi> doublev<Abi> ldexp(doublev<Abi> x, samesize<int, doublev<Abi>> exp);
template <class Abi> ldoublev<Abi> ldexp(ldoublev<Abi> x, samesize<int, ldoublev<Abi>> exp);

template <class Abi> floatv<Abi> log(floatv<Abi> x);
template <class Abi> doublev<Abi> log(doublev<Abi> x);
template <class Abi> ldoublev<Abi> log(ldoublev<Abi> x);

template <class Abi> floatv<Abi> log10(floatv<Abi> x);
template <class Abi> doublev<Abi> log10(doublev<Abi> x);
template <class Abi> ldoublev<Abi> log10(ldoublev<Abi> x);

template <class Abi> floatv<Abi> log1p(floatv<Abi> x);
template <class Abi> doublev<Abi> log1p(doublev<Abi> x);
template <class Abi> ldoublev<Abi> log1p(ldoublev<Abi> x);

template <class Abi> floatv<Abi> log2(floatv<Abi> x);
template <class Abi> doublev<Abi> log2(doublev<Abi> x);
template <class Abi> ldoublev<Abi> log2(ldoublev<Abi> x);

template <class Abi> floatv<Abi> logb(floatv<Abi> x);
template <class Abi> doublev<Abi> logb(doublev<Abi> x);
template <class Abi> ldoublev<Abi> logb(ldoublev<Abi> x);

template <class Abi> floatv<Abi> modf(floatv<Abi> value, floatv<Abi>* iptr);
template <class Abi> doublev<Abi> modf(doublev<Abi> value, doublev<Abi>* iptr);
template <class Abi> ldoublev<Abi> modf(ldoublev<Abi> value, ldoublev<Abi>* iptr);

template <class Abi> floatv<Abi> scalbn(floatv<Abi> x, samesize<int, floatv<Abi>> n);
template <class Abi> doublev<Abi> scalbn(doublev<Abi> x, samesize<int, doublev<Abi>> n);
template <class Abi> ldoublev<Abi> scalbn(ldoublev<Abi> x, samesize<int, ldoublev<Abi>> n);
template <class Abi> floatv<Abi> scalbln(floatv<Abi> x, samesize<long int, floatv<Abi>> n);
template <class Abi> doublev<Abi> scalbln(doublev<Abi> x, samesize<long int, doublev<Abi>> n);
template <class Abi> ldoublev<Abi> scalbln(ldoublev<Abi> x, samesize<long int, ldoublev<Abi>> n);

template <class Abi> floatv<Abi> cbrt(floatv<Abi> x);
template <class Abi> doublev<Abi> cbrt(doublev<Abi> x);
template <class Abi> ldoublev<Abi> cbrt(ldoublev<Abi> x);

template <class Abi> scharv<Abi> abs(scharv<Abi> j);
template <class Abi> shortv<Abi> abs(shortv<Abi> j);
template <class Abi> intv<Abi> abs(intv<Abi> j);
template <class Abi> longv<Abi> abs(longv<Abi> j);
template <class Abi> llongv<Abi> abs(llongv<Abi> j);
template <class Abi> floatv<Abi> abs(floatv<Abi> j);
template <class Abi> doublev<Abi> abs(doublev<Abi> j);
template <class Abi> ldoublev<Abi> abs(ldoublev<Abi> j);

template <class Abi> floatv<Abi> hypot(floatv<Abi> x, floatv<Abi> y);
template <class Abi> doublev<Abi> hypot(doublev<Abi> x, doublev<Abi> y);
template <class Abi> ldoublev<Abi> hypot(doublev<Abi> x, doublev<Abi> y);
template <class Abi> floatv<Abi> hypot(floatv<Abi> x, floatv<Abi> y, floatv<Abi> z);
template <class Abi> doublev<Abi> hypot(doublev<Abi> x, doublev<Abi> y, doublev<Abi> z);
template <class Abi> ldoublev<Abi> hypot(ldoublev<Abi> x, ldoublev<Abi> y, ldoublev<Abi> z);

template <class Abi> floatv<Abi> pow(floatv<Abi> x, floatv<Abi> y);
template <class Abi> doublev<Abi> pow(doublev<Abi> x, doublev<Abi> y);
template <class Abi> ldoublev<Abi> pow(ldoublev<Abi> x, ldoublev<Abi> y);

template <class Abi> floatv<Abi> sqrt(floatv<Abi> x);
template <class Abi> doublev<Abi> sqrt(doublev<Abi> x);
template <class Abi> ldoublev<Abi> sqrt(ldoublev<Abi> x);

template <class Abi> floatv<Abi> erf(floatv<Abi> x);
template <class Abi> doublev<Abi> erf(doublev<Abi> x);
template <class Abi> ldoublev<Abi> erf(ldoublev<Abi> x);
template <class Abi> floatv<Abi> erfc(floatv<Abi> x);
template <class Abi> doublev<Abi> erfc(doublev<Abi> x);
template <class Abi> ldoublev<Abi> erfc(ldoublev<Abi> x);

template <class Abi> floatv<Abi> lgamma(floatv<Abi> x);
template <class Abi> doublev<Abi> lgamma(doublev<Abi> x);
template <class Abi> ldoublev<Abi> lgamma(ldoublev<Abi> x);

template <class Abi> floatv<Abi> tgamma(floatv<Abi> x);
template <class Abi> doublev<Abi> tgamma(doublev<Abi> x);
template <class Abi> ldoublev<Abi> tgamma(ldoublev<Abi> x);

template <class Abi> floatv<Abi> ceil(floatv<Abi> x);
template <class Abi> doublev<Abi> ceil(doublev<Abi> x);
template <class Abi> ldoublev<Abi> ceil(ldoublev<Abi> x);

template <class Abi> floatv<Abi> floor(floatv<Abi> x);
template <class Abi> doublev<Abi> floor(doublev<Abi> x);
template <class Abi> ldoublev<Abi> floor(ldoublev<Abi> x);

template <class Abi> floatv<Abi> nearbyint(floatv<Abi> x);
template <class Abi> doublev<Abi> nearbyint(doublev<Abi> x);
template <class Abi> ldoublev<Abi> nearbyint(ldoublev<Abi> x);

template <class Abi> floatv<Abi> rint(floatv<Abi> x);
template <class Abi> doublev<Abi> rint(doublev<Abi> x);
template <class Abi> ldoublev<Abi> rint(ldoublev<Abi> x);

template <class Abi> samesize<long int, floatv<Abi>> lrint(floatv<Abi> x);
template <class Abi> samesize<long int, doublev<Abi>> lrint(doublev<Abi> x);
template <class Abi> samesize<long int, ldoublev<Abi>> lrint(ldoublev<Abi> x);
template <class Abi> samesize<long long int, floatv<Abi>> llrint(floatv<Abi> x);
template <class Abi> samesize<long long int, doublev<Abi>> llrint(doublev<Abi> x);
template <class Abi> samesize<long long int, ldoublev<Abi>> llrint(ldoublev<Abi> x);

template <class Abi> floatv<Abi> round(floatv<Abi> x);
template <class Abi> doublev<Abi> round(doublev<Abi> x);
template <class Abi> ldoublev<Abi> round(ldoublev<Abi> x);
template <class Abi> samesize<long int, floatv<Abi>> lround(floatv<Abi> x);
template <class Abi> samesize<long int, doublev<Abi>> lround(doublev<Abi> x);
template <class Abi> samesize<long int, ldoublev<Abi>> lround(ldoublev<Abi> x);
template <class Abi> samesize<long long int, floatv<Abi>> llround(floatv<Abi> x);
template <class Abi> samesize<long long int, doublev<Abi>> llround(doublev<Abi> x);
template <class Abi> samesize<long long int, ldoublev<Abi>> llround(ldoublev<Abi> x);

template <class Abi> floatv<Abi> trunc(floatv<Abi> x);
template <class Abi> doublev<Abi> trunc(doublev<Abi> x);
template <class Abi> ldoublev<Abi> trunc(ldoublev<Abi> x);

template <class Abi> floatv<Abi> fmod(floatv<Abi> x, floatv<Abi> y);
template <class Abi> doublev<Abi> fmod(doublev<Abi> x, doublev<Abi> y);
template <class Abi> ldoublev<Abi> fmod(ldoublev<Abi> x, ldoublev<Abi> y);

template <class Abi> floatv<Abi> remainder(floatv<Abi> x, floatv<Abi> y);
template <class Abi> doublev<Abi> remainder(doublev<Abi> x, doublev<Abi> y);
template <class Abi> ldoublev<Abi> remainder(ldoublev<Abi> x, ldoublev<Abi> y);

template <class Abi> floatv<Abi> remquo(floatv<Abi> x, floatv<Abi> y, samesize<int, floatv<Abi>>* quo);
template <class Abi> doublev<Abi> remquo(doublev<Abi> x, doublev<Abi> y, samesize<int, doublev<Abi>>* quo);
template <class Abi> ldoublev<Abi> remquo(ldoublev<Abi> x, ldoublev<Abi> y, samesize<int, ldoublev<Abi>>* quo);

template <class Abi> floatv<Abi> copysign(floatv<Abi> x, floatv<Abi> y);
template <class Abi> doublev<Abi> copysign(doublev<Abi> x, doublev<Abi> y);
template <class Abi> ldoublev<Abi> copysign(ldoublev<Abi> x, ldoublev<Abi> y);

template <class Abi> doublev<Abi> nan(const char* tagp);
template <class Abi> floatv<Abi> nanf(const char* tagp);
template <class Abi> ldoublev<Abi> nanl(const char* tagp);

template <class Abi> floatv<Abi> nextafter(floatv<Abi> x, floatv<Abi> y);
template <class Abi> doublev<Abi> nextafter(doublev<Abi> x, doublev<Abi> y);
template <class Abi> ldoublev<Abi> nextafter(ldoublev<Abi> x, ldoublev<Abi> y);

template <class Abi> floatv<Abi> nexttoward(floatv<Abi> x, ldoublev<Abi> y);
template <class Abi> doublev<Abi> nexttoward(doublev<Abi> x, ldoublev<Abi> y);
template <class Abi> ldoublev<Abi> nexttoward(ldoublev<Abi> x, ldoublev<Abi> y);

template <class Abi> floatv<Abi> fdim(floatv<Abi> x, floatv<Abi> y);
template <class Abi> doublev<Abi> fdim(doublev<Abi> x, doublev<Abi> y);
template <class Abi> ldoublev<Abi> fdim(ldoublev<Abi> x, ldoublev<Abi> y);

template <class Abi> floatv<Abi> fmax(floatv<Abi> x, floatv<Abi> y);
template <class Abi> doublev<Abi> fmax(doublev<Abi> x, doublev<Abi> y);
template <class Abi> ldoublev<Abi> fmax(ldoublev<Abi> x, ldoublev<Abi> y);

template <class Abi> floatv<Abi> fmin(floatv<Abi> x, floatv<Abi> y);
template <class Abi> doublev<Abi> fmin(doublev<Abi> x, doublev<Abi> y);
template <class Abi> ldoublev<Abi> fmin(ldoublev<Abi> x, ldoublev<Abi> y);

template <class Abi> floatv<Abi> fma(floatv<Abi> x, floatv<Abi> y, floatv<Abi> z);
template <class Abi> doublev<Abi> fma(doublev<Abi> x, doublev<Abi> y, doublev<Abi> z);
template <class Abi> ldoublev<Abi> fma(ldoublev<Abi> x, ldoublev<Abi> y, ldoublev<Abi> z);

template <class Abi> samesize<int, floatv<Abi>> fpclassify(floatv<Abi> x);
template <class Abi> samesize<int, doublev<Abi>> fpclassify(doublev<Abi> x);
template <class Abi> samesize<int, ldoublev<Abi>> fpclassify(ldoublev<Abi> x);

template <class Abi> simd_mask<float, Abi> isfinite(floatv<Abi> x);
template <class Abi> simd_mask<double, Abi> isfinite(doublev<Abi> x);
template <class Abi> simd_mask<long double, Abi> isfinite(ldoublev<Abi> x);

template <class Abi> simd_mask<float, Abi> isinf(floatv<Abi> x);
template <class Abi> simd_mask<double, Abi> isinf(doublev<Abi> x);
template <class Abi> simd_mask<long double, Abi> isinf(ldoublev<Abi> x);

template <class Abi> simd_mask<float, Abi> isnan(floatv<Abi> x);
template <class Abi> simd_mask<double, Abi> isnan(doublev<Abi> x);
template <class Abi> simd_mask<long double, Abi> isnan(ldoublev<Abi> x);

template <class Abi> simd_mask<float, Abi> isnormal(floatv<Abi> x);
template <class Abi> simd_mask<double, Abi> isnormal(doublev<Abi> x);
template <class Abi> simd_mask<long double, Abi> isnormal(ldoublev<Abi> x);

template <class Abi> simd_mask<float, Abi> signbit(floatv<Abi> x);
template <class Abi> simd_mask<double, Abi> signbit(doublev<Abi> x);
template <class Abi> simd_mask<long double, Abi> signbit(ldoublev<Abi> x);

template <class Abi> simd_mask<float, Abi> isgreater(floatv<Abi> x, floatv<Abi> y);
template <class Abi> simd_mask<double, Abi> isgreater(doublev<Abi> x, doublev<Abi> y);
template <class Abi> simd_mask<long double, Abi> isgreater(ldoublev<Abi> x, ldoublev<Abi> y);

template <class Abi> simd_mask<float, Abi> isgreaterequal(floatv<Abi> x, floatv<Abi> y);
template <class Abi> simd_mask<double, Abi> isgreaterequal(doublev<Abi> x, doublev<Abi> y);
template <class Abi> simd_mask<long double, Abi> isgreaterequal(ldoublev<Abi> x, ldoublev<Abi> y);

template <class Abi> simd_mask<float, Abi> isless(floatv<Abi> x, floatv<Abi> y);
template <class Abi> simd_mask<double, Abi> isless(doublev<Abi> x, doublev<Abi> y);
template <class Abi> simd_mask<long double, Abi> isless(ldoublev<Abi> x, ldoublev<Abi> y);

template <class Abi> simd_mask<float, Abi> islessequal(floatv<Abi> x, floatv<Abi> y);
template <class Abi> simd_mask<double, Abi> islessequal(doublev<Abi> x, doublev<Abi> y);
template <class Abi> simd_mask<long double, Abi> islessequal(ldoublev<Abi> x, ldoublev<Abi> y);

template <class Abi> simd_mask<float, Abi> islessgreater(floatv<Abi> x, floatv<Abi> y);
template <class Abi> simd_mask<double, Abi> islessgreater(doublev<Abi> x, doublev<Abi> y);
template <class Abi> simd_mask<long double, Abi> islessgreater(ldoublev<Abi> x, ldoublev<Abi> y);

template <class Abi> simd_mask<float, Abi> isunordered(floatv<Abi> x, floatv<Abi> y);
template <class Abi> simd_mask<double, Abi> isunordered(doublev<Abi> x, doublev<Abi> y);
template <class Abi> simd_mask<long double, Abi> isunordered(ldoublev<Abi> x, ldoublev<Abi> y);

template <class V> struct simd_div_t { V quot, rem; };
template <class Abi> simd_div_t<scharv<Abi>> div(scharv<Abi> numer, scharv<Abi> denom);
template <class Abi> simd_div_t<shortv<Abi>> div(shortv<Abi> numer, shortv<Abi> denom);
template <class Abi> simd_div_t<intv<Abi>> div(intv<Abi> numer, intv<Abi> denom);
template <class Abi> simd_div_t<longv<Abi>> div(longv<Abi> numer, longv<Abi> denom);
template <class Abi> simd_div_t<llongv<Abi>> div(llongv<Abi> numer, llongv<Abi> denom);

// [simd.mask.class]
template <class T, class Abi>
class simd_mask {
public:
  using value_type = bool;
  using reference = see below;
  using simd_type = simd<T, Abi>;
  using abi_type = Abi;
  static constexpr size_t size() noexcept;
  simd_mask() = default;

  // broadcast constructor
  explicit simd_mask(value_type) noexcept;

  // implicit type conversion constructor
  template <class U> simd_mask(const simd_mask<U, simd_abi::fixed_size<size()>>&) noexcept;

  // load constructor
  template <class Flags> simd_mask(const value_type* mem, Flags);

  // loads [simd.mask.copy]
  template <class Flags> void copy_from(const value_type* mem, Flags);
  template <class Flags> void copy_to(value_type* mem, Flags) const;

  // scalar access [simd.mask.subscr]
  reference operator[](size_t);
  value_type operator[](size_t) const;

  // unary operators [simd.mask.unary]
  simd_mask operator!() const noexcept;

  // simd_mask binary operators [simd.mask.binary]
  friend simd_mask operator&&(const simd_mask&, const simd_mask&) noexcept;
  friend simd_mask operator||(const simd_mask&, const simd_mask&) noexcept;
  friend simd_mask operator& (const simd_mask&, const simd_mask&) noexcept;
  friend simd_mask operator| (const simd_mask&, const simd_mask&) noexcept;
  friend simd_mask operator^ (const simd_mask&, const simd_mask&) noexcept;

  // simd_mask compound assignment [simd.mask.cassign]
  friend simd_mask& operator&=(simd_mask&, const simd_mask&) noexcept;
  friend simd_mask& operator|=(simd_mask&, const simd_mask&) noexcept;
  friend simd_mask& operator^=(simd_mask&, const simd_mask&) noexcept;

  // simd_mask compares [simd.mask.comparison]
  friend simd_mask operator==(const simd_mask&, const simd_mask&) noexcept;
  friend simd_mask operator!=(const simd_mask&, const simd_mask&) noexcept;
};

} // parallelism_v2
} // std::experimental

*/

#ifdef ENABLE_SYCL_EXT_ONEAPI_INVOKE_SIMD
#define _LIBCPP_STD_VER 17
#define _LIBCPP_COMPILER_CLANG_BASED 1
#define _LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER 1
#define _LIBCPP_BEGIN_NAMESPACE_EXPERIMENTAL_SIMD namespace std::experimental {
#define _LIBCPP_END_NAMESPACE_EXPERIMENTAL_SIMD }
#define _LIBCPP_BEGIN_NAMESPACE_EXPERIMENTAL_SIMD_ABI namespace std::experimental::simd_abi {
#define _LIBCPP_END_NAMESPACE_EXPERIMENTAL_SIMD_ABI }
#define _LIBCPP_INLINE_VAR inline
#define _LIBCPP_PUSH_MACROS
#define _LIBCPP_POP_MACROS
#define _LIBCPP_NATIVE_SIMD_WIDTH_IN_BYTES 512 // is not really used for now for sycl::ext::oneapi::experimental::invoke_simd

#include <algorithm>
#endif // ENABLE_SYCL_EXT_ONEAPI_INVOKE_SIMD



#include <array>
#include <cstddef>
#ifndef ENABLE_SYCL_EXT_ONEAPI_INVOKE_SIMD
#include <experimental/__config>
#endif // !ENABLE_SYCL_EXT_ONEAPI_INVOKE_SIMD
#ifdef ENABLE_SYCL_EXT_ONEAPI_INVOKE_SIMD
#include <cstdint>
#endif // ENABLE_SYCL_EXT_ONEAPI_INVOKE_SIMD
#include <functional>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#ifndef ENABLE_SYCL_EXT_ONEAPI_INVOKE_SIMD
#include <__undef_macros>
#endif // !ENABLE_SYCL_EXT_ONEAPI_INVOKE_SIMD


_LIBCPP_BEGIN_NAMESPACE_EXPERIMENTAL_SIMD

#if _LIBCPP_STD_VER >= 17

enum class _StorageKind {
  _Scalar,
  _Array,
  _VecExt,
};

template <_StorageKind __kind, int _Np>
struct __simd_abi {};

template <class _Tp, class _Abi>
class __simd_storage {};

template <class _Tp, int __num_element>
class __simd_storage<_Tp, __simd_abi<_StorageKind::_Array, __num_element>> {
  std::array<_Tp, __num_element> __storage_;

  template <class, class>
  friend struct simd;

  template <class, class>
  friend struct simd_mask;

public:
  _Tp __get(size_t __index) const noexcept { return __storage_[__index]; }
  void __set(size_t __index, _Tp __val) noexcept {
    __storage_[__index] = __val;
  }
};

template <class _Tp>
class __simd_storage<_Tp, __simd_abi<_StorageKind::_Scalar, 1>> {
  _Tp __storage_;

  template <class, class>
  friend struct simd;

  template <class, class>
  friend struct simd_mask;

public:
  _Tp __get(size_t __index) const noexcept { return (&__storage_)[__index]; }
  void __set(size_t __index, _Tp __val) noexcept {
    (&__storage_)[__index] = __val;
  }
};

#ifndef _LIBCPP_HAS_NO_VECTOR_EXTENSION

constexpr size_t __floor_pow_of_2(size_t __val) {
  return ((__val - 1) & __val) == 0 ? __val
                                    : __floor_pow_of_2((__val - 1) & __val);
}

constexpr size_t __ceil_pow_of_2(size_t __val) {
  return __val == 1 ? 1 : __floor_pow_of_2(__val - 1) << 1;
}

template <class _Tp, size_t __bytes>
struct __vec_ext_traits {
#if !defined(_LIBCPP_COMPILER_CLANG_BASED)
  typedef _Tp type __attribute__((vector_size(__ceil_pow_of_2(__bytes))));
#endif
};

#if defined(_LIBCPP_COMPILER_CLANG_BASED)
#define _LIBCPP_SPECIALIZE_VEC_EXT(_TYPE, _NUM_ELEMENT)                        \
  template <>                                                                  \
  struct __vec_ext_traits<_TYPE, sizeof(_TYPE) * _NUM_ELEMENT> {               \
    using type =                                                               \
        _TYPE __attribute__((vector_size(sizeof(_TYPE) * _NUM_ELEMENT)));      \
  }

#define _LIBCPP_SPECIALIZE_VEC_EXT_32(_TYPE)                                   \
  _LIBCPP_SPECIALIZE_VEC_EXT(_TYPE, 1);                                        \
  _LIBCPP_SPECIALIZE_VEC_EXT(_TYPE, 2);                                        \
  _LIBCPP_SPECIALIZE_VEC_EXT(_TYPE, 3);                                        \
  _LIBCPP_SPECIALIZE_VEC_EXT(_TYPE, 4);                                        \
  _LIBCPP_SPECIALIZE_VEC_EXT(_TYPE, 5);                                        \
  _LIBCPP_SPECIALIZE_VEC_EXT(_TYPE, 6);                                        \
  _LIBCPP_SPECIALIZE_VEC_EXT(_TYPE, 7);                                        \
  _LIBCPP_SPECIALIZE_VEC_EXT(_TYPE, 8);                                        \
  _LIBCPP_SPECIALIZE_VEC_EXT(_TYPE, 9);                                        \
  _LIBCPP_SPECIALIZE_VEC_EXT(_TYPE, 10);                                       \
  _LIBCPP_SPECIALIZE_VEC_EXT(_TYPE, 11);                                       \
  _LIBCPP_SPECIALIZE_VEC_EXT(_TYPE, 12);                                       \
  _LIBCPP_SPECIALIZE_VEC_EXT(_TYPE, 13);                                       \
  _LIBCPP_SPECIALIZE_VEC_EXT(_TYPE, 14);                                       \
  _LIBCPP_SPECIALIZE_VEC_EXT(_TYPE, 15);                                       \
  _LIBCPP_SPECIALIZE_VEC_EXT(_TYPE, 16);                                       \
  _LIBCPP_SPECIALIZE_VEC_EXT(_TYPE, 17);                                       \
  _LIBCPP_SPECIALIZE_VEC_EXT(_TYPE, 18);                                       \
  _LIBCPP_SPECIALIZE_VEC_EXT(_TYPE, 19);                                       \
  _LIBCPP_SPECIALIZE_VEC_EXT(_TYPE, 20);                                       \
  _LIBCPP_SPECIALIZE_VEC_EXT(_TYPE, 21);                                       \
  _LIBCPP_SPECIALIZE_VEC_EXT(_TYPE, 22);                                       \
  _LIBCPP_SPECIALIZE_VEC_EXT(_TYPE, 23);                                       \
  _LIBCPP_SPECIALIZE_VEC_EXT(_TYPE, 24);                                       \
  _LIBCPP_SPECIALIZE_VEC_EXT(_TYPE, 25);                                       \
  _LIBCPP_SPECIALIZE_VEC_EXT(_TYPE, 26);                                       \
  _LIBCPP_SPECIALIZE_VEC_EXT(_TYPE, 27);                                       \
  _LIBCPP_SPECIALIZE_VEC_EXT(_TYPE, 28);                                       \
  _LIBCPP_SPECIALIZE_VEC_EXT(_TYPE, 29);                                       \
  _LIBCPP_SPECIALIZE_VEC_EXT(_TYPE, 30);                                       \
  _LIBCPP_SPECIALIZE_VEC_EXT(_TYPE, 31);                                       \
  _LIBCPP_SPECIALIZE_VEC_EXT(_TYPE, 32)

_LIBCPP_SPECIALIZE_VEC_EXT_32(char);
_LIBCPP_SPECIALIZE_VEC_EXT_32(char16_t);
_LIBCPP_SPECIALIZE_VEC_EXT_32(char32_t);
_LIBCPP_SPECIALIZE_VEC_EXT_32(wchar_t);
_LIBCPP_SPECIALIZE_VEC_EXT_32(signed char);
_LIBCPP_SPECIALIZE_VEC_EXT_32(signed short);
_LIBCPP_SPECIALIZE_VEC_EXT_32(signed int);
_LIBCPP_SPECIALIZE_VEC_EXT_32(signed long);
_LIBCPP_SPECIALIZE_VEC_EXT_32(signed long long);
_LIBCPP_SPECIALIZE_VEC_EXT_32(unsigned char);
_LIBCPP_SPECIALIZE_VEC_EXT_32(unsigned short);
_LIBCPP_SPECIALIZE_VEC_EXT_32(unsigned int);
_LIBCPP_SPECIALIZE_VEC_EXT_32(unsigned long);
_LIBCPP_SPECIALIZE_VEC_EXT_32(unsigned long long);
_LIBCPP_SPECIALIZE_VEC_EXT_32(float);
_LIBCPP_SPECIALIZE_VEC_EXT_32(double);
_LIBCPP_SPECIALIZE_VEC_EXT_32(long double);

#undef _LIBCPP_SPECIALIZE_VEC_EXT_32
#undef _LIBCPP_SPECIALIZE_VEC_EXT
#endif

template <class _Tp, int __num_element>
class __simd_storage<_Tp, __simd_abi<_StorageKind::_VecExt, __num_element>> {
  using _StorageType =
      typename __vec_ext_traits<_Tp, sizeof(_Tp) * __num_element>::type;

  _StorageType __storage_;

  template <class, class>
  friend struct simd;

  template <class, class>
  friend struct simd_mask;

public:
  _Tp __get(size_t __index) const noexcept { return __storage_[__index]; }
  void __set(size_t __index, _Tp __val) noexcept {
    __storage_[__index] = __val;
  }
#ifdef ENABLE_SYCL_EXT_ONEAPI_INVOKE_SIMD
  const _StorageType& data() const noexcept { return __storage_; }
#endif
};

#endif // _LIBCPP_HAS_NO_VECTOR_EXTENSION

template <class _Vp, class _Tp, class _Abi>
class __simd_reference {
  static_assert(std::is_same_v<_Vp, _Tp>, "");

  template <class, class>
  friend struct simd;

  template <class, class>
  friend struct simd_mask;

  __simd_storage<_Tp, _Abi>* __ptr_;
  size_t __index_;

  __simd_reference(__simd_storage<_Tp, _Abi>* __ptr, size_t __index)
      : __ptr_(__ptr), __index_(__index) {}

  __simd_reference(const __simd_reference&) = default;

public:
  __simd_reference() = delete;
  __simd_reference& operator=(const __simd_reference&) = delete;

  operator _Vp() const { return __ptr_->__get(__index_); }

  __simd_reference operator=(_Vp __value) && {
    __ptr_->__set(__index_, __value);
    return *this;
  }

  __simd_reference operator++() && {
    return std::move(*this) = __ptr_->__get(__index_) + 1;
  }

  _Vp operator++(int) && {
    auto __val = __ptr_->__get(__index_);
    __ptr_->__set(__index_, __val + 1);
    return __val;
  }

  __simd_reference operator--() && {
    return std::move(*this) = __ptr_->__get(__index_) - 1;
  }

  _Vp operator--(int) && {
    auto __val = __ptr_->__get(__index_);
    __ptr_->__set(__index_, __val - 1);
    return __val;
  }

  __simd_reference operator+=(_Vp __value) && {
    return std::move(*this) = __ptr_->__get(__index_) + __value;
  }

  __simd_reference operator-=(_Vp __value) && {
    return std::move(*this) = __ptr_->__get(__index_) - __value;
  }

  __simd_reference operator*=(_Vp __value) && {
    return std::move(*this) = __ptr_->__get(__index_) * __value;
  }

  __simd_reference operator/=(_Vp __value) && {
    return std::move(*this) = __ptr_->__get(__index_) / __value;
  }

  __simd_reference operator%=(_Vp __value) && {
    return std::move(*this) = __ptr_->__get(__index_) % __value;
  }

  __simd_reference operator>>=(_Vp __value) && {
    return std::move(*this) = __ptr_->__get(__index_) >> __value;
  }

  __simd_reference operator<<=(_Vp __value) && {
    return std::move(*this) = __ptr_->__get(__index_) << __value;
  }

  __simd_reference operator&=(_Vp __value) && {
    return std::move(*this) = __ptr_->__get(__index_) & __value;
  }

  __simd_reference operator|=(_Vp __value) && {
    return std::move(*this) = __ptr_->__get(__index_) | __value;
  }

  __simd_reference operator^=(_Vp __value) && {
    return std::move(*this) = __ptr_->__get(__index_) ^ __value;
  }
};

template <class _To, class _From>
constexpr decltype(_To{std::declval<_From>()}, true)
__is_non_narrowing_convertible_impl(_From) {
  return true;
}

template <class _To>
constexpr bool __is_non_narrowing_convertible_impl(...) {
  return false;
}

template <class _From, class _To>
constexpr std::enable_if_t<std::is_arithmetic_v<_To> &&
                                      std::is_arithmetic_v<_From>,
                                  bool>
__is_non_narrowing_arithmetic_convertible() {
  return __is_non_narrowing_convertible_impl<_To>(_From{});
}

template <class _From, class _To>
constexpr std::enable_if_t<!(std::is_arithmetic_v<_To> &&
                                    std::is_arithmetic_v<_From>),
                                  bool>
__is_non_narrowing_arithmetic_convertible() {
  return false;
}

template <class _Tp>
constexpr _Tp __variadic_sum() {
  return _Tp{};
}

template <class _Tp, class _Up, class... _Args>
constexpr _Tp __variadic_sum(_Up __first, _Args... __rest) {
  return static_cast<_Tp>(__first) + __variadic_sum<_Tp>(__rest...);
}

template <class _Tp>
struct __nodeduce {
  using type = _Tp;
};

template <class _Tp>
constexpr bool __vectorizable() {
  return std::is_arithmetic_v<_Tp> && !std::is_const_v<_Tp> &&
         !std::is_volatile_v<_Tp> && !std::is_same_v<_Tp, bool>;
}

_LIBCPP_END_NAMESPACE_EXPERIMENTAL_SIMD
_LIBCPP_BEGIN_NAMESPACE_EXPERIMENTAL_SIMD_ABI

using scalar = __simd_abi<_StorageKind::_Scalar, 1>;

template <int _Np>
using fixed_size = __simd_abi<_StorageKind::_Array, _Np>;

template <class _Tp>
inline constexpr size_t max_fixed_size = 32;

template <class _Tp>
using compatible = fixed_size<16 / sizeof(_Tp)>;

#ifndef _LIBCPP_HAS_NO_VECTOR_EXTENSION
template <class _Tp>
using native = __simd_abi<_StorageKind::_VecExt,
                          _LIBCPP_NATIVE_SIMD_WIDTH_IN_BYTES / sizeof(_Tp)>;
#else
template <class _Tp>
using native =
    fixed_size<_Tp, _LIBCPP_NATIVE_SIMD_WIDTH_IN_BYTES / sizeof(_Tp)>;
#endif // _LIBCPP_HAS_NO_VECTOR_EXTENSION

_LIBCPP_END_NAMESPACE_EXPERIMENTAL_SIMD_ABI
_LIBCPP_BEGIN_NAMESPACE_EXPERIMENTAL_SIMD

template <class _Tp, class _Abi = simd_abi::compatible<_Tp>>
class simd;
template <class _Tp, class _Abi = simd_abi::compatible<_Tp>>
class simd_mask;

struct element_aligned_tag {};
struct vector_aligned_tag {};
template <size_t>
struct overaligned_tag {};
inline constexpr element_aligned_tag element_aligned{};
inline constexpr vector_aligned_tag vector_aligned{};
template <size_t _Np>
inline constexpr overaligned_tag<_Np> overaligned{};

// traits [simd.traits]
template <class _Tp>
struct is_abi_tag : std::integral_constant<bool, false> {};

template <_StorageKind __kind, int _Np>
struct is_abi_tag<__simd_abi<__kind, _Np>>
    : std::integral_constant<bool, true> {};

template <class _Tp>
struct is_simd : std::integral_constant<bool, false> {};

template <class _Tp, class _Abi>
struct is_simd<simd<_Tp, _Abi>> : std::integral_constant<bool, true> {};

template <class _Tp>
struct is_simd_mask : std::integral_constant<bool, false> {};

template <class _Tp, class _Abi>
struct is_simd_mask<simd_mask<_Tp, _Abi>> : std::integral_constant<bool, true> {
};

template <class _Tp>
struct is_simd_flag_type : std::integral_constant<bool, false> {};

template <>
struct is_simd_flag_type<element_aligned_tag>
    : std::integral_constant<bool, true> {};

template <>
struct is_simd_flag_type<vector_aligned_tag>
    : std::integral_constant<bool, true> {};

template <size_t _Align>
struct is_simd_flag_type<overaligned_tag<_Align>>
    : std::integral_constant<bool, true> {};

template <class _Tp>
inline constexpr bool is_abi_tag_v = is_abi_tag<_Tp>::value;
template <class _Tp>
inline constexpr bool is_simd_v = is_simd<_Tp>::value;
template <class _Tp>
inline constexpr bool is_simd_mask_v = is_simd_mask<_Tp>::value;
template <class _Tp>
inline constexpr bool is_simd_flag_type_v = is_simd_flag_type<_Tp>::value;
template <class _Tp, size_t _Np>
struct abi_for_size {
  using type = simd_abi::fixed_size<_Np>;
};
template <class _Tp, size_t _Np>
using abi_for_size_t = typename abi_for_size<_Tp, _Np>::type;

template <class _Tp, class _Abi = simd_abi::compatible<_Tp>>
struct simd_size;

template <class _Tp, _StorageKind __kind, int _Np>
struct simd_size<_Tp, __simd_abi<__kind, _Np>>
    : std::integral_constant<size_t, _Np> {
  static_assert(
      std::is_arithmetic_v<_Tp> &&
          !std::is_same_v<std::remove_const_t<_Tp>, bool>,
      "Element type should be vectorizable");
};

// TODO: implement it.
template <class _Tp, class _Up = typename _Tp::value_type>
struct memory_alignment;

template <class _Tp, class _Abi = simd_abi::compatible<_Tp>>
inline constexpr size_t simd_size_v = simd_size<_Tp, _Abi>::value;

template <class _Tp, class _Up = typename _Tp::value_type>
inline constexpr size_t memory_alignment_v = memory_alignment<_Tp, _Up>::value;

// class template simd [simd.class]
template <class _Tp>
using native_simd = simd<_Tp, simd_abi::native<_Tp>>;
template <class _Tp, int _Np>
using fixed_size_simd = simd<_Tp, simd_abi::fixed_size<_Np>>;

// class template simd_mask [simd.mask.class]
template <class _Tp>
using native_simd_mask = simd_mask<_Tp, simd_abi::native<_Tp>>;

template <class _Tp, int _Np>
using fixed_size_simd_mask = simd_mask<_Tp, simd_abi::fixed_size<_Np>>;

// casts [simd.casts]
template <class _Tp>
struct __static_simd_cast_traits {
  template <class _Up, class _Abi>
  static simd<_Tp, _Abi> __apply(const simd<_Up, _Abi>& __v);
};

template <class _Tp, class _NewAbi>
struct __static_simd_cast_traits<simd<_Tp, _NewAbi>> {
  template <class _Up, class _Abi>
  static std::enable_if_t<simd<_Up, _Abi>::size() ==
                                     simd<_Tp, _NewAbi>::size(),
                                 simd<_Tp, _NewAbi>>
  __apply(const simd<_Up, _Abi>& __v);
};

template <class _Tp>
struct __simd_cast_traits {
  template <class _Up, class _Abi>
  static std::enable_if_t<
      __is_non_narrowing_arithmetic_convertible<_Up, _Tp>(),
      simd<_Tp, _Abi>>
  __apply(const simd<_Up, _Abi>& __v);
};

template <class _Tp, class _NewAbi>
struct __simd_cast_traits<simd<_Tp, _NewAbi>> {
  template <class _Up, class _Abi>
  static std::enable_if_t<
      __is_non_narrowing_arithmetic_convertible<_Up, _Tp>() &&
          simd<_Up, _Abi>::size() == simd<_Tp, _NewAbi>::size(),
      simd<_Tp, _NewAbi>>
  __apply(const simd<_Up, _Abi>& __v);
};

template <class _Tp, class _Up, class _Abi>
auto simd_cast(const simd<_Up, _Abi>& __v)
    -> decltype(__simd_cast_traits<_Tp>::__apply(__v)) {
  return __simd_cast_traits<_Tp>::__apply(__v);
}

template <class _Tp, class _Up, class _Abi>
auto static_simd_cast(const simd<_Up, _Abi>& __v)
    -> decltype(__static_simd_cast_traits<_Tp>::__apply(__v)) {
  return __static_simd_cast_traits<_Tp>::__apply(__v);
}

template <class _Tp, class _Abi>
fixed_size_simd<_Tp, simd_size<_Tp, _Abi>::value>
to_fixed_size(const simd<_Tp, _Abi>&) noexcept;

template <class _Tp, class _Abi>
fixed_size_simd_mask<_Tp, simd_size<_Tp, _Abi>::value>
to_fixed_size(const simd_mask<_Tp, _Abi>&) noexcept;

template <class _Tp, size_t _Np>
native_simd<_Tp> to_native(const fixed_size_simd<_Tp, _Np>&) noexcept;

template <class _Tp, size_t _Np>
native_simd_mask<_Tp> to_native(const fixed_size_simd_mask<_Tp, _Np>&) noexcept;

template <class _Tp, size_t _Np>
simd<_Tp> to_compatible(const fixed_size_simd<_Tp, _Np>&) noexcept;

template <class _Tp, size_t _Np>
simd_mask<_Tp> to_compatible(const fixed_size_simd_mask<_Tp, _Np>&) noexcept;

template <size_t... __sizes, class _Tp, class _Abi>
tuple<simd<_Tp, abi_for_size_t<_Tp, __sizes>>...> split(const simd<_Tp, _Abi>&);

template <size_t... __sizes, class _Tp, class _Abi>
tuple<simd_mask<_Tp, abi_for_size_t<_Tp, __sizes>>...>
split(const simd_mask<_Tp, _Abi>&);

template <class _SimdType, class _Abi>
array<_SimdType, simd_size<typename _SimdType::value_type, _Abi>::value /
                     _SimdType::size()>
split(const simd<typename _SimdType::value_type, _Abi>&);

template <class _SimdType, class _Abi>
array<_SimdType, simd_size<typename _SimdType::value_type, _Abi>::value /
                     _SimdType::size()>
split(const simd_mask<typename _SimdType::value_type, _Abi>&);

template <class _Tp, class... _Abis>
simd<_Tp, abi_for_size_t<_Tp, __variadic_sum(simd_size<_Tp, _Abis>::value...)>>
concat(const simd<_Tp, _Abis>&...);

template <class _Tp, class... _Abis>
simd_mask<_Tp,
          abi_for_size_t<_Tp, __variadic_sum(simd_size<_Tp, _Abis>::value...)>>
concat(const simd_mask<_Tp, _Abis>&...);

// reductions [simd.mask.reductions]
template <class _Tp, class _Abi>
bool all_of(const simd_mask<_Tp, _Abi>&) noexcept;
template <class _Tp, class _Abi>
bool any_of(const simd_mask<_Tp, _Abi>&) noexcept;
template <class _Tp, class _Abi>
bool none_of(const simd_mask<_Tp, _Abi>&) noexcept;
template <class _Tp, class _Abi>
bool some_of(const simd_mask<_Tp, _Abi>&) noexcept;
template <class _Tp, class _Abi>
int popcount(const simd_mask<_Tp, _Abi>&) noexcept;
template <class _Tp, class _Abi>
int find_first_set(const simd_mask<_Tp, _Abi>&);
template <class _Tp, class _Abi>
int find_last_set(const simd_mask<_Tp, _Abi>&);
bool all_of(bool) noexcept;
bool any_of(bool) noexcept;
bool none_of(bool) noexcept;
bool some_of(bool) noexcept;
int popcount(bool) noexcept;
int find_first_set(bool) noexcept;
int find_last_set(bool) noexcept;

// masked assignment [simd.whereexpr]
template <class _MaskType, class _Tp>
class const_where_expression;
template <class _MaskType, class _Tp>
class where_expression;

// masked assignment [simd.mask.where]
template <class _Tp, class _Abi>
where_expression<simd_mask<_Tp, _Abi>, simd<_Tp, _Abi>>
where(const typename simd<_Tp, _Abi>::mask_type&, simd<_Tp, _Abi>&) noexcept;

template <class _Tp, class _Abi>
const_where_expression<simd_mask<_Tp, _Abi>, const simd<_Tp, _Abi>>
where(const typename simd<_Tp, _Abi>::mask_type&,
      const simd<_Tp, _Abi>&) noexcept;

template <class _Tp, class _Abi>
where_expression<simd_mask<_Tp, _Abi>, simd_mask<_Tp, _Abi>>
where(const typename __nodeduce<simd_mask<_Tp, _Abi>>::type&,
      simd_mask<_Tp, _Abi>&) noexcept;

template <class _Tp, class _Abi>
const_where_expression<simd_mask<_Tp, _Abi>, const simd_mask<_Tp, _Abi>>
where(const typename __nodeduce<simd_mask<_Tp, _Abi>>::type&,
      const simd_mask<_Tp, _Abi>&) noexcept;

template <class _Tp>
where_expression<bool, _Tp> where(bool, _Tp&) noexcept;

template <class _Tp>
const_where_expression<bool, const _Tp> where(bool, const _Tp&) noexcept;

// reductions [simd.reductions]
template <class _Tp, class _Abi, class _BinaryOp = std::plus<_Tp>>
_Tp reduce(const simd<_Tp, _Abi>&, _BinaryOp = _BinaryOp());

template <class _MaskType, class _SimdType, class _BinaryOp>
typename _SimdType::value_type
reduce(const const_where_expression<_MaskType, _SimdType>&,
       typename _SimdType::value_type neutral_element, _BinaryOp binary_op);

template <class _MaskType, class _SimdType>
typename _SimdType::value_type
reduce(const const_where_expression<_MaskType, _SimdType>&,
       plus<typename _SimdType::value_type> binary_op = {});

template <class _MaskType, class _SimdType>
typename _SimdType::value_type
reduce(const const_where_expression<_MaskType, _SimdType>&,
       multiplies<typename _SimdType::value_type> binary_op);

template <class _MaskType, class _SimdType>
typename _SimdType::value_type
reduce(const const_where_expression<_MaskType, _SimdType>&,
       bit_and<typename _SimdType::value_type> binary_op);

template <class _MaskType, class _SimdType>
typename _SimdType::value_type
reduce(const const_where_expression<_MaskType, _SimdType>&,
       bit_or<typename _SimdType::value_type> binary_op);

template <class _MaskType, class _SimdType>
typename _SimdType::value_type
reduce(const const_where_expression<_MaskType, _SimdType>&,
       bit_xor<typename _SimdType::value_type> binary_op);

template <class _Tp, class _Abi>
_Tp hmin(const simd<_Tp, _Abi>&);
template <class _MaskType, class _SimdType>
typename _SimdType::value_type
hmin(const const_where_expression<_MaskType, _SimdType>&);
template <class _Tp, class _Abi>
_Tp hmax(const simd<_Tp, _Abi>&);
template <class _MaskType, class _SimdType>
typename _SimdType::value_type
hmax(const const_where_expression<_MaskType, _SimdType>&);

// algorithms [simd.alg]
template <class _Tp, class _Abi>
simd<_Tp, _Abi> (min)(const simd<_Tp, _Abi>&, const simd<_Tp, _Abi>&) noexcept;

template <class _Tp, class _Abi>
simd<_Tp, _Abi> (max)(const simd<_Tp, _Abi>&, const simd<_Tp, _Abi>&) noexcept;

template <class _Tp, class _Abi>
std::pair<simd<_Tp, _Abi>, simd<_Tp, _Abi>>
minmax(const simd<_Tp, _Abi>&, const simd<_Tp, _Abi>&) noexcept;

template <class _Tp, class _Abi>
simd<_Tp, _Abi> clamp(const simd<_Tp, _Abi>&, const simd<_Tp, _Abi>&,
                      const simd<_Tp, _Abi>&);

// [simd.whereexpr]
// TODO implement where expressions.
template <class _MaskType, class _Tp>
class const_where_expression {
public:
  const_where_expression(const const_where_expression&) = delete;
  const_where_expression& operator=(const const_where_expression&) = delete;
  remove_const_t<_Tp>operator-() const&&;
  template <class _Up, class _Flags>
  void copy_to(_Up*, _Flags) const&&;
};

template <class _MaskType, class _Tp>
class where_expression : public const_where_expression<_MaskType, _Tp> {
public:
  where_expression(const where_expression&) = delete;
  where_expression& operator=(const where_expression&) = delete;
  template <class _Up>
  void operator=(_Up&&);
  template <class _Up>
  void operator+=(_Up&&);
  template <class _Up>
  void operator-=(_Up&&);
  template <class _Up>
  void operator*=(_Up&&);
  template <class _Up>
  void operator/=(_Up&&);
  template <class _Up>
  void operator%=(_Up&&);
  template <class _Up>
  void operator&=(_Up&&);
  template <class _Up>
  void operator|=(_Up&&);
  template <class _Up>
  void operator^=(_Up&&);
  template <class _Up>
  void operator<<=(_Up&&);
  template <class _Up>
  void operator>>=(_Up&&);
  void operator++();
  void operator++(int);
  void operator--();
  void operator--(int);
  template <class _Up, class _Flags>
  void copy_from(const _Up*, _Flags);
};

// [simd.class]
// TODO: implement simd
template <class _Tp, class _Abi>
class simd {
#ifdef ENABLE_SYCL_EXT_ONEAPI_INVOKE_SIMD
  template <class, class> friend class simd;
#endif // ENABLE_SYCL_EXT_ONEAPI_INVOKE_SIMD
public:
  using value_type = _Tp;
  using reference = __simd_reference<_Tp, _Tp, _Abi>;
  using mask_type = simd_mask<_Tp, _Abi>;
  using abi_type = _Abi;

  simd() = default;
  simd(const simd&) = default;
  simd& operator=(const simd&) = default;

  static constexpr size_t size() noexcept {
    return simd_size<_Tp, _Abi>::value;
  }

private:
  __simd_storage<_Tp, _Abi> __s_;

#ifdef ENABLE_SYCL_EXT_ONEAPI_INVOKE_SIMD
  // TODO Temp implemenation to convert between esimd::simd and oneapi::simd.
  // Implement free conversion functions instead.
public:
  // TODO Won't compile for non-VecExt, maybe _StorageType should be added to
  // all ABIs.
  using raw_storage_type = typename __simd_storage<_Tp, _Abi>::_StorageType;
  
  // implicit conversion to storage type
  operator raw_storage_type() const { return __s_.__storage_; }
  
  // implicit conversion from storage type
  simd(const raw_storage_type &__raw_simd) { __s_.__storage_ = __raw_simd; }

#endif // ENABLE_SYCL_EXT_ONEAPI_INVOKE_SIMD

private:
  template <class _Up>
  static constexpr bool __can_broadcast() {
    return (std::is_arithmetic_v<_Up> &&
            __is_non_narrowing_arithmetic_convertible<_Up, _Tp>()) ||
           (!std::is_arithmetic_v<_Up> &&
            std::is_convertible_v<_Up, _Tp>) ||
           std::is_same_v<std::remove_const_t<_Up>, int> ||
           (std::is_same_v<std::remove_const_t<_Up>,
                         unsigned int> &&
            std::is_unsigned_v<_Tp>);
  }

  template <class _Generator, size_t... __indicies>
  static constexpr decltype(
      std::forward_as_tuple(std::declval<_Generator>()(
          std::integral_constant<size_t, __indicies>())...),
      bool())
  __can_generate(std::index_sequence<__indicies...>) {
    return !__variadic_sum<bool>(
        !__can_broadcast<decltype(std::declval<_Generator>()(
            std::integral_constant<size_t, __indicies>()))>()...);
  }

  template <class _Generator>
  static bool __can_generate(...) {
    return false;
  }

  template <class _Generator, size_t... __indicies>
  void __generator_init(_Generator&& __g, std::index_sequence<__indicies...>) {
    int __not_used[]{((*this)[__indicies] =
                          __g(std::integral_constant<size_t, __indicies>()),
                      0)...};
    (void)__not_used;
  }

public:
  // implicit type conversion constructor
#ifdef ENABLE_SYCL_EXT_ONEAPI_INVOKE_SIMD
  template <class _Up,
    class = std::enable_if_t<
      std::is_same_v<_Abi, __simd_abi<_StorageKind::_VecExt, size()>> &&
    __is_non_narrowing_arithmetic_convertible<_Up, _Tp>()>>
    simd(const simd<_Up, _Abi>& __v) {
    __s_.__storage_ = __builtin_convertvector(__v.__s_.__storage_, raw_storage_type);
  }
#endif // ENABLE_SYCL_EXT_ONEAPI_INVOKE_SIMD
  template <class _Up,
            class = std::enable_if_t<
                std::is_same_v<_Abi, simd_abi::fixed_size<size()>> &&
                __is_non_narrowing_arithmetic_convertible<_Up, _Tp>()>>
  simd(const simd<_Up, simd_abi::fixed_size<size()>>& __v) {
    for (size_t __i = 0; __i < size(); __i++) {
      (*this)[__i] = static_cast<_Tp>(__v[__i]);
    }
  }

  // implicit broadcast constructor
  template <class _Up,
            class = std::enable_if_t<__can_broadcast<_Up>()>>
  simd(_Up&& __rv) {
    auto __v = static_cast<_Tp>(__rv);
    for (size_t __i = 0; __i < size(); __i++) {
      (*this)[__i] = __v;
    }
  }

  // generator constructor
  template <class _Generator,
            int = std::enable_if_t<
                __can_generate<_Generator>(std::make_index_sequence<size()>()),
                int>()>
  explicit simd(_Generator&& __g) {
    __generator_init(std::forward<_Generator>(__g),
                     std::make_index_sequence<size()>());
  }

  // load constructor
  template <
      class _Up, class _Flags,
      class = std::enable_if_t<__vectorizable<_Up>()>,
      class = std::enable_if_t<is_simd_flag_type<_Flags>::value>>
  simd(const _Up* __buffer, _Flags) {
    // TODO: optimize for overaligned flags
    for (size_t __i = 0; __i < size(); __i++) {
      (*this)[__i] = static_cast<_Tp>(__buffer[__i]);
    }
  }

  // loads [simd.load]
  template <class _Up, class _Flags>
  std::enable_if_t<__vectorizable<_Up>() &&
                          is_simd_flag_type<_Flags>::value>
  copy_from(const _Up* __buffer, _Flags) {
    *this = simd(__buffer, _Flags());
  }

  // stores [simd.store]
  template <class _Up, class _Flags>
  std::enable_if_t<__vectorizable<_Up>() &&
                          is_simd_flag_type<_Flags>::value>
  copy_to(_Up* __buffer, _Flags) const {
    // TODO: optimize for overaligned flags
    for (size_t __i = 0; __i < size(); __i++) {
      __buffer[__i] = static_cast<_Up>((*this)[__i]);
    }
  }

  // scalar access [simd.subscr]
  reference operator[](size_t __i) { return reference(&__s_, __i); }

  value_type operator[](size_t __i) const { return __s_.__get(__i); }

  // unary operators [simd.unary]
  simd& operator++();
  simd operator++(int);
  simd& operator--();
  simd operator--(int);
  mask_type operator!() const;
  simd operator~() const;
  simd operator+() const;
  simd operator-() const;

  // binary operators [simd.binary]
  friend simd operator+(const simd&, const simd&);
  friend simd operator-(const simd&, const simd&);
  friend simd operator*(const simd&, const simd&);
  friend simd operator/(const simd&, const simd&);
  friend simd operator%(const simd&, const simd&);
  friend simd operator&(const simd&, const simd&);
  friend simd operator|(const simd&, const simd&);
  friend simd operator^(const simd&, const simd&);
  friend simd operator<<(const simd&, const simd&);
  friend simd operator>>(const simd&, const simd&);
  friend simd operator<<(const simd&, int);
  friend simd operator>>(const simd&, int);

  // compound assignment [simd.cassign]
  friend simd& operator+=(simd&, const simd&);
  friend simd& operator-=(simd&, const simd&);
  friend simd& operator*=(simd&, const simd&);
  friend simd& operator/=(simd&, const simd&);
  friend simd& operator%=(simd&, const simd&);

  friend simd& operator&=(simd&, const simd&);
  friend simd& operator|=(simd&, const simd&);
  friend simd& operator^=(simd&, const simd&);
  friend simd& operator<<=(simd&, const simd&);
  friend simd& operator>>=(simd&, const simd&);
  friend simd& operator<<=(simd&, int);
  friend simd& operator>>=(simd&, int);

  // compares [simd.comparison]
  friend mask_type operator==(const simd&, const simd&);
  friend mask_type operator!=(const simd&, const simd&);
  friend mask_type operator>=(const simd&, const simd&);
  friend mask_type operator<=(const simd&, const simd&);
  friend mask_type operator>(const simd&, const simd&);
  //friend mask_type operator<(const simd&, const simd&);
};

#ifdef ENABLE_SYCL_EXT_ONEAPI_INVOKE_SIMD
template <class _Abi>
struct __abi_storage_kind : public std::false_type {};

template <_StorageKind _K, int _Np>
struct __abi_storage_kind<__simd_abi<_K, _Np>> : public std::true_type {
  static constexpr _StorageKind value = _K;
};

template <typename _Tp, class _Abi> struct __mask_element {
  static_assert(__abi_storage_kind<_Abi>::value == _StorageKind::_VecExt,
                "only clang vector-based ABI is supported for now.");
  using type =
    std::conditional_t<sizeof(_Tp) == 1, uint8_t,
    std::conditional_t<sizeof(_Tp) == 2, uint16_t,
    std::conditional_t<sizeof(_Tp) == 4, uint32_t,
    std::conditional_t<sizeof(_Tp) == 8, uint64_t, void>>>>;
};

// Represents a reference to a simd_mask object element.
template <class _Tp, class _Abi>
class __simd_mask_reference {
private:
  using _Vp = bool;
  using _simd_mask_element_type = typename __mask_element<_Tp, _Abi>::type;

public:
  template <class, class>
  friend struct simd_mask;

  __simd_storage<_simd_mask_element_type, _Abi>* __ptr_;
  size_t __index_;

  __simd_mask_reference(__simd_storage<_Tp, _Abi>* __ptr, size_t __index)
    : __ptr_(__ptr), __index_(__index) {}

  __simd_mask_reference(const __simd_mask_reference&) = default;

  __simd_mask_reference() = delete;
  __simd_mask_reference& operator=(const __simd_mask_reference&) = delete;

  operator _Vp() const { return (_Vp)__ptr_->__get(__index_); }

  __simd_mask_reference operator=(_Vp __value) && {
    __ptr_->__set(__index_, (_Tp)__value);
    return *this;
  }

  __simd_mask_reference operator&=(_Vp __value) && {
    return std::move(*this) = ((_Vp)__ptr_->__get(__index_)) & __value;
  }

  __simd_mask_reference operator|=(_Vp __value) && {
    return std::move(*this) = ((_Vp)__ptr_->__get(__index_)) | __value;
  }

  __simd_mask_reference operator^=(_Vp __value) && {
    return std::move(*this) = ((_Vp)__ptr_->__get(__index_)) ^ __value;
  }
};
#endif // ENABLE_SYCL_EXT_ONEAPI_INVOKE_SIMD

// [simd.mask.class]
template <class _Tp, class _Abi>
// TODO: implement simd_mask
class simd_mask {
#ifdef ENABLE_SYCL_EXT_ONEAPI_INVOKE_SIMD
  using element_type = typename __mask_element<_Tp, _Abi>::type;
#endif // ENABLE_SYCL_EXT_ONEAPI_INVOKE_SIMD
public:
  using value_type = bool;
#ifdef ENABLE_SYCL_EXT_ONEAPI_INVOKE_SIMD
  using reference = __simd_mask_reference<element_type, _Abi>;
#else
  // TODO: this is strawman implementation. Turn it into a proxy type.
  using reference = bool&;
#endif // ENABLE_SYCL_EXT_ONEAPI_INVOKE_SIMD
  using simd_type = simd<_Tp, _Abi>;
  using abi_type = _Abi;

#ifdef ENABLE_SYCL_EXT_ONEAPI_INVOKE_SIMD
  static constexpr size_t size() noexcept { return simd_type::size(); }
#else
  static constexpr size_t size() noexcept;
#endif // ENABLE_SYCL_EXT_ONEAPI_INVOKE_SIMD

#ifdef ENABLE_SYCL_EXT_ONEAPI_INVOKE_SIMD
  const auto& data() const noexcept { return __s_.data(); }
#endif

  simd_mask() = default;

  // broadcast constructor
#ifdef ENABLE_SYCL_EXT_ONEAPI_INVOKE_SIMD
  // TODO inefficient, use this's storage directly
  explicit simd_mask(value_type __v) noexcept {
    for (size_t __i = 0; __i < size(); __i++) {
      (*this)[__i] = __v;
    }
  }
#else
  explicit simd_mask(value_type) noexcept;
#endif // ENABLE_SYCL_EXT_ONEAPI_INVOKE_SIMD

  // implicit type conversion constructor
#ifdef ENABLE_SYCL_EXT_ONEAPI_INVOKE_SIMD
  template <class _Up>
  simd_mask(const simd_mask<_Up, simd_abi::fixed_size<size()>>& __v) noexcept {
    copyElements(__v);
  }

  template <class _Up>
  simd_mask(const simd_mask<_Up, abi_type>& __v) noexcept {
    copyElements(__v);
  }
#else
  template <class _Up>
  simd_mask(const simd_mask<_Up, simd_abi::fixed_size<size()>>&) noexcept;
#endif // ENABLE_SYCL_EXT_ONEAPI_INVOKE_SIMD

  // load constructor
  template <class _Flags>
  simd_mask(const value_type*, _Flags);

  // loads [simd.mask.copy]
  template <class _Flags>
  void copy_from(const value_type*, _Flags);
  template <class _Flags>
  void copy_to(value_type*, _Flags) const;

  // scalar access [simd.mask.subscr]
#ifdef ENABLE_SYCL_EXT_ONEAPI_INVOKE_SIMD
  reference operator[](size_t __i) { return reference(&__s_, __i); }
  value_type operator[](size_t __i) const { return __s_.__get(__i) != 0; }
#else
  reference operator[](size_t);
  value_type operator[](size_t) const;
#endif // ENABLE_SYCL_EXT_ONEAPI_INVOKE_SIMD

  // unary operators [simd.mask.unary]
  simd_mask operator!() const noexcept;

  // simd_mask binary operators [simd.mask.binary]
  friend simd_mask operator&&(const simd_mask&, const simd_mask&) noexcept;
  friend simd_mask operator||(const simd_mask&, const simd_mask&) noexcept;
  friend simd_mask operator&(const simd_mask&, const simd_mask&)noexcept;
  friend simd_mask operator|(const simd_mask&, const simd_mask&) noexcept;
  friend simd_mask operator^(const simd_mask&, const simd_mask&) noexcept;

  // simd_mask compound assignment [simd.mask.cassign]
  friend simd_mask& operator&=(simd_mask&, const simd_mask&) noexcept;
  friend simd_mask& operator|=(simd_mask&, const simd_mask&) noexcept;
  friend simd_mask& operator^=(simd_mask&, const simd_mask&) noexcept;

  // simd_mask compares [simd.mask.comparison]
  friend simd_mask operator==(const simd_mask&, const simd_mask&) noexcept;
  friend simd_mask operator!=(const simd_mask&, const simd_mask&) noexcept;

#ifdef ENABLE_SYCL_EXT_ONEAPI_INVOKE_SIMD
private:
  __simd_storage<element_type, _Abi> __s_;

  // TODO inefficient, use this's and __v's storage directly
  template <class _Up, class _UAbi>
  inline void copyElements(const simd_mask<_Up, _UAbi> & __v) noexcept {
    for (size_t __i = 0; __i < size(); __i++) {
      (*this)[__i] = static_cast<element_type>(__v[__i]);
    }
  }
#endif // ENABLE_SYCL_EXT_ONEAPI_INVOKE_SIMD
};

#endif // _LIBCPP_STD_VER >= 17

_LIBCPP_END_NAMESPACE_EXPERIMENTAL_SIMD

_LIBCPP_POP_MACROS

// Removed for ENABLE_SYCL_EXT_ONEAPI_INVOKE_SIMD {
//#endif /* _LIBCPP_EXPERIMENTAL_SIMD */
// } Removed for ENABLE_SYCL_EXT_ONEAPI_INVOKE_SIMD
