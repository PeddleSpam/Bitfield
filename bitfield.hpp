/**
 * \file Bitfield.h
 * \brief `Bitfield` class template.
 *
 * This file defines the `Bitfield` class template and its supporting utilities.
 */
#ifndef _BITFIELD_HPP
#define _BITFIELD_HPP

#include <cstdint>
#include <limits>
#include <type_traits>

// `Bitfield` Forward Declaration ----------------------------------------------

template<std::size_t, bool>
class Bitfield;

// Integer Type Selection Metafunctions ----------------------------------------

/**
 * Defines a fixed-width unsigned integer member type called `type`. This is
 * the smallest type whose width is no less than `num_bits`.
 *
 * \param num_bits The minimum number of bits that `type` must contain. This
 *                 value cannot be zero.
 */
template<std::size_t num_bits>
struct least_uint {
private:

  template<typename T>
  struct type_wrapper {using type = T;};

  template<std::size_t, typename...>
  struct find_least;

  template<typename... Args>
  struct find_least<0u, Args...> {};

  template<std::size_t N, typename T, typename... Args>
  struct find_least<N, T, Args...>: 
    public std::conditional_t<N <= std::numeric_limits<T>::digits,
      type_wrapper<T>, find_least<N, Args...>> {};

  template<std::size_t N, typename... Args>
  using find_least_t = typename find_least<N, Args...>::type;

public:

  using type = find_least_t<num_bits,
    std::uint8_t, std::uint16_t, std::uint32_t, std::uint64_t>;
};

template<std::size_t num_bits>
using least_uint_t = typename least_uint<num_bits>::type;

/**
 * Defines a fixed-width signed integer member type called `type`. This is the
 * smallest type whose width is no less than `num_bits`.
 *
 * \param num_bits The minimum number of bits that `type` must contain. This
 *                 value cannot be zero.
 */
template<std::size_t num_bits>
using least_int = std::make_signed<least_uint_t<num_bits>>;

template<std::size_t num_bits>
using least_int_t = typename least_int<num_bits>::type;

// Standard Metafunction Aliases -----------------------------------------------

#if __cplusplus < 201703L //####################################################

#define BITFIELD_NODISCARD __attribute__((warn_unused_result))

template<typename...>
using void_t = void;

template<typename T>
static constexpr auto is_integral_v = std::is_integral<T>::value;

#else //########################################################################

#define BITFIELD_NODISCARD [[nodiscard]]

template<typename... Args>
using void_t = std::void_t<Args...>;

template<typename T>
static constexpr auto is_integral_v = std::is_integral_v<T>;

#endif //#######################################################################

// Common Type Detection Metafunctions -----------------------------------------

/**
 * Member constant `value` is equal to `true` if a common type exists for all
 * types in `Args`. Otherwise `value` is equal to `false`.
 */
template<typename... Args>
struct common_type_exists {
private:

  template<typename, typename...>
  struct helper: std::false_type {};

  template<typename... Ts>
  struct helper<void_t<std::common_type_t<Ts...>>, Ts...>: std::true_type {};

public:

  static constexpr auto value = helper<void, Args...>::value;
};

template<typename... Args>
static constexpr auto common_type_exists_v =
  common_type_exists<Args...>::value;

/**
 * Defines a member type alias of `T` called `type` if common type exists for
 * all types in `Args`. Otherwise no member type exists.
 */
template<typename T = void, typename... Args>
using enable_if_common_type =
  std::enable_if<common_type_exists_v<Args...>, T>;

template<typename T = void, typename... Args>
using enable_if_common_type_t =
  typename enable_if_common_type<T, Args...>::type;

// `Bitfield` Detection Metafunction -------------------------------------------

/**
 * Member constant `value` is equal to `true` if `T` is a `Bitfield` type.
 * Otherwise `value` is equal to `false`.
 */
template<typename T>
struct is_bitfield {
private:

  template<typename U>
  struct helper: std::false_type {};

  template<std::size_t N, bool S>
  struct helper<Bitfield<N, S>>: std::true_type {};

public:

  static constexpr auto value = helper<std::decay_t<T>>::value;
};

template<typename T>
static constexpr auto is_bitfield_v = is_bitfield<T>::value;

// `Bitfield` Make Function ----------------------------------------------------

/**
 * Constructs a `Bitfield` with the same sign and number of bits as type `T`.
 * `T` must be an integral type (either signed or unsigned), and cannot be a 
 * `Bitfield` type.
 *
 * \param val A a value of type `T`. 
 *             
 * \return A `Bitfield` equal to `val`.
 */
template<typename T, typename = 
  std::enable_if_t<is_integral_v<T> && !is_bitfield_v<T>>>
BITFIELD_NODISCARD constexpr auto make_bitfield(T const & val) {

  using limits_t = std::numeric_limits<T>;
  constexpr auto num_bits = limits_t::digits + limits_t::is_signed;
  return Bitfield<num_bits, limits_t::is_signed>(val);
}

// `Bitfield` Class Template ---------------------------------------------------

/**
 * An arbitrary width integer type.
 *
 * \param num_bits_v Size of the contained integer type in bits. Must be
 *                   greater than or equal to 1 when `is_signed_v` is `false`,
 *                   and greater than or equal to 2 when `is_signed_v` is
 *                   `true`.
 *                   The value of `num_bits_v` is accessible via the
 *                   public member constant `num_bits`.
 * \param is_signed_v Indicates the signedness of the contained integer type.
 *                    If `true` the type is signed. If `false` the type is
 *                    unsigned.
 *                    The value of `is_signed_v` is accessible via the public
 *                    member constant `is_signed`.
 */
template<std::size_t num_bits_v, bool is_signed_v = true>
class BITFIELD_NODISCARD Bitfield {
public:

  static_assert((is_signed_v && num_bits_v > 1u) || (num_bits_v > 0u), "");
  static_assert(num_bits_v <= std::numeric_limits<std::uintmax_t>::digits, "");

  static constexpr auto num_bits = num_bits_v;
  static constexpr auto is_signed = is_signed_v;

  using int_type = std::conditional_t<
    is_signed, _ExtInt(num_bits), unsigned _ExtInt(num_bits)>;

  using uint_type = unsigned _ExtInt(num_bits);

  using fixed_width_type = std::conditional_t<
    is_signed, least_int_t<num_bits>, least_uint_t<num_bits>>;

  template<std::size_t, bool>
  friend class Bitfield;

  // Constructors --------------------------------------------------------------

  constexpr Bitfield() = default;
  constexpr Bitfield(Bitfield &&) = default;
  constexpr Bitfield(Bitfield const &) = default;

  constexpr Bitfield(int_type const & val): value(val) {}

  template<std::size_t N, bool S>
  constexpr Bitfield(Bitfield<N, S> const & other): Bitfield(other.value) {}

  // Accessors -----------------------------------------------------------------

  constexpr int_type const & get() const {return value;}

  // Assignment Operators ------------------------------------------------------

  constexpr Bitfield & operator=(Bitfield &&) = default;
  constexpr Bitfield & operator=(Bitfield const &) = default;

  constexpr Bitfield & operator=(int_type const & val) {
    value = val;
    return *this;
  }

  template<std::size_t N, bool S>
  constexpr Bitfield & operator=(Bitfield<N, S> const & other) {
    value = other.value;
    return *this;
  }

  // Compound Assignment Operators ---------------------------------------------

  template<typename T, typename = enable_if_common_type_t<Bitfield, T>>
  constexpr Bitfield & operator+=(T const & rhs) {
    *this = *this + rhs;
    return *this;
  }

  template<typename T, typename = enable_if_common_type_t<Bitfield, T>>
  constexpr Bitfield && operator+=(T && rhs) && {
    *this = *this + rhs;
    return *this;
  }

  template<typename T, typename = enable_if_common_type_t<Bitfield, T>>
  constexpr Bitfield & operator-=(T const & rhs) {
    *this = *this - rhs;
    return *this;
  }

  template<typename T, typename = enable_if_common_type_t<Bitfield, T>>
  constexpr Bitfield && operator-=(T && rhs) && {
    *this = *this - rhs;
    return *this;
  }

  template<typename T, typename = enable_if_common_type_t<Bitfield, T>>
  constexpr Bitfield & operator*=(T const & rhs) {
    *this = *this * rhs;
    return *this;
  }

  template<typename T, typename = enable_if_common_type_t<Bitfield, T>>
  constexpr Bitfield && operator*=(T && rhs) && {
    *this = *this * rhs;
    return *this;
  }

  template<typename T, typename = enable_if_common_type_t<Bitfield, T>>
  constexpr Bitfield & operator/=(T const & rhs) {
    *this = *this / rhs;
    return *this;
  }

  template<typename T, typename = enable_if_common_type_t<Bitfield, T>>
  constexpr Bitfield && operator/=(T && rhs) && {
    *this = *this / rhs;
    return *this;
  }

  template<typename T, typename = enable_if_common_type_t<Bitfield, T>>
  constexpr Bitfield & operator%=(T const & rhs) {
    *this = *this % rhs;
    return *this;
  }

  template<typename T, typename = enable_if_common_type_t<Bitfield, T>>
  constexpr Bitfield && operator%=(T && rhs) && {
    *this = *this % rhs;
    return *this;
  }

  template<typename T, typename = enable_if_common_type_t<Bitfield, T>>
  constexpr Bitfield & operator&=(T const & rhs) {
    *this = *this & rhs;
    return *this;
  }

  template<typename T, typename = enable_if_common_type_t<Bitfield, T>>
  constexpr Bitfield && operator&=(T && rhs) && {
    *this = *this & rhs;
    return *this;
  }

  template<typename T, typename = enable_if_common_type_t<Bitfield, T>>
  constexpr Bitfield & operator|=(T const & rhs) {
    *this = *this | rhs;
    return *this;
  }

  template<typename T, typename = enable_if_common_type_t<Bitfield, T>>
  constexpr Bitfield && operator|=(T && rhs) && {
    *this = *this | rhs;
    return *this;
  }

  template<typename T, typename = enable_if_common_type_t<Bitfield, T>>
  constexpr Bitfield & operator^=(T const & rhs) {
    *this = *this ^ rhs;
    return *this;
  }

  template<typename T, typename = enable_if_common_type_t<Bitfield, T>>
  constexpr Bitfield && operator^=(T && rhs) && {
    *this = *this ^ rhs;
    return *this;
  }

  constexpr Bitfield & operator<<=(std::size_t const & count) {
    value <<= count;
    return *this;
  }

  constexpr Bitfield && operator<<=(std::size_t && count) && {
    value <<= count;
    return *this;
  }

  constexpr Bitfield & operator>>=(std::size_t const & count) {
    value >>= count;
    return *this;
  }

  constexpr Bitfield && operator>>=(std::size_t && count) && {
    value >>= count;
    return *this;
  }

  // Unary Operators -----------------------------------------------------------

  constexpr Bitfield operator+() const {return +value;}
  constexpr Bitfield operator-() const {return -value;}
  constexpr Bitfield operator~() const {return ~value;}

  constexpr Bitfield const operator++(int) {
    return value++;
  }

  constexpr Bitfield const operator--(int) {
    return value--;
  }

  constexpr Bitfield & operator++() {
    ++value;
    return *this;
  }

  constexpr Bitfield & operator--() {
    --value;
    return *this;
  }

  // Non-Converting Binary Operators -------------------------------------------

  constexpr Bitfield operator<<(std::size_t count) const {
    return value << count;
  }

  constexpr Bitfield operator>>(std::size_t count) const {
    return value >> count;
  }

  // Type Conversion Operators -------------------------------------------------

  explicit constexpr operator int_type() const {return value;}

  constexpr operator fixed_width_type() const {return value;}

private:

  int_type value;
};

// Comparison Operators --------------------------------------------------------

template<typename T1, typename T2, typename = std::enable_if_t<
  (is_bitfield_v<T1> || is_bitfield_v<T2>) && common_type_exists_v<T1, T2>>>
BITFIELD_NODISCARD constexpr bool operator>(T1 const & lhs, T2 const & rhs) {

  using common_t = std::common_type_t<T1, T2>;
  return common_t(lhs) > common_t(rhs);
}

template<typename T1, typename T2, typename = std::enable_if_t<
  (is_bitfield_v<T1> || is_bitfield_v<T2>) && (common_type_exists_v<T1, T2>)>>
BITFIELD_NODISCARD constexpr bool operator<(T1 const & lhs, T2 const & rhs) {

  using common_t = std::common_type_t<T1, T2>;
  return common_t(lhs) < common_t(rhs);
}

template<typename T1, typename T2, typename = std::enable_if_t<
  (is_bitfield_v<T1> || is_bitfield_v<T2>) && common_type_exists_v<T1, T2>>>
BITFIELD_NODISCARD constexpr bool operator==(T1 const & lhs, T2 const & rhs) {

  using common_t = std::common_type_t<T1, T2>;
  return common_t(lhs) == common_t(rhs);
}

template<typename T1, typename T2, typename = std::enable_if_t<
  (is_bitfield_v<T1> || is_bitfield_v<T2>) && common_type_exists_v<T1, T2>>>
BITFIELD_NODISCARD constexpr bool operator!=(T1 const & lhs, T2 const & rhs) {

  using common_t = std::common_type_t<T1, T2>;
  return common_t(lhs) != common_t(rhs);
}

template<typename T1, typename T2, typename = std::enable_if_t<
  (is_bitfield_v<T1> || is_bitfield_v<T2>) && common_type_exists_v<T1, T2>>>
BITFIELD_NODISCARD constexpr bool operator>=(T1 const & lhs, T2 const & rhs) {

  using common_t = std::common_type_t<T1, T2>;
  return common_t(lhs) >= common_t(rhs);
}

template<typename T1, typename T2, typename = std::enable_if_t<
  (is_bitfield_v<T1> || is_bitfield_v<T2>) && common_type_exists_v<T1, T2>>>
BITFIELD_NODISCARD constexpr bool operator<=(T1 const & lhs, T2 const & rhs) {

  using common_t = std::common_type_t<T1, T2>;
  return common_t(lhs) <= common_t(rhs);
}

// Converting Binary Operators -------------------------------------------------

template<typename T1, typename T2, typename = std::enable_if_t<
  (is_bitfield_v<T1> || is_bitfield_v<T2>) && common_type_exists_v<T1, T2>>>
BITFIELD_NODISCARD constexpr auto operator+(T1 const & lhs, T2 const & rhs) {

  using common_t = std::common_type_t<T1, T2>;
  return make_bitfield(common_t(lhs) + common_t(rhs));
}

template<typename T1, typename T2, typename = std::enable_if_t<
  (is_bitfield_v<T1> || is_bitfield_v<T2>) && common_type_exists_v<T1, T2>>>
BITFIELD_NODISCARD constexpr auto operator-(T1 const & lhs, T2 const & rhs) {

  using common_t = std::common_type_t<T1, T2>;
  return make_bitfield(common_t(lhs) - common_t(rhs));
}

template<typename T1, typename T2, typename = std::enable_if_t<
  (is_bitfield_v<T1> || is_bitfield_v<T2>) && common_type_exists_v<T1, T2>>>
BITFIELD_NODISCARD constexpr auto operator*(T1 const & lhs, T2 const & rhs) {

  using common_t = std::common_type_t<T1, T2>;
  return make_bitfield(common_t(lhs) * common_t(rhs));
}

template<typename T1, typename T2, typename = std::enable_if_t<
  (is_bitfield_v<T1> || is_bitfield_v<T2>) && common_type_exists_v<T1, T2>>>
BITFIELD_NODISCARD constexpr auto operator/(T1 const & lhs, T2 const & rhs) {

  using common_t = std::common_type_t<T1, T2>;
  return make_bitfield(common_t(lhs) / common_t(rhs));
}

template<typename T1, typename T2, typename = std::enable_if_t<
  (is_bitfield_v<T1> || is_bitfield_v<T2>) && common_type_exists_v<T1, T2>>>
BITFIELD_NODISCARD constexpr auto operator%(T1 const & lhs, T2 const & rhs) {

  using common_t = std::common_type_t<T1, T2>;
  return make_bitfield(common_t(lhs) % common_t(rhs));
}

template<typename T1, typename T2, typename = std::enable_if_t<
  (is_bitfield_v<T1> || is_bitfield_v<T2>) && common_type_exists_v<T1, T2>>>
BITFIELD_NODISCARD constexpr auto operator&(T1 const & lhs, T2 const & rhs) {

  using common_t = std::common_type_t<T1, T2>;
  return make_bitfield(common_t(lhs) & common_t(rhs));
}

template<typename T1, typename T2, typename = std::enable_if_t<
  (is_bitfield_v<T1> || is_bitfield_v<T2>) && common_type_exists_v<T1, T2>>>
BITFIELD_NODISCARD constexpr auto operator|(T1 const & lhs, T2 const & rhs) {

  using common_t = std::common_type_t<T1, T2>;
  return make_bitfield(common_t(lhs) | common_t(rhs));
}

template<typename T1, typename T2, typename = std::enable_if_t<
  (is_bitfield_v<T1> || is_bitfield_v<T2>) && common_type_exists_v<T1, T2>>>
BITFIELD_NODISCARD constexpr auto operator^(T1 const & lhs, T2 const & rhs) {

  using common_t = std::common_type_t<T1, T2>;
  return make_bitfield(common_t(lhs) ^ common_t(rhs));
}

// `Bitfield` Standard Library Specialisations ---------------------------------

namespace std {

  // Common Type ---------------------------------------------------------------

  template<size_t N, bool S, typename T>
  struct common_type<Bitfield<N, S>, T> {
    using type = common_type_t<
      typename Bitfield<N, S>::fixed_width_type, std::decay_t<T>>;
  };

  template<typename T, size_t N, bool S>
  struct common_type<T, Bitfield<N, S>> {
    using type = common_type_t<
        std::decay_t<T>, typename Bitfield<N, S>::fixed_width_type>;
  };

  template<size_t N1, bool S1, size_t N2, bool S2>
  struct common_type<Bitfield<N1, S1>, Bitfield<N2, S2>> {
    using type = common_type_t<
      typename Bitfield<N1, S1>::fixed_width_type,
      typename Bitfield<N2, S2>::fixed_width_type>;
  };

  // Numeric Limits ------------------------------------------------------------

  template<size_t N, bool S>
  class numeric_limits<Bitfield<N, S>> {
  private:

    using type = Bitfield<N, S>;
    using int_type = typename type::int_type;
    using uint_type = typename type::uint_type;

    static constexpr size_t num_bits = type::num_bits;

    template<int_type value_v>
    struct calc_digits10 {
      static constexpr int value =
        calc_digits10<value_v / int_type(10)>::value + 1u;
    };

    template<>
    struct calc_digits10<0> {
      static constexpr int value = 0u;
    };

  public:

    static constexpr type max() noexcept {
      if (type::is_signed) {
        constexpr auto shift_count = uint_type(num_bits - 1u);
        return int_type((uint_type(1u) << shift_count) - uint_type(1u));
      }
      return ~int_type(0u);
    }

    static constexpr type min() noexcept {
      return ~max();
    }

    static constexpr type lowest() noexcept {
      return min();
    }

    static constexpr bool is_specialized = true;
    static constexpr bool is_signed = type::is_signed;
    static constexpr bool is_integer = true;
    static constexpr bool is_exact = true;
    static constexpr bool has_infinity = false;
    static constexpr bool has_quiet_NaN = false;
    static constexpr bool has_signaling_NaN = false;
    static constexpr float_denorm_style has_denorm = denorm_absent;
    static constexpr bool has_denorm_loss = false;
    static constexpr float_round_style round_style = round_toward_zero;
    static constexpr bool is_iec559 = false;
    static constexpr bool is_bounded = true;
    static constexpr bool is_modulo = !type::is_signed;
    static constexpr int digits = num_bits;
    static constexpr int digits10 = calc_digits10<max()>::value - 1;
    static constexpr int max_digits10 = 0;
    static constexpr int radix = 2;
    static constexpr int min_exponent = 0;
    static constexpr int min_exponent10 = 0;
    static constexpr int max_exponent = 0;
    static constexpr int max_exponent10 = 0;
    static constexpr bool traps = true;
    static constexpr bool tinyness_before = false;

    static constexpr type epsilon() noexcept {return 0;}
    static constexpr type round_error() noexcept {return 0;}
    static constexpr type infinity() noexcept {return 0;}
    static constexpr type quiet_NaN() noexcept {return 0;}
    static constexpr type signaling_NaN() noexcept {return 0;}
    static constexpr type denorm_min() noexcept {return 0;}
  };
}

// -----------------------------------------------------------------------------

#undef BITFIELD_NODISCARD

#endif // _BITFIELD_HPP
