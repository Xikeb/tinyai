//
// Created by qard-elliot on 4/5/22.
//

#pragma once

#include <utility>


#define STRONG_TYPEDEF_BINOP_DIRECT(op, opName)                                                 \
template <class StrongTypedef>                                                                  \
struct opName {                                                                                 \
public:                                                                                         \
    friend decltype(auto) operator op(const StrongTypedef& lhs, const StrongTypedef& rhs) {     \
        using type = underlying_type<StrongTypedef>;                                            \
        return static_cast<const type&>(lhs) op static_cast<const type&>(rhs);                  \
    }                                                                                           \
};

#define STRONG_TYPEDEF_BINOP_COMBINE(op, opName)                                                \
template <class StrongTypedef>                                                                  \
struct opName {                                                                                 \
    friend StrongTypedef operator op(const StrongTypedef& lhs, const StrongTypedef& rhs) {      \
        using type = underlying_type<StrongTypedef>;                                            \
        return StrongTypedef(static_cast<const type&>(lhs) op static_cast<const type&>(rhs));   \
    }                                                                                           \
};

#define STRONG_TYPEDEF_BINOP_COMBINE_REASSIGN(op, opName) \
template <class StrongTypedef>                                                                \
struct opName {                                                                               \
    friend StrongTypedef& operator op##=(StrongTypedef& lhs, const StrongTypedef& rhs) {      \
        using type = underlying_type<StrongTypedef>;                                          \
        static_cast<type&>(lhs) op##= static_cast<const type&>(rhs);                          \
        return lhs;                                                                           \
    }                                                                                         \
    friend StrongTypedef operator op(const StrongTypedef& lhs, const StrongTypedef& rhs) {    \
        using type = underlying_type<StrongTypedef>;                                          \
        return StrongTypedef(static_cast<const type&>(lhs) op static_cast<const type&>(rhs)); \
    }                                                                                         \
};


#define STRONG_TYPEDEF_CONST_UNOP_DIRECT(op, opName)               \
template <class StrongTypedef>                                     \
struct opName {                                                    \
public:                                                            \
    friend decltype(auto) operator op(const StrongTypedef& lhs) {  \
        using type = underlying_type<StrongTypedef>;               \
        return op static_cast<const type&>(lhs);                   \
    }                                                              \
};

#define STRONG_TYPEDEF_UNOP_DIRECT(op, opName)                      \
template <class StrongTypedef>                                      \
struct opName {                                                     \
public:                                                             \
    friend decltype(auto) operator op(const StrongTypedef& lhs) {   \
        using type = underlying_type<StrongTypedef>;                \
        return op static_cast<const type&>(lhs);                    \
    }                                                               \
};


namespace types {

    template<class Tag, typename T>
    class strong_typedef_safe {
    public:
        strong_typedef_safe() : value_() {
        }

        explicit strong_typedef_safe(const T &value) : value_(value) {
        }

        explicit strong_typedef_safe(T&& value) noexcept(std::is_nothrow_move_constructible<T>::value)
                : value_(std::move(value)) {
                }

        explicit operator T &()noexcept { return value_; }

        explicit operator const T &() const noexcept { return value_; }

        friend void swap(strong_typedef_safe &a, strong_typedef_safe &b) noexcept {
            using std::swap;
            swap(static_cast<T &>(a), static_cast<T &>(b));
        }

    private:
        T value_;
    };

    namespace impl {
        template<typename Tag, typename T>
        T typedef_underlying_type(strong_typedef_safe<Tag, T>);
    } // impl

    template<typename StrongTypedef>
    using underlying_type = decltype(impl::typedef_underlying_type(std::declval<StrongTypedef>()));



    STRONG_TYPEDEF_BINOP_DIRECT(<=, less_than)
    STRONG_TYPEDEF_BINOP_DIRECT(<, less)
    STRONG_TYPEDEF_BINOP_DIRECT(==, equal_to)
    STRONG_TYPEDEF_BINOP_DIRECT(!=, not_equal_to)
    STRONG_TYPEDEF_BINOP_DIRECT(>=, greater_than)
    STRONG_TYPEDEF_BINOP_DIRECT(>, greater)

    STRONG_TYPEDEF_BINOP_COMBINE_REASSIGN(+, addition)
    STRONG_TYPEDEF_BINOP_COMBINE_REASSIGN(-, substraction)
    STRONG_TYPEDEF_BINOP_COMBINE_REASSIGN(*, multiplication)
    STRONG_TYPEDEF_BINOP_COMBINE_REASSIGN(/, division)
    STRONG_TYPEDEF_BINOP_COMBINE_REASSIGN(%, integer_division)

    STRONG_TYPEDEF_BINOP_COMBINE_REASSIGN(<<, binary_lshift)
    STRONG_TYPEDEF_BINOP_COMBINE_REASSIGN(>>, binary_rshift)

    STRONG_TYPEDEF_BINOP_COMBINE_REASSIGN(&, binary_and)
    STRONG_TYPEDEF_BINOP_COMBINE_REASSIGN(|, binary_or)
    STRONG_TYPEDEF_BINOP_COMBINE_REASSIGN(^, binary_xor)


    STRONG_TYPEDEF_BINOP_COMBINE(+, immut_addition)
    STRONG_TYPEDEF_BINOP_COMBINE(-, immut_substraction)
    STRONG_TYPEDEF_BINOP_COMBINE(*, immut_multiplication)
    STRONG_TYPEDEF_BINOP_COMBINE(/, immut_division)
    STRONG_TYPEDEF_BINOP_COMBINE(%, immut_integer_division)

    STRONG_TYPEDEF_BINOP_COMBINE(<<, immut_binary_lshift)
    STRONG_TYPEDEF_BINOP_COMBINE(>>, immut_binary_rshift)

    STRONG_TYPEDEF_BINOP_COMBINE(&, immut_binary_and)
    STRONG_TYPEDEF_BINOP_COMBINE(|, immut_binary_or)
    STRONG_TYPEDEF_BINOP_COMBINE(^, immut_binary_xor)

    STRONG_TYPEDEF_BINOP_COMBINE(&&, logical_and)
    STRONG_TYPEDEF_BINOP_COMBINE(||, logical_or)

#if __cplusplus >= 202002L
    STRONG_TYPEDEF_BINOP_DIRECT(<=>, spaceship)
#endif

    STRONG_TYPEDEF_CONST_UNOP_DIRECT(+, unary_plus)
    STRONG_TYPEDEF_CONST_UNOP_DIRECT(-, unary_minus)

    STRONG_TYPEDEF_UNOP_DIRECT(++, unary_increment)
    STRONG_TYPEDEF_UNOP_DIRECT(--, unary_decrement)



    template<typename StrongTypedef>
    struct math_arithmetic :
            addition<StrongTypedef>,
            substraction<StrongTypedef>,
            multiplication<StrongTypedef>,
            division<StrongTypedef>,
            integer_division<StrongTypedef> {
    };

    template<typename StrongTypedef>
    struct binary_arithmetic :
            binary_lshift<StrongTypedef>,
            binary_rshift<StrongTypedef> {
    };



    template<typename StrongTypedef>
    struct immut_math_arithmetic :
            addition<StrongTypedef>,
            substraction<StrongTypedef>,
            multiplication<StrongTypedef>,
            division<StrongTypedef>,
            integer_division<StrongTypedef> {
    };

    template<typename StrongTypedef>
    struct immut_binary_arithmetic :
            immut_binary_lshift<StrongTypedef>,
            immut_binary_rshift<StrongTypedef> {
    };

}