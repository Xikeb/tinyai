#pragma once

#include <chrono>
#include <tuple>
#include <ext/pb_ds/assoc_container.hpp>

namespace ann::detail {
    /**
     * Taken from https://codeforces.com/blog/entry/62393?#comment-464874
     */
    constexpr static uint64_t splitmix64(uint64_t x) noexcept {
        // http://xorshift.di.unimi.it/splitmix64.c
        x += 0x9e3779b97f4a7c15;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9;
        x = (x ^ (x >> 27)) * 0x94d049bb133111eb;
        return x ^ (x >> 31);
    }

    template<typename T>
    struct integral_hash {
        std::size_t operator()(T x) const noexcept {
            static const uint64_t FIXED_RANDOM = std::chrono::steady_clock::now().time_since_epoch().count();
            return splitmix64(x + FIXED_RANDOM);
        }
    };

    template<typename T>
    struct integral_pair_hash: integral_hash<T> {
        size_t operator()(std::pair<T,T> x) const noexcept {
            static const uint64_t FIXED_RANDOM = std::chrono::steady_clock::now().time_since_epoch().count();
            return splitmix64(static_cast<uint64_t>(x.first) + FIXED_RANDOM) ^ (splitmix64(static_cast<uint64_t>(x.second) + FIXED_RANDOM) >> 1);
        }
    };

    using ManuallyResizeablePolicy = __gnu_pbds::hash_standard_resize_policy<
            __gnu_pbds::hash_exponential_size_policy<>,
            __gnu_pbds::hash_load_check_resize_trigger<true>,
            true
    >;

    using gnu_default_probe = __gnu_pbds::direct_mask_range_hashing<>;
    using gnu_default_comb_probe = typename __gnu_pbds::detail::default_probe_fn<
            __gnu_pbds::direct_mask_range_hashing<>
    >::type;
    template<typename K>
    using gnu_downsizing_probe = __gnu_pbds::direct_mask_range_hashing<
            std::conditional_t<sizeof(K) < sizeof(std::size_t), K, std::size_t>
    >;
    template<typename K>
    using gnu_downsizing_comb_probe = typename __gnu_pbds::detail::default_probe_fn<
            gnu_downsizing_probe<K>
    >::type;

    /**
     * Open-Address hash map (faster insertion/deletion)
     */
    template<typename K, typename V = __gnu_pbds::null_type>
    using gnu_oa_hash_table = __gnu_pbds::gp_hash_table<
            K, V
            //, std::hash<K>, std::equal_to<K>
            , integral_hash<K>, std::equal_to<K>
            , gnu_downsizing_probe<K>
            , gnu_downsizing_comb_probe<K>
            , ManuallyResizeablePolicy
    >;

    /**
     * Open-Address hash set
     */
    template<typename V>
    using gnu_oa_hash_set = gnu_oa_hash_table<V, __gnu_pbds::null_type>;

    /**
     * CCollision chaining hash map (expect marginally faster read/write??)
     * https://codeforces.com/blog/entry/60737?#comment-446346
     */
    template<typename K, typename V = __gnu_pbds::null_type>
    using gnu_cc_hash_table = __gnu_pbds::cc_hash_table<
            K, V,
            std::hash<K>, std::equal_to<K>,
            gnu_default_probe,
            ManuallyResizeablePolicy
    >;

    template<typename V>
    using gnu_cc_hash_set = gnu_cc_hash_table<V, __gnu_pbds::null_type>;

    template<typename K, typename V = __gnu_pbds::null_type>
    using oa_hash_table = gnu_oa_hash_table<K, V>;
    template<typename K>
    using oa_hash_set = gnu_oa_hash_set<K>;

} // ann::detail

namespace ann {
    template<typename K, typename V = __gnu_pbds::null_type>
    using gnu_hash_table = __gnu_pbds::gp_hash_table<
            K, V
            , std::hash<K>
            , std::equal_to<K>
            , __gnu_pbds::direct_mask_range_hashing<K>
            , typename __gnu_pbds::detail::default_probe_fn<__gnu_pbds::direct_mask_range_hashing<K>>::type
            , __gnu_pbds::hash_standard_resize_policy<
                    __gnu_pbds::hash_exponential_size_policy<>,
                    __gnu_pbds::hash_load_check_resize_trigger<true>,
                    true
            >
    >;

    template<typename V>
    using gnu_hash_set = gnu_hash_table<V, __gnu_pbds::null_type>;
} // ann