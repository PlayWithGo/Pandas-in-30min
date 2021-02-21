
// AUTOGENERATED, DO NOT EDIT

#if !defined CV_DISABLE_OPTIMIZATION && defined CV_ENABLE_INTRINSICS && defined CV_CPU_COMPILE_SSE
#  define CV_TRY_SSE 1
#  define CV_CPU_HAS_SUPPORT_SSE 1
#  define CV_CPU_CALL_SSE(fn, args) return (cpu_baseline::fn args)
#  define CV_CPU_CALL_SSE_(fn, args) return (opt_SSE::fn args)
#elif !defined CV_DISABLE_OPTIMIZATION && defined CV_ENABLE_INTRINSICS && defined CV_CPU_DISPATCH_COMPILE_SSE
#  define CV_TRY_SSE 1
#  define CV_CPU_HAS_SUPPORT_SSE (cv::checkHardwareSupport(CV_CPU_SSE))
#  define CV_CPU_CALL_SSE(fn, args) if (CV_CPU_HAS_SUPPORT_SSE) return (opt_SSE::fn args)
#  define CV_CPU_CALL_SSE_(fn, args) if (CV_CPU_HAS_SUPPORT_SSE) return (opt_SSE::fn args)
#else
#  define CV_TRY_SSE 0
#  define CV_CPU_HAS_SUPPORT_SSE 0
#  define CV_CPU_CALL_SSE(fn, args)
#  define CV_CPU_CALL_SSE_(fn, args)
#endif
#define __CV_CPU_DISPATCH_CHAIN_SSE(fn, args, mode, ...)  CV_CPU_CALL_SSE(fn, args); __CV_EXPAND(__CV_CPU_DISPATCH_CHAIN_ ## mode(fn, args, __VA_ARGS__))

#if !defined CV_DISABLE_OPTIMIZATION && defined CV_ENABLE_INTRINSICS && defined CV_CPU_COMPILE_SSE2
#  define CV_TRY_SSE2 1
#  define CV_CPU_HAS_SUPPORT_SSE2 1
#  define CV_CPU_CALL_SSE2(fn, args) return (cpu_baseline::fn args)
#  define CV_CPU_CALL_SSE2_(fn, args) return (opt_SSE2::fn args)
#elif !defined CV_DISABLE_OPTIMIZATION && defined CV_ENABLE_INTRINSICS && defined CV_CPU_DISPATCH_COMPILE_SSE2
#  define CV_TRY_SSE2 1
#  define CV_CPU_HAS_SUPPORT_SSE2 (cv::checkHardwareSupport(CV_CPU_SSE2))
#  define CV_CPU_CALL_SSE2(fn, args) if (CV_CPU_HAS_SUPPORT_SSE2) return (opt_SSE2::fn args)
#  define CV_CPU_CALL_SSE2_(fn, args) if (CV_CPU_HAS_SUPPORT_SSE2) return (opt_SSE2::fn args)
#else
#  define CV_TRY_SSE2 0
#  define CV_CPU_HAS_SUPPORT_SSE2 0
#  define CV_CPU_CALL_SSE2(fn, args)
#  define CV_CPU_CALL_SSE2_(fn, args)
#endif
#define __CV_CPU_DISPATCH_CHAIN_SSE2(fn, args, mode, ...)  CV_CPU_CALL_SSE2(fn, args); __CV_EXPAND(__CV_CPU_DISPATCH_CHAIN_ ## mode(fn, args, __VA_ARGS__))

#if !defined CV_DISABLE_OPTIMIZATION && defined CV_ENABLE_INTRINSICS && defined CV_CPU_COMPILE_SSE3
#  define CV_TRY_SSE3 1
#  define CV_CPU_HAS_SUPPORT_SSE3 1
#  define CV_CPU_CALL_SSE3(fn, args) return (cpu_baseline::fn args)
#  define CV_CPU_CALL_SSE3_(fn, args) return (opt_SSE3::fn args)
#elif !defined CV_DISABLE_OPTIMIZATION && defined CV_ENABLE_INTRINSICS && defined CV_CPU_DISPATCH_COMPILE_SSE3
#  define CV_TRY_SSE3 1
#  define CV_CPU_HAS_SUPPORT_SSE3 (cv::checkHardwareSupport(CV_CPU_SSE3))
#  define CV_CPU_CALL_SSE3(fn, args) if (CV_CPU_HAS_SUPPORT_SSE3) return (opt_SSE3::fn args)
#  define CV_CPU_CALL_SSE3_(fn, args) if (CV_CPU_HAS_SUPPORT_SSE3) return (opt_SSE3::fn args)
#else
#  define CV_TRY_SSE3 0
#  define CV_CPU_HAS_SUPPORT_SSE3 0
#  define CV_CPU_CALL_SSE3(fn, args)
#  define CV_CPU_CALL_SSE3_(fn, args)
#endif
#define __CV_CPU_DISPATCH_CHAIN_SSE3(fn, args, mode, ...)  CV_CPU_CALL_SSE3(fn, args); __CV_EXPAND(__CV_CPU_DISPATCH_CHAIN_ ## mode(fn, args, __VA_ARGS__))

#if !defined CV_DISABLE_OPTIMIZATION && defined CV_ENABLE_INTRINSICS && defined CV_CPU_COMPILE_SSSE3
#  define CV_TRY_SSSE3 1
#  define CV_CPU_HAS_SUPPORT_SSSE3 1
#  define CV_CPU_CALL_SSSE3(fn, args) return (cpu_baseline::fn args)
#  define CV_CPU_CALL_SSSE3_(fn, args) return (opt_SSSE3::fn args)
#elif !defined CV_DISABLE_OPTIMIZATION && defined CV_ENABLE_INTRINSICS && defined CV_CPU_DISPATCH_COMPILE_SSSE3
#  define CV_TRY_SSSE3 1
#  define CV_CPU_HAS_SUPPORT_SSSE3 (cv::checkHardwareSupport(CV_CPU_SSSE3))
#  define CV_CPU_CALL_SSSE3(fn, args) if (CV_CPU_HAS_SUPPORT_SSSE3) return (opt_SSSE3::fn args)
#  define CV_CPU_CALL_SSSE3_(fn, args) if (CV_CPU_HAS_SUPPORT_SSSE3) return (opt_SSSE3::fn args)
#else
#  define CV_TRY_SSSE3 0
#  define CV_CPU_HAS_SUPPORT_SSSE3 0
#  define CV_CPU_CALL_SSSE3(fn, args)
#  define CV_CPU_CALL_SSSE3_(fn, args)
#endif
#define __CV_CPU_DISPATCH_CHAIN_SSSE3(fn, args, mode, ...)  CV_CPU_CALL_SSSE3(fn, args); __CV_EXPAND(__CV_CPU_DISPATCH_CHAIN_ ## mode(fn, args, __VA_ARGS__))

#if !defined CV_DISABLE_OPTIMIZATION && defined CV_ENABLE_INTRINSICS && defined CV_CPU_COMPILE_SSE4_1
#  define CV_TRY_SSE4_1 1
#  define CV_CPU_HAS_SUPPORT_SSE4_1 1
#  define CV_CPU_CALL_SSE4_1(fn, args) return (cpu_baseline::fn args)
#  define CV_CPU_CALL_SSE4_1_(fn, args) return (opt_SSE4_1::fn args)
#elif !defined CV_DISABLE_OPTIMIZATION && defined CV_ENABLE_INTRINSICS && defined CV_CPU_DISPATCH_COMPILE_SSE4_1
#  define CV_TRY_SSE4_1 1
#  define CV_CPU_HAS_SUPPORT_SSE4_1 (cv::checkHardwareSupport(CV_CPU_SSE4_1))
#  define CV_CPU_CALL_SSE4_1(fn, args) if (CV_CPU_HAS_SUPPORT_SSE4_1) return (opt_SSE4_1::fn args)
#  define CV_CPU_CALL_SSE4_1_(fn, args) if (CV_CPU_HAS_SUPPORT_SSE4_1) return (opt_SSE4_1::fn args)
#else
#  define CV_TRY_SSE4_1 0
#  define CV_CPU_HAS_SUPPORT_SSE4_1 0
#  define CV_CPU_CALL_SSE4_1(fn, args)
#  define CV_CPU_CALL_SSE4_1_(fn, args)
#endif
#define __CV_CPU_DISPATCH_CHAIN_SSE4_1(fn, args, mode, ...)  CV_CPU_CALL_SSE4_1(fn, args); __CV_EXPAND(__CV_CPU_DISPATCH_CHAIN_ ## mode(fn, args, __VA_ARGS__))

#if !defined CV_DISABLE_OPTIMIZATION && defined CV_ENABLE_INTRINSICS && defined CV_CPU_COMPILE_SSE4_2
#  define CV_TRY_SSE4_2 1
#  define CV_CPU_HAS_SUPPORT_SSE4_2 1
#  define CV_CPU_CALL_SSE4_2(fn, args) return (cpu_baseline::fn args)
#  define CV_CPU_CALL_SSE4_2_(fn, args) return (opt_SSE4_2::fn args)
#elif !defined CV_DISABLE_OPTIMIZATION && defined CV_ENABLE_INTRINSICS && defined CV_CPU_DISPATCH_COMPILE_SSE4_2
#  define CV_TRY_SSE4_2 1
#  define CV_CPU_HAS_SUPPORT_SSE4_2 (cv::checkHardwareSupport(CV_CPU_SSE4_2))
#  define CV_CPU_CALL_SSE4_2(fn, args) if (CV_CPU_HAS_SUPPORT_SSE4_2) return (opt_SSE4_2::fn args)
#  define CV_CPU_CALL_SSE4_2_(fn, args) if (CV_CPU_HAS_SUPPORT_SSE4_2) return (opt_SSE4_2::fn args)
#else
#  define CV_TRY_SSE4_2 0
#  define CV_CPU_HAS_SUPPORT_SSE4_2 0
#  define CV_CPU_CALL_SSE4_2(fn, args)
#  define CV_CPU_CALL_SSE4_2_(fn, args)
#endif
#define __CV_CPU_DISPATCH_CHAIN_SSE4_2(fn, args, mode, ...)  CV_CPU_CALL_SSE4_2(fn, args); __CV_EXPAND(__CV_CPU_DISPATCH_CHAIN_ ## mode(fn, args, __VA_ARGS__))

#if !defined CV_DISABLE_OPTIMIZATION && defined CV_ENABLE_INTRINSICS && defined CV_CPU_COMPILE_POPCNT
#  define CV_TRY_POPCNT 1
#  define CV_CPU_HAS_SUPPORT_POPCNT 1
#  define CV_CPU_CALL_POPCNT(fn, args) return (cpu_baseline::fn args)
#  define CV_CPU_CALL_POPCNT_(fn, args) return (opt_POPCNT::fn args)
#elif !defined CV_DISABLE_OPTIMIZATION && defined CV_ENABLE_INTRINSICS && defined CV_CPU_DISPATCH_COMPILE_POPCNT
#  define CV_TRY_POPCNT 1
#  define CV_CPU_HAS_SUPPORT_POPCNT (cv::checkHardwareSupport(CV_CPU_POPCNT))
#  define CV_CPU_CALL_POPCNT(fn, args) if (CV_CPU_HAS_SUPPORT_POPCNT) return (opt_POPCNT::fn args)
#  define CV_CPU_CALL_POPCNT_(fn, args) if (CV_CPU_HAS_SUPPORT_POPCNT) return (opt_POPCNT::fn args)
#else
#  define CV_TRY_POPCNT 0
#  define CV_CPU_HAS_SUPPORT_POPCNT 0
#  define CV_CPU_CALL_POPCNT(fn, args)
#  define CV_CPU_CALL_POPCNT_(fn, args)
#endif
#define __CV_CPU_DISPATCH_CHAIN_POPCNT(fn, args, mode, ...)  CV_CPU_CALL_POPCNT(fn, args); __CV_EXPAND(__CV_CPU_DISPATCH_CHAIN_ ## mode(fn, args, __VA_ARGS__))

#if !defined CV_DISABLE_OPTIMIZATION && defined CV_ENABLE_INTRINSICS && defined CV_CPU_COMPILE_AVX
#  define CV_TRY_AVX 1
#  define CV_CPU_HAS_SUPPORT_AVX 1
#  define CV_CPU_CALL_AVX(fn, args) return (cpu_baseline::fn args)
#  define CV_CPU_CALL_AVX_(fn, args) return (opt_AVX::fn args)
#elif !defined CV_DISABLE_OPTIMIZATION && defined CV_ENABLE_INTRINSICS && defined CV_CPU_DISPATCH_COMPILE_AVX
#  define CV_TRY_AVX 1
#  define CV_CPU_HAS_SUPPORT_AVX (cv::checkHardwareSupport(CV_CPU_AVX))
#  define CV_CPU_CALL_AVX(fn, args) if (CV_CPU_HAS_SUPPORT_AVX) return (opt_AVX::fn args)
#  define CV_CPU_CALL_AVX_(fn, args) if (CV_CPU_HAS_SUPPORT_AVX) return (opt_AVX::fn args)
#else
#  define CV_TRY_AVX 0
#  define CV_CPU_HAS_SUPPORT_AVX 0
#  define CV_CPU_CALL_AVX(fn, args)
#  define CV_CPU_CALL_AVX_(fn, args)
#endif
#define __CV_CPU_DISPATCH_CHAIN_AVX(fn, args, mode, ...)  CV_CPU_CALL_AVX(fn, args); __CV_EXPAND(__CV_CPU_DISPATCH_CHAIN_ ## mode(fn, args, __VA_ARGS__))

#if !defined CV_DISABLE_OPTIMIZATION && defined CV_ENABLE_INTRINSICS && defined CV_CPU_COMPILE_FP16
#  define CV_TRY_FP16 1
#  define CV_CPU_HAS_SUPPORT_FP16 1
#  define CV_CPU_CALL_FP16(fn, args) return (cpu_baseline::fn args)
#  define CV_CPU_CALL_FP16_(fn, args) return (opt_FP16::fn args)
#elif !defined CV_DISABLE_OPTIMIZATION && defined CV_ENABLE_INTRINSICS && defined CV_CPU_DISPATCH_COMPILE_FP16
#  define CV_TRY_FP16 1
#  define CV_CPU_HAS_SUPPORT_FP16 (cv::checkHardwareSupport(CV_CPU_FP16))
#  define CV_CPU_CALL_FP16(fn, args) if (CV_CPU_HAS_SUPPORT_FP16) return (opt_FP16::fn args)
#  define CV_CPU_CALL_FP16_(fn, args) if (CV_CPU_HAS_SUPPORT_FP16) return (opt_FP16::fn args)
#else
#  define CV_TRY_FP16 0
#  define CV_CPU_HAS_SUPPORT_FP16 0
#  define CV_CPU_CALL_FP16(fn, args)
#  define CV_CPU_CALL_FP16_(fn, args)
#endif
#define __CV_CPU_DISPATCH_CHAIN_FP16(fn, args, mode, ...)  CV_CPU_CALL_FP16(fn, args); __CV_EXPAND(__CV_CPU_DISPATCH_CHAIN_ ## mode(fn, args, __VA_ARGS__))

#if !defined CV_DISABLE_OPTIMIZATION && defined CV_ENABLE_INTRINSICS && defined CV_CPU_COMPILE_AVX2
#  define CV_TRY_AVX2 1
#  define CV_CPU_HAS_SUPPORT_AVX2 1
#  define CV_CPU_CALL_AVX2(fn, args) return (cpu_baseline::fn args)
#  define CV_CPU_CALL_AVX2_(fn, args) return (opt_AVX2::fn args)
#elif !defined CV_DISABLE_OPTIMIZATION && defined CV_ENABLE_INTRINSICS && defined CV_CPU_DISPATCH_COMPILE_AVX2
#  define CV_TRY_AVX2 1
#  define CV_CPU_HAS_SUPPORT_AVX2 (cv::checkHardwareSupport(CV_CPU_AVX2))
#  define CV_CPU_CALL_AVX2(fn, args) if (CV_CPU_HAS_SUPPORT_AVX2) return (opt_AVX2::fn args)
#  define CV_CPU_CALL_AVX2_(fn, args) if (CV_CPU_HAS_SUPPORT_AVX2) return (opt_AVX2::fn args)
#else
#  define CV_TRY_AVX2 0
#  define CV_CPU_HAS_SUPPORT_AVX2 0
#  define CV_CPU_CALL_AVX2(fn, args)
#  define CV_CPU_CALL_AVX2_(fn, args)
#endif
#define __CV_CPU_DISPATCH_CHAIN_AVX2(fn, args, mode, ...)  CV_CPU_CALL_AVX2(fn, args); __CV_EXPAND(__CV_CPU_DISPATCH_CHAIN_ ## mode(fn, args, __VA_ARGS__))

#if !defined CV_DISABLE_OPTIMIZATION && defined CV_ENABLE_INTRINSICS && defined CV_CPU_COMPILE_FMA3
#  define CV_TRY_FMA3 1
#  define CV_CPU_HAS_SUPPORT_FMA3 1
#  define CV_CPU_CALL_FMA3(fn, args) return (cpu_baseline::fn args)
#  define CV_CPU_CALL_FMA3_(fn, args) return (opt_FMA3::fn args)
#elif !defined CV_DISABLE_OPTIMIZATION && defined CV_ENABLE_INTRINSICS && defined CV_CPU_DISPATCH_COMPILE_FMA3
#  define CV_TRY_FMA3 1
#  define CV_CPU_HAS_SUPPORT_FMA3 (cv::checkHardwareSupport(CV_CPU_FMA3))
#  define CV_CPU_CALL_FMA3(fn, args) if (CV_CPU_HAS_SUPPORT_FMA3) return (opt_FMA3::fn args)
#  define CV_CPU_CALL_FMA3_(fn, args) if (CV_CPU_HAS_SUPPORT_FMA3) return (opt_FMA3::fn args)
#else
#  define CV_TRY_FMA3 0
#  define CV_CPU_HAS_SUPPORT_FMA3 0
#  define CV_CPU_CALL_FMA3(fn, args)
#  define CV_CPU_CALL_FMA3_(fn, args)
#endif
#define __CV_CPU_DISPATCH_CHAIN_FMA3(fn, args, mode, ...)  CV_CPU_CALL_FMA3(fn, args); __CV_EXPAND(__CV_CPU_DISPATCH_CHAIN_ ## mode(fn, args, __VA_ARGS__))

#if !defined CV_DISABLE_OPTIMIZATION && defined CV_ENABLE_INTRINSICS && defined CV_CPU_COMPILE_AVX_512F
#  define CV_TRY_AVX_512F 1
#  define CV_CPU_HAS_SUPPORT_AVX_512F 1
#  define CV_CPU_CALL_AVX_512F(fn, args) return (cpu_baseline::fn args)
#  define CV_CPU_CALL_AVX_512F_(fn, args) return (opt_AVX_512F::fn args)
#elif !defined CV_DISABLE_OPTIMIZATION && defined CV_ENABLE_INTRINSICS && defined CV_CPU_DISPATCH_COMPILE_AVX_512F
#  define CV_TRY_AVX_512F 1
#  define CV_CPU_HAS_SUPPORT_AVX_512F (cv::checkHardwareSupport(CV_CPU_AVX_512F))
#  define CV_CPU_CALL_AVX_512F(fn, args) if (CV_CPU_HAS_SUPPORT_AVX_512F) return (opt_AVX_512F::fn args)
#  define CV_CPU_CALL_AVX_512F_(fn, args) if (CV_CPU_HAS_SUPPORT_AVX_512F) return (opt_AVX_512F::fn args)
#else
#  define CV_TRY_AVX_512F 0
#  define CV_CPU_HAS_SUPPORT_AVX_512F 0
#  define CV_CPU_CALL_AVX_512F(fn, args)
#  define CV_CPU_CALL_AVX_512F_(fn, args)
#endif
#define __CV_CPU_DISPATCH_CHAIN_AVX_512F(fn, args, mode, ...)  CV_CPU_CALL_AVX_512F(fn, args); __CV_EXPAND(__CV_CPU_DISPATCH_CHAIN_ ## mode(fn, args, __VA_ARGS__))

#if !defined CV_DISABLE_OPTIMIZATION && defined CV_ENABLE_INTRINSICS && defined CV_CPU_COMPILE_AVX512_SKX
#  define CV_TRY_AVX512_SKX 1
#  define CV_CPU_HAS_SUPPORT_AVX512_SKX 1
#  define CV_CPU_CALL_AVX512_SKX(fn, args) return (cpu_baseline::fn args)
#  define CV_CPU_CALL_AVX512_SKX_(fn, args) return (opt_AVX512_SKX::fn args)
#elif !defined CV_DISABLE_OPTIMIZATION && defined CV_ENABLE_INTRINSICS && defined CV_CPU_DISPATCH_COMPILE_AVX512_SKX
#  define CV_TRY_AVX512_SKX 1
#  define CV_CPU_HAS_SUPPORT_AVX512_SKX (cv::checkHardwareSupport(CV_CPU_AVX512_SKX))
#  define CV_CPU_CALL_AVX512_SKX(fn, args) if (CV_CPU_HAS_SUPPORT_AVX512_SKX) return (opt_AVX512_SKX::fn args)
#  define CV_CPU_CALL_AVX512_SKX_(fn, args) if (CV_CPU_HAS_SUPPORT_AVX512_SKX) return (opt_AVX512_SKX::fn args)
#else
#  define CV_TRY_AVX512_SKX 0
#  define CV_CPU_HAS_SUPPORT_AVX512_SKX 0
#  define CV_CPU_CALL_AVX512_SKX(fn, args)
#  define CV_CPU_CALL_AVX512_SKX_(fn, args)
#endif
#define __CV_CPU_DISPATCH_CHAIN_AVX512_SKX(fn, args, mode, ...)  CV_CPU_CALL_AVX512_SKX(fn, args); __CV_EXPAND(__CV_CPU_DISPATCH_CHAIN_ ## mode(fn, args, __VA_ARGS__))

#if !defined CV_DISABLE_OPTIMIZATION && defined CV_ENABLE_INTRINSICS && defined CV_CPU_COMPILE_NEON
#  define CV_TRY_NEON 1
#  define CV_CPU_HAS_SUPPORT_NEON 1
#  define CV_CPU_CALL_NEON(fn, args) return (cpu_baseline::fn args)
#  define CV_CPU_CALL_NEON_(fn, args) return (opt_NEON::fn args)
#elif !defined CV_DISABLE_OPTIMIZATION && defined CV_ENABLE_INTRINSICS && defined CV_CPU_DISPATCH_COMPILE_NEON
#  define CV_TRY_NEON 1
#  define CV_CPU_HAS_SUPPORT_NEON (cv::checkHardwareSupport(CV_CPU_NEON))
#  define CV_CPU_CALL_NEON(fn, args) if (CV_CPU_HAS_SUPPORT_NEON) return (opt_NEON::fn args)
#  define CV_CPU_CALL_NEON_(fn, args) if (CV_CPU_HAS_SUPPORT_NEON) return (opt_NEON::fn args)
#else
#  define CV_TRY_NEON 0
#  define CV_CPU_HAS_SUPPORT_NEON 0
#  define CV_CPU_CALL_NEON(fn, args)
#  define CV_CPU_CALL_NEON_(fn, args)
#endif
#define __CV_CPU_DISPATCH_CHAIN_NEON(fn, args, mode, ...)  CV_CPU_CALL_NEON(fn, args); __CV_EXPAND(__CV_CPU_DISPATCH_CHAIN_ ## mode(fn, args, __VA_ARGS__))

#if !defined CV_DISABLE_OPTIMIZATION && defined CV_ENABLE_INTRINSICS && defined CV_CPU_COMPILE_VSX
#  define CV_TRY_VSX 1
#  define CV_CPU_HAS_SUPPORT_VSX 1
#  define CV_CPU_CALL_VSX(fn, args) return (cpu_baseline::fn args)
#  define CV_CPU_CALL_VSX_(fn, args) return (opt_VSX::fn args)
#elif !defined CV_DISABLE_OPTIMIZATION && defined CV_ENABLE_INTRINSICS && defined CV_CPU_DISPATCH_COMPILE_VSX
#  define CV_TRY_VSX 1
#  define CV_CPU_HAS_SUPPORT_VSX (cv::checkHardwareSupport(CV_CPU_VSX))
#  define CV_CPU_CALL_VSX(fn, args) if (CV_CPU_HAS_SUPPORT_VSX) return (opt_VSX::fn args)
#  define CV_CPU_CALL_VSX_(fn, args) if (CV_CPU_HAS_SUPPORT_VSX) return (opt_VSX::fn args)
#else
#  define CV_TRY_VSX 0
#  define CV_CPU_HAS_SUPPORT_VSX 0
#  define CV_CPU_CALL_VSX(fn, args)
#  define CV_CPU_CALL_VSX_(fn, args)
#endif
#define __CV_CPU_DISPATCH_CHAIN_VSX(fn, args, mode, ...)  CV_CPU_CALL_VSX(fn, args); __CV_EXPAND(__CV_CPU_DISPATCH_CHAIN_ ## mode(fn, args, __VA_ARGS__))

#define CV_CPU_CALL_BASELINE(fn, args) return (cpu_baseline::fn args)
#define __CV_CPU_DISPATCH_CHAIN_BASELINE(fn, args, mode, ...)  CV_CPU_CALL_BASELINE(fn, args) /* last in sequence */