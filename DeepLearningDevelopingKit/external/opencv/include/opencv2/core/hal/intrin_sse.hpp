
/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Copyright (C) 2015, Itseez Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef OPENCV_HAL_SSE_HPP
#define OPENCV_HAL_SSE_HPP

#include <algorithm>
#include "opencv2/core/utility.hpp"

#define CV_SIMD128 1
#define CV_SIMD128_64F 1

namespace cv
{

//! @cond IGNORED

CV_CPU_OPTIMIZATION_HAL_NAMESPACE_BEGIN

struct v_uint8x16
{
    typedef uchar lane_type;
    enum { nlanes = 16 };

    v_uint8x16() : val(_mm_setzero_si128()) {}
    explicit v_uint8x16(__m128i v) : val(v) {}
    v_uint8x16(uchar v0, uchar v1, uchar v2, uchar v3, uchar v4, uchar v5, uchar v6, uchar v7,
               uchar v8, uchar v9, uchar v10, uchar v11, uchar v12, uchar v13, uchar v14, uchar v15)
    {
        val = _mm_setr_epi8((char)v0, (char)v1, (char)v2, (char)v3,
                            (char)v4, (char)v5, (char)v6, (char)v7,
                            (char)v8, (char)v9, (char)v10, (char)v11,
                            (char)v12, (char)v13, (char)v14, (char)v15);
    }
    uchar get0() const
    {
        return (uchar)_mm_cvtsi128_si32(val);
    }

    __m128i val;
};

struct v_int8x16
{
    typedef schar lane_type;
    enum { nlanes = 16 };

    v_int8x16() : val(_mm_setzero_si128()) {}
    explicit v_int8x16(__m128i v) : val(v) {}
    v_int8x16(schar v0, schar v1, schar v2, schar v3, schar v4, schar v5, schar v6, schar v7,
              schar v8, schar v9, schar v10, schar v11, schar v12, schar v13, schar v14, schar v15)
    {
        val = _mm_setr_epi8((char)v0, (char)v1, (char)v2, (char)v3,
                            (char)v4, (char)v5, (char)v6, (char)v7,
                            (char)v8, (char)v9, (char)v10, (char)v11,
                            (char)v12, (char)v13, (char)v14, (char)v15);
    }
    schar get0() const
    {
        return (schar)_mm_cvtsi128_si32(val);
    }

    __m128i val;
};

struct v_uint16x8
{
    typedef ushort lane_type;
    enum { nlanes = 8 };

    v_uint16x8() : val(_mm_setzero_si128()) {}
    explicit v_uint16x8(__m128i v) : val(v) {}
    v_uint16x8(ushort v0, ushort v1, ushort v2, ushort v3, ushort v4, ushort v5, ushort v6, ushort v7)
    {
        val = _mm_setr_epi16((short)v0, (short)v1, (short)v2, (short)v3,
                             (short)v4, (short)v5, (short)v6, (short)v7);
    }
    ushort get0() const
    {
        return (ushort)_mm_cvtsi128_si32(val);
    }

    __m128i val;
};

struct v_int16x8
{
    typedef short lane_type;
    enum { nlanes = 8 };

    v_int16x8() : val(_mm_setzero_si128()) {}
    explicit v_int16x8(__m128i v) : val(v) {}
    v_int16x8(short v0, short v1, short v2, short v3, short v4, short v5, short v6, short v7)
    {
        val = _mm_setr_epi16((short)v0, (short)v1, (short)v2, (short)v3,
                             (short)v4, (short)v5, (short)v6, (short)v7);
    }
    short get0() const
    {
        return (short)_mm_cvtsi128_si32(val);
    }
    __m128i val;
};

struct v_uint32x4
{
    typedef unsigned lane_type;
    enum { nlanes = 4 };

    v_uint32x4() : val(_mm_setzero_si128()) {}
    explicit v_uint32x4(__m128i v) : val(v) {}
    v_uint32x4(unsigned v0, unsigned v1, unsigned v2, unsigned v3)
    {
        val = _mm_setr_epi32((int)v0, (int)v1, (int)v2, (int)v3);
    }
    unsigned get0() const
    {
        return (unsigned)_mm_cvtsi128_si32(val);
    }
    __m128i val;
};

struct v_int32x4
{
    typedef int lane_type;
    enum { nlanes = 4 };

    v_int32x4() : val(_mm_setzero_si128()) {}
    explicit v_int32x4(__m128i v) : val(v) {}
    v_int32x4(int v0, int v1, int v2, int v3)
    {
        val = _mm_setr_epi32(v0, v1, v2, v3);
    }
    int get0() const
    {
        return _mm_cvtsi128_si32(val);
    }
    __m128i val;
};

struct v_float32x4
{
    typedef float lane_type;
    enum { nlanes = 4 };

    v_float32x4() : val(_mm_setzero_ps()) {}
    explicit v_float32x4(__m128 v) : val(v) {}
    v_float32x4(float v0, float v1, float v2, float v3)
    {
        val = _mm_setr_ps(v0, v1, v2, v3);
    }
    float get0() const
    {
        return _mm_cvtss_f32(val);
    }
    __m128 val;
};

struct v_uint64x2
{
    typedef uint64 lane_type;
    enum { nlanes = 2 };

    v_uint64x2() : val(_mm_setzero_si128()) {}
    explicit v_uint64x2(__m128i v) : val(v) {}
    v_uint64x2(uint64 v0, uint64 v1)
    {
        val = _mm_setr_epi32((int)v0, (int)(v0 >> 32), (int)v1, (int)(v1 >> 32));
    }
    uint64 get0() const
    {
        int a = _mm_cvtsi128_si32(val);
        int b = _mm_cvtsi128_si32(_mm_srli_epi64(val, 32));
        return (unsigned)a | ((uint64)(unsigned)b << 32);
    }
    __m128i val;
};

struct v_int64x2
{
    typedef int64 lane_type;
    enum { nlanes = 2 };

    v_int64x2() : val(_mm_setzero_si128()) {}
    explicit v_int64x2(__m128i v) : val(v) {}
    v_int64x2(int64 v0, int64 v1)
    {
        val = _mm_setr_epi32((int)v0, (int)(v0 >> 32), (int)v1, (int)(v1 >> 32));
    }
    int64 get0() const
    {
        int a = _mm_cvtsi128_si32(val);
        int b = _mm_cvtsi128_si32(_mm_srli_epi64(val, 32));
        return (int64)((unsigned)a | ((uint64)(unsigned)b << 32));
    }
    __m128i val;
};

struct v_float64x2
{
    typedef double lane_type;
    enum { nlanes = 2 };

    v_float64x2() : val(_mm_setzero_pd()) {}
    explicit v_float64x2(__m128d v) : val(v) {}
    v_float64x2(double v0, double v1)
    {
        val = _mm_setr_pd(v0, v1);
    }
    double get0() const
    {
        return _mm_cvtsd_f64(val);
    }
    __m128d val;
};

#if CV_FP16
struct v_float16x4
{
    typedef short lane_type;
    enum { nlanes = 4 };

    v_float16x4() : val(_mm_setzero_si128()) {}
    explicit v_float16x4(__m128i v) : val(v) {}
    v_float16x4(short v0, short v1, short v2, short v3)
    {
        val = _mm_setr_epi16(v0, v1, v2, v3, 0, 0, 0, 0);
    }
    short get0() const
    {
        return (short)_mm_cvtsi128_si32(val);
    }
    __m128i val;
};
#endif

#define OPENCV_HAL_IMPL_SSE_INITVEC(_Tpvec, _Tp, suffix, zsuffix, ssuffix, _Tps, cast) \
inline _Tpvec v_setzero_##suffix() { return _Tpvec(_mm_setzero_##zsuffix()); } \
inline _Tpvec v_setall_##suffix(_Tp v) { return _Tpvec(_mm_set1_##ssuffix((_Tps)v)); } \
template<typename _Tpvec0> inline _Tpvec v_reinterpret_as_##suffix(const _Tpvec0& a) \
{ return _Tpvec(cast(a.val)); }

OPENCV_HAL_IMPL_SSE_INITVEC(v_uint8x16, uchar, u8, si128, epi8, char, OPENCV_HAL_NOP)
OPENCV_HAL_IMPL_SSE_INITVEC(v_int8x16, schar, s8, si128, epi8, char, OPENCV_HAL_NOP)
OPENCV_HAL_IMPL_SSE_INITVEC(v_uint16x8, ushort, u16, si128, epi16, short, OPENCV_HAL_NOP)
OPENCV_HAL_IMPL_SSE_INITVEC(v_int16x8, short, s16, si128, epi16, short, OPENCV_HAL_NOP)
OPENCV_HAL_IMPL_SSE_INITVEC(v_uint32x4, unsigned, u32, si128, epi32, int, OPENCV_HAL_NOP)
OPENCV_HAL_IMPL_SSE_INITVEC(v_int32x4, int, s32, si128, epi32, int, OPENCV_HAL_NOP)
OPENCV_HAL_IMPL_SSE_INITVEC(v_float32x4, float, f32, ps, ps, float, _mm_castsi128_ps)
OPENCV_HAL_IMPL_SSE_INITVEC(v_float64x2, double, f64, pd, pd, double, _mm_castsi128_pd)

inline v_uint64x2 v_setzero_u64() { return v_uint64x2(_mm_setzero_si128()); }
inline v_int64x2 v_setzero_s64() { return v_int64x2(_mm_setzero_si128()); }
inline v_uint64x2 v_setall_u64(uint64 val) { return v_uint64x2(val, val); }
inline v_int64x2 v_setall_s64(int64 val) { return v_int64x2(val, val); }

template<typename _Tpvec> inline
v_uint64x2 v_reinterpret_as_u64(const _Tpvec& a) { return v_uint64x2(a.val); }
template<typename _Tpvec> inline
v_int64x2 v_reinterpret_as_s64(const _Tpvec& a) { return v_int64x2(a.val); }
inline v_float32x4 v_reinterpret_as_f32(const v_uint64x2& a)
{ return v_float32x4(_mm_castsi128_ps(a.val)); }
inline v_float32x4 v_reinterpret_as_f32(const v_int64x2& a)
{ return v_float32x4(_mm_castsi128_ps(a.val)); }
inline v_float64x2 v_reinterpret_as_f64(const v_uint64x2& a)
{ return v_float64x2(_mm_castsi128_pd(a.val)); }
inline v_float64x2 v_reinterpret_as_f64(const v_int64x2& a)
{ return v_float64x2(_mm_castsi128_pd(a.val)); }

#define OPENCV_HAL_IMPL_SSE_INIT_FROM_FLT(_Tpvec, suffix) \
inline _Tpvec v_reinterpret_as_##suffix(const v_float32x4& a) \
{ return _Tpvec(_mm_castps_si128(a.val)); } \
inline _Tpvec v_reinterpret_as_##suffix(const v_float64x2& a) \
{ return _Tpvec(_mm_castpd_si128(a.val)); }

OPENCV_HAL_IMPL_SSE_INIT_FROM_FLT(v_uint8x16, u8)
OPENCV_HAL_IMPL_SSE_INIT_FROM_FLT(v_int8x16, s8)
OPENCV_HAL_IMPL_SSE_INIT_FROM_FLT(v_uint16x8, u16)
OPENCV_HAL_IMPL_SSE_INIT_FROM_FLT(v_int16x8, s16)
OPENCV_HAL_IMPL_SSE_INIT_FROM_FLT(v_uint32x4, u32)
OPENCV_HAL_IMPL_SSE_INIT_FROM_FLT(v_int32x4, s32)
OPENCV_HAL_IMPL_SSE_INIT_FROM_FLT(v_uint64x2, u64)
OPENCV_HAL_IMPL_SSE_INIT_FROM_FLT(v_int64x2, s64)

inline v_float32x4 v_reinterpret_as_f32(const v_float32x4& a) {return a; }
inline v_float64x2 v_reinterpret_as_f64(const v_float64x2& a) {return a; }
inline v_float32x4 v_reinterpret_as_f32(const v_float64x2& a) {return v_float32x4(_mm_castpd_ps(a.val)); }
inline v_float64x2 v_reinterpret_as_f64(const v_float32x4& a) {return v_float64x2(_mm_castps_pd(a.val)); }

//////////////// PACK ///////////////
inline v_uint8x16 v_pack(const v_uint16x8& a, const v_uint16x8& b)
{
    __m128i delta = _mm_set1_epi16(255);
    return v_uint8x16(_mm_packus_epi16(_mm_subs_epu16(a.val, _mm_subs_epu16(a.val, delta)),
                                       _mm_subs_epu16(b.val, _mm_subs_epu16(b.val, delta))));
}

inline void v_pack_store(uchar* ptr, const v_uint16x8& a)
{
    __m128i delta = _mm_set1_epi16(255);
    __m128i a1 = _mm_subs_epu16(a.val, _mm_subs_epu16(a.val, delta));
    _mm_storel_epi64((__m128i*)ptr, _mm_packus_epi16(a1, a1));
}

inline v_uint8x16 v_pack_u(const v_int16x8& a, const v_int16x8& b)
{ return v_uint8x16(_mm_packus_epi16(a.val, b.val)); }

inline void v_pack_u_store(uchar* ptr, const v_int16x8& a)
{ _mm_storel_epi64((__m128i*)ptr, _mm_packus_epi16(a.val, a.val)); }

template<int n> inline
v_uint8x16 v_rshr_pack(const v_uint16x8& a, const v_uint16x8& b)
{
    // we assume that n > 0, and so the shifted 16-bit values can be treated as signed numbers.
    __m128i delta = _mm_set1_epi16((short)(1 << (n-1)));
    return v_uint8x16(_mm_packus_epi16(_mm_srli_epi16(_mm_adds_epu16(a.val, delta), n),
                                       _mm_srli_epi16(_mm_adds_epu16(b.val, delta), n)));
}

template<int n> inline
void v_rshr_pack_store(uchar* ptr, const v_uint16x8& a)
{
    __m128i delta = _mm_set1_epi16((short)(1 << (n-1)));
    __m128i a1 = _mm_srli_epi16(_mm_adds_epu16(a.val, delta), n);
    _mm_storel_epi64((__m128i*)ptr, _mm_packus_epi16(a1, a1));
}

template<int n> inline
v_uint8x16 v_rshr_pack_u(const v_int16x8& a, const v_int16x8& b)
{
    __m128i delta = _mm_set1_epi16((short)(1 << (n-1)));
    return v_uint8x16(_mm_packus_epi16(_mm_srai_epi16(_mm_adds_epi16(a.val, delta), n),
                                       _mm_srai_epi16(_mm_adds_epi16(b.val, delta), n)));
}

template<int n> inline
void v_rshr_pack_u_store(uchar* ptr, const v_int16x8& a)
{
    __m128i delta = _mm_set1_epi16((short)(1 << (n-1)));
    __m128i a1 = _mm_srai_epi16(_mm_adds_epi16(a.val, delta), n);
    _mm_storel_epi64((__m128i*)ptr, _mm_packus_epi16(a1, a1));
}

inline v_int8x16 v_pack(const v_int16x8& a, const v_int16x8& b)
{ return v_int8x16(_mm_packs_epi16(a.val, b.val)); }

inline void v_pack_store(schar* ptr, v_int16x8& a)
{ _mm_storel_epi64((__m128i*)ptr, _mm_packs_epi16(a.val, a.val)); }

template<int n> inline
v_int8x16 v_rshr_pack(const v_int16x8& a, const v_int16x8& b)
{
    // we assume that n > 0, and so the shifted 16-bit values can be treated as signed numbers.
    __m128i delta = _mm_set1_epi16((short)(1 << (n-1)));
    return v_int8x16(_mm_packs_epi16(_mm_srai_epi16(_mm_adds_epi16(a.val, delta), n),
                                     _mm_srai_epi16(_mm_adds_epi16(b.val, delta), n)));
}
template<int n> inline
void v_rshr_pack_store(schar* ptr, const v_int16x8& a)
{
    // we assume that n > 0, and so the shifted 16-bit values can be treated as signed numbers.
    __m128i delta = _mm_set1_epi16((short)(1 << (n-1)));
    __m128i a1 = _mm_srai_epi16(_mm_adds_epi16(a.val, delta), n);
    _mm_storel_epi64((__m128i*)ptr, _mm_packs_epi16(a1, a1));
}


// bit-wise "mask ? a : b"
inline __m128i v_select_si128(__m128i mask, __m128i a, __m128i b)
{
    return _mm_xor_si128(b, _mm_and_si128(_mm_xor_si128(a, b), mask));
}

inline v_uint16x8 v_pack(const v_uint32x4& a, const v_uint32x4& b)
{
    __m128i z = _mm_setzero_si128(), maxval32 = _mm_set1_epi32(65535), delta32 = _mm_set1_epi32(32768);
    __m128i a1 = _mm_sub_epi32(v_select_si128(_mm_cmpgt_epi32(z, a.val), maxval32, a.val), delta32);
    __m128i b1 = _mm_sub_epi32(v_select_si128(_mm_cmpgt_epi32(z, b.val), maxval32, b.val), delta32);
    __m128i r = _mm_packs_epi32(a1, b1);
    return v_uint16x8(_mm_sub_epi16(r, _mm_set1_epi16(-32768)));
}

inline void v_pack_store(ushort* ptr, const v_uint32x4& a)
{
    __m128i z = _mm_setzero_si128(), maxval32 = _mm_set1_epi32(65535), delta32 = _mm_set1_epi32(32768);
    __m128i a1 = _mm_sub_epi32(v_select_si128(_mm_cmpgt_epi32(z, a.val), maxval32, a.val), delta32);
    __m128i r = _mm_packs_epi32(a1, a1);
    _mm_storel_epi64((__m128i*)ptr, _mm_sub_epi16(r, _mm_set1_epi16(-32768)));
}

template<int n> inline
v_uint16x8 v_rshr_pack(const v_uint32x4& a, const v_uint32x4& b)
{
    __m128i delta = _mm_set1_epi32(1 << (n-1)), delta32 = _mm_set1_epi32(32768);
    __m128i a1 = _mm_sub_epi32(_mm_srli_epi32(_mm_add_epi32(a.val, delta), n), delta32);
    __m128i b1 = _mm_sub_epi32(_mm_srli_epi32(_mm_add_epi32(b.val, delta), n), delta32);
    return v_uint16x8(_mm_sub_epi16(_mm_packs_epi32(a1, b1), _mm_set1_epi16(-32768)));
}

template<int n> inline
void v_rshr_pack_store(ushort* ptr, const v_uint32x4& a)
{
    __m128i delta = _mm_set1_epi32(1 << (n-1)), delta32 = _mm_set1_epi32(32768);
    __m128i a1 = _mm_sub_epi32(_mm_srli_epi32(_mm_add_epi32(a.val, delta), n), delta32);
    __m128i a2 = _mm_sub_epi16(_mm_packs_epi32(a1, a1), _mm_set1_epi16(-32768));
    _mm_storel_epi64((__m128i*)ptr, a2);
}

inline v_uint16x8 v_pack_u(const v_int32x4& a, const v_int32x4& b)
{
    __m128i delta32 = _mm_set1_epi32(32768);
    __m128i r = _mm_packs_epi32(_mm_sub_epi32(a.val, delta32), _mm_sub_epi32(b.val, delta32));
    return v_uint16x8(_mm_sub_epi16(r, _mm_set1_epi16(-32768)));
}

inline void v_pack_u_store(ushort* ptr, const v_int32x4& a)
{
    __m128i delta32 = _mm_set1_epi32(32768);
    __m128i a1 = _mm_sub_epi32(a.val, delta32);
    __m128i r = _mm_sub_epi16(_mm_packs_epi32(a1, a1), _mm_set1_epi16(-32768));
    _mm_storel_epi64((__m128i*)ptr, r);
}

template<int n> inline
v_uint16x8 v_rshr_pack_u(const v_int32x4& a, const v_int32x4& b)
{
    __m128i delta = _mm_set1_epi32(1 << (n-1)), delta32 = _mm_set1_epi32(32768);
    __m128i a1 = _mm_sub_epi32(_mm_srai_epi32(_mm_add_epi32(a.val, delta), n), delta32);
    __m128i a2 = _mm_sub_epi16(_mm_packs_epi32(a1, a1), _mm_set1_epi16(-32768));
    __m128i b1 = _mm_sub_epi32(_mm_srai_epi32(_mm_add_epi32(b.val, delta), n), delta32);
    __m128i b2 = _mm_sub_epi16(_mm_packs_epi32(b1, b1), _mm_set1_epi16(-32768));
    return v_uint16x8(_mm_unpacklo_epi64(a2, b2));
}

template<int n> inline
void v_rshr_pack_u_store(ushort* ptr, const v_int32x4& a)
{
    __m128i delta = _mm_set1_epi32(1 << (n-1)), delta32 = _mm_set1_epi32(32768);
    __m128i a1 = _mm_sub_epi32(_mm_srai_epi32(_mm_add_epi32(a.val, delta), n), delta32);
    __m128i a2 = _mm_sub_epi16(_mm_packs_epi32(a1, a1), _mm_set1_epi16(-32768));
    _mm_storel_epi64((__m128i*)ptr, a2);
}

inline v_int16x8 v_pack(const v_int32x4& a, const v_int32x4& b)
{ return v_int16x8(_mm_packs_epi32(a.val, b.val)); }

inline void v_pack_store(short* ptr, const v_int32x4& a)
{
    _mm_storel_epi64((__m128i*)ptr, _mm_packs_epi32(a.val, a.val));
}

template<int n> inline
v_int16x8 v_rshr_pack(const v_int32x4& a, const v_int32x4& b)
{
    __m128i delta = _mm_set1_epi32(1 << (n-1));
    return v_int16x8(_mm_packs_epi32(_mm_srai_epi32(_mm_add_epi32(a.val, delta), n),
                                     _mm_srai_epi32(_mm_add_epi32(b.val, delta), n)));
}

template<int n> inline
void v_rshr_pack_store(short* ptr, const v_int32x4& a)
{
    __m128i delta = _mm_set1_epi32(1 << (n-1));
    __m128i a1 = _mm_srai_epi32(_mm_add_epi32(a.val, delta), n);