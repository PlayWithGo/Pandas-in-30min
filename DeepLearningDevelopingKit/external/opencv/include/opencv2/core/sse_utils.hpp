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

#ifndef OPENCV_CORE_SSE_UTILS_HPP
#define OPENCV_CORE_SSE_UTILS_HPP

#ifndef __cplusplus
#  error sse_utils.hpp header must be compiled as C++
#endif

#include "opencv2/core/cvdef.h"

//! @addtogroup core_utils_sse
//! @{

#if CV_SSE2

inline void _mm_deinterleave_epi8(__m128i & v_r0, __m128i & v_r1, __m128i & v_g0, __m128i & v_g1)
{
    __m128i layer1_chunk0 = _mm_unpacklo_epi8(v_r0, v_g0);
    __m128i layer1_chunk1 = _mm_unpackhi_epi8(v_r0, v_g0);
    __m128i layer1_chunk2 = _mm_unpacklo_epi8(v_r1, v_g1);
    __m128i layer1_chunk3 = _mm_unpackhi_epi8(v_r1, v_g1);

    __m128i layer2_chunk0 = _mm_unpacklo_epi8(layer1_chunk0, layer1_chunk2);
    __m128i layer2_chunk1 = _mm_unpackhi_epi8(layer1_chunk0, layer1_chunk2);
    __m128i layer2_chunk2 = _mm_unpacklo_epi8(layer1_chunk1, layer1_chunk3);
    __m128i layer2_chunk3 = _mm_unpackhi_epi8(layer1_chunk1, layer1_chunk3);

    __m128i layer3_chunk0 = _mm_unpacklo_epi8(layer2_chunk0, layer2_chunk2);
    __m128i layer3_chunk1 = _mm_unpackhi_epi8(layer2_chunk0, layer2_chunk2);
    __m128i layer3_chunk2 = _mm_unpacklo_epi8(layer2_chunk1, layer2_chunk3);
    __m128i layer3_chunk3 = _mm_unpackhi_epi8(layer2_chunk1, layer2_chunk3);

    __m128i layer4_chunk0 = _mm_unpacklo_epi8(layer3_chunk0, layer3_chunk2);
    __m128i layer4_chunk1 = _mm_unpackhi_epi8(layer3_chunk0, layer3_chunk2);
    __m128i layer4_chunk2 = _mm_unpacklo_epi8(layer3_chunk1, layer3_chunk3);
    __m128i layer4_chunk3 = _mm_unpackhi_epi8(layer3_chunk1, layer3_chunk3);

    v_r0 = _mm_unpacklo_epi8(layer4_chunk0, layer4_chunk2);
    v_r1 = _mm_unpackhi_epi8(layer4_chunk0, layer4_chunk2);
    v_g0 = _mm_unpacklo_epi8(layer4_chunk1, layer4_chunk3);
    v_g1 = _mm_unpackhi_epi8(layer4_chunk1, layer4_chunk3);
}

inline void _mm_deinterleave_epi8(__m128i & v_r0, __m128i & v_r1, __m128i & v_g0,
                                  __m128i & v_g1, __m128i & v_b0, __m128i & v_b1)
{
    __m128i layer1_chunk0 = _mm_unpacklo_epi8(v_r0, v_g1);
    __m128i layer1_chunk1 = _mm_unpackhi_epi8(v_r0, v_g1);
    __m128i layer1_chunk2 = _mm_unpacklo_epi8(v_r1, v_b0);
    __m128i layer1_chunk3 = _mm_unpackhi_epi8(v_r1, v_b0);
    __m128i layer1_chunk4 = _mm_unpacklo_epi8(v_g0, v_b1);
    __m128i layer1_chunk5 = _mm_unpackhi_epi8(v_g0, v_b1);

    __m128i layer2_chunk0 = _mm_unpacklo_epi8(layer1_chunk0, layer1_chunk3);
    __m128i layer2_chunk1 = _mm_unpackhi_epi8(layer1_chunk0, layer1_chunk3);
    __m128i layer2_chunk2 = _m