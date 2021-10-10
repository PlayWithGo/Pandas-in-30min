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

#ifndef OPENCV_CORE_TYPES_H
#define OPENCV_CORE_TYPES_H

#ifdef HAVE_IPL
#  ifndef __IPL_H__
#    if defined _WIN32
#      include <ipl.h>
#    else
#      include <ipl/ipl.h>
#    endif
#  endif
#elif defined __IPL_H__
#  define HAVE_IPL
#endif

#include "opencv2/core/cvdef.h"

#ifndef SKIP_INCLUDES
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#endif // SKIP_INCLUDES

#if defined _WIN32
#  define CV_CDECL __cdecl
#  define CV_STDCALL __stdcall
#else
#  define CV_CDECL
#  define CV_STDCALL
#endif

#ifndef CV_DEFAULT
#  ifdef __cplusplus
#    define CV_DEFAULT(val) = val
#  else
#    define CV_DEFAULT(val)
#  endif
#endif

#ifndef CV_EXTERN_C_FUNCPTR
#  ifdef __cplusplus
#    define CV_EXTERN_C_FUNCPTR(x) extern "C" { typedef x; }
#  else
#    define CV_EXTERN_C_FUNCPTR(x) typedef x
#  endif
#endif

#ifndef CVAPI
#  define CVAPI(rettype) CV_EXTERN_C CV_EXPORTS rettype CV_CDECL
#endif

#ifndef CV_IMPL
#  define CV_IMPL CV_EXTERN_C
#endif

#ifdef __cplusplus
#  include "opencv2/core.hpp"
#endif

/** @addtogroup core_c
    @{
*/

/** @brief This is the "metatype" used *only* as a function parameter.

It denotes that the function accepts arrays of multiple types, such as IplImage*, CvMat* or even
CvSeq* sometimes. The particular array type is determined at runtime by analyzing the first 4
bytes of the header. In C++ interface the role of CvArr is played by InputArray and OutputArray.
 */
typedef void CvArr;

typedef int CVStatus;

/** @s