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

/** @see cv::Error::Code */
enum {
 CV_StsOk=                       0,  /**< everything is ok                */
 CV_StsBackTrace=               -1,  /**< pseudo error for back trace     */
 CV_StsError=                   -2,  /**< unknown /unspecified error      */
 CV_StsInternal=                -3,  /**< internal error (bad state)      */
 CV_StsNoMem=                   -4,  /**< insufficient memory             */
 CV_StsBadArg=                  -5,  /**< function arg/param is bad       */
 CV_StsBadFunc=                 -6,  /**< unsupported function            */
 CV_StsNoConv=                  -7,  /**< iter. didn't converge           */
 CV_StsAutoTrace=               -8,  /**< tracing                         */
 CV_HeaderIsNull=               -9,  /**< image header is NULL            */
 CV_BadImageSize=              -10,  /**< image size is invalid           */
 CV_BadOffset=                 -11,  /**< offset is invalid               */
 CV_BadDataPtr=                -12,  /**/
 CV_BadStep=                   -13,  /**< image step is wrong, this may happen for a non-continuous matrix */
 CV_BadModelOrChSeq=           -14,  /**/
 CV_BadNumChannels=            -15,  /**< bad number of channels, for example, some functions accept only single channel matrices */
 CV_BadNumChannel1U=           -16,  /**/
 CV_BadDepth=                  -17,  /**< input image depth is not supported by the function */
 CV_BadAlphaChannel=           -18,  /**/
 CV_BadOrder=                  -19,  /**< number of dimensions is out of range */
 CV_BadOrigin=                 -20,  /**< incorrect input origin               */
 CV_BadAlign=                  -21,  /**< incorrect input align                */
 CV_BadCallBack=               -22,  /**/
 CV_BadTileSize=               -23,  /**/
 CV_BadCOI=                    -24,  /**< input COI is not supported           */
 CV_BadROISize=                -25,  /**< incorrect input roi                  */
 CV_MaskIsTiled=               -26,  /**/
 CV_StsNullPtr=                -27,  /**< null pointer */
 CV_StsVecLengthErr=           -28,  /**< incorrect vector length */
 CV_StsFilterStructContentErr= -29,  /**< incorrect filter structure content */
 CV_StsKernelStructContentErr= -30,  /**< incorrect transform kernel content */
 CV_StsFilterOffsetErr=        -31,  /**< incorrect filter offset value */
 CV_StsBadSize=                -201, /**< the input/output structure size is incorrect  */
 CV_StsDivByZero=              -202, /**< division by zero */
 CV_StsInplaceNotSupported=    -203, /**< in-place operation is not supported */
 CV_StsObjectNotFound=         -204, /**< request can't be completed */
 CV_StsUnmatchedFormats=       -205, /**< formats of input/output arrays differ */
 CV_StsBadFlag=                -206, /**< flag is wrong or not supported */
 CV_StsBadPoint=               -207, /**< bad CvPoint */
 CV_StsBadMask=                -208, /**< bad format of mask (neither 8uC1 nor 8sC1)*/
 CV_StsUnmatchedSizes=         -209, /**< sizes of input/output structures do not match */
 CV_StsUnsupportedFormat=      -210, /**< the data format/type is not supported by the function*/
 CV_StsOutOfRange=             -211, /**< some of parameters are out of range */
 CV_StsParseError=             -212, /**< invalid syntax/structure of the parsed file */
 CV_StsNotImplemented=         -213, /**< the requested function/feature is not implemented */
 CV_StsBadMemBlock=            -214, /**< an allocated block has been corrupted */
 CV_StsAssert=                 -215, /**< assertion failed   */
 CV_GpuNotSupported=           -216, /**< no CUDA support    */
 CV_GpuApiCallError=           -217, /**< GPU API call error */
 CV_OpenGlNotSupported=        -218, /**< no OpenGL support  */
 CV_OpenGlApiCallError=        -219, /**< OpenGL API call error */
 CV_OpenCLApiCallError=        -220, /**< OpenCL API call error */
 CV_OpenCLDoubleNotSupported=  -221,
 CV_OpenCLInitError=           -222, /**< OpenCL initialization error */
 CV_OpenCLNoAMDBlasFft=        -223
};

/****************************************************************************************\
*                             Common macros and inline functions                         *
\****************************************************************************************/

#define CV_SWAP(a,b,t) ((t) = (a), (a) = (b), (b) = (t))

/** min & max without jumps */
#define  CV_IMIN(a, b)  ((a) ^ (((a)^(b)) & (((a) < (b)) - 1)))

#define  CV_IMAX(a, b)  ((a) ^ (((a)^(b)) & (((a) > (b)) - 1)))

/** absolute value without jumps */
#ifndef __cplusplus
#  define  CV_IABS(a)     (((a) ^ ((a) < 0 ? -1 : 0)) - ((a) < 0 ? -1 : 0))
#else
#  define  CV_IABS(a)     abs(a)
#endif
#define  CV_CMP(a,b)    (((a) > (b)) - ((a) < (b)))
#define  CV_SIGN(a)     CV_CMP((a),0)

#define cvInvSqrt(value) ((float)(1./sqrt(value)))
#define cvSqrt(value)  ((float)sqrt(value))


/*************** Random number generation *******************/

typedef uint64 CvRNG;

#define CV_RNG_COEFF 4164903690U

/** @brief Initializes a random number generator state.

The function initializes a random number generator and returns the state. The pointer to the state
can be then passed to the cvRandInt, cvRandReal and cvRandArr functions. In the current
implementation a multiply-with-carry generator is used.
@param seed 64-bit value used to initiate a random sequence
@sa the C++ class RNG replaced CvRNG.
 */
CV_INLINE CvRNG cvRNG( int64 seed CV_DEFAULT(-1))
{
    CvRNG rng = seed ? (uint64)seed : (uint64)(int64)-1;
    return rng;
}

/** @brief Returns a 32-bit unsigned integer and updates RNG.

The function returns a uniformly-distributed random 32-bit unsigned integer and updates the RNG
state. It is similar to the rand() function from the C runtime library, except that OpenCV functions
always generates a 32-bit random number, regardless of the platform.
@param rng CvRNG state initialized by cvRNG.
 */
CV_INLINE unsigned cvRandInt( CvRNG* rng )
{
    uint64 temp = *rng;
    temp = (uint64)(unsigned)temp*CV_RNG_COEFF + (temp >> 32);
    *rng = temp;
    return (unsigned)temp;
}

/** @brief Returns a floating-point random number and updates RNG.

The function returns a uniformly-distributed random floating-point number between 0 and 1 (1 is not
included).
@param rng RNG state initialized by cvRNG
 */
CV_INLINE double cvRandReal( CvRNG* rng )
{
    return cvRandInt(rng)*2.3283064365386962890625e-10 /* 2^-32 */;
}

/****************************************************************************************\
*                                  Image type (IplImage)                                 *
\****************************************************************************************/

#ifndef HAVE_IPL

/*
 * The following definitions (until #endif)
 * is an extract from IPL headers.
 * Copyright (c) 1995 Intel Corporation.
 */
#define IPL_DEPTH_SIGN 0x80000000

#define IPL_DEPTH_1U     1
#define IPL_DEPTH_8U     8
#define IPL_DEPTH_16U   16
#define IPL_DEPTH_32F   32

#define IPL_DEPTH_8S  (IPL_DEPTH_SIGN| 8)
#define IPL_DEPTH_16S (IPL_DEPTH_SIGN|16)
#define IPL_DEPTH_32S (IPL_DEPTH_SIGN|32)

#define IPL_DATA_ORDER_PIXEL  0
#define IPL_DATA_ORDER_PLANE  1

#define IPL_ORIGIN_TL 0
#define IPL_ORIGIN_BL 1

#define IPL_ALIGN_4BYTES   4
#define IPL_ALIGN_8BYTES   8
#define IPL_ALIGN_16BYTES 16
#define IPL_ALIGN_32BYTES 32

#define IPL_ALIGN_DWORD   IPL_ALIGN_4BYTES
#define IPL_ALIGN_QWORD   IPL_ALIGN_8BYTES

#define IPL_BORDER_CONSTANT   0
#define IPL_BORDER_REPLICATE  1
#define IPL_BORDER_REFLECT    2
#define IPL_BORDER_WRAP       3

/** The IplImage is taken from the Intel Image Processing Library, in which the format is native. OpenCV
only supports a subset of possible IplImage formats, as outlined in the parameter list above.

In addition to the above restrictions, OpenCV handles ROIs differently. OpenCV functions require
that the image size or ROI size of all source and destination images match exactly. On the other
hand, the Intel Image Processing Library processes the area of intersection between the source and
destination images (or ROIs), allowing them to vary independently.
*/
typedef struct
#ifdef __cplusplus
  CV_EXPORTS
#endif
_IplImage
{
    int  nSize;             /**< sizeof(IplImage) */
    int  ID;                /**< version (=0)*/
    int  nChannels;         /**< Most of OpenCV functions support 1,2,3 or 4 channels */
    int  alphaChannel;      /**< Ignored by OpenCV */
    int  depth;             /**< Pixel depth in bits: IPL_DEPTH_8U, IPL_DEPTH_8S, IPL_DEPTH_16S,
                               IPL_DEPTH_32S, IPL_DEPTH_32F and IPL_DEPTH_64F are supported.  */
    char colorModel[4];     /**< Ignored by OpenCV */
    char channelSeq[4];     /**< ditto */
    int  dataOrder;         /**< 0 - interleaved color channels, 1 - separate color channels.
                               cvCreateImage can only create interleaved images */
    int  origin;            /**< 0 - top-left origin,
                               1 - bottom-left origin (Windows bitmaps style).  */
    int  align;             /**< Alignment of image rows (4 or 8).
                               OpenCV ignores it and uses widthStep instead.    */
    int  width;             /**< Image width in pixels.                           */
    int  height;            /**< Image height in pixels.                          */
    struct _IplROI *roi;    /**< Image ROI. If NULL, the whole image is selected. */
    struct _IplImage *maskROI;      /**< Must be NULL. */
    void  *imageId;                 /**< "           " */
    struct _IplTileInfo *tileInfo;  /**< "           " */
    int  imageSize;         /**< Image data size in bytes
                               (==image->height*image->widthStep
                               in case of interleaved data)*/
    char *imageData;        /**< Pointer to aligned image data.         */
    int  widthStep;         /**< Size of aligned image row in bytes.    */
    int  BorderMode[4];     /**< Ignored by OpenCV.                     */
    int  BorderConst[4];    /**< Ditto.                                 */
    char *imageDataOrigin;  /**< Pointer to very origin of image data
                               (not necessarily aligned) -
                               needed for correct deallocation */

#ifdef __cplusplus
    _IplImage() {}
    _IplImage(const cv::Mat& m);
#endif
}
IplImage;

typedef struct _IplTileInfo IplTileInfo;

typedef struct _IplROI
{
    int  coi; /**< 0 - no COI (all channels are selected), 1 - 0th channel is selected ...*/
    int  xOffset;
    int  yOffset;
    int  width;
    int  height;
}
IplROI;

typedef struct _IplConvKernel
{
    int  nCols;
    int  nRows;
    int  anchorX;
    int  anchorY;
    int *values;
    int  nShiftR;
}
IplConvKernel;

typedef struct _IplConvKernelFP
{
    int  nCols;
    int  nRows;
    int  anchorX;
    int  anchorY;
    float *values;
}
IplConvKernelFP;

#define IPL_IMAGE_HEADER 1
#define IPL_IMAGE_DATA   2
#define IPL_IMAGE_ROI    4

#endif/*HAVE_IPL*/

/** extra border mode */
#define IPL_BORDER_REFLECT_101    4
#define IPL_BORDER_TRANSPARENT    5

#define IPL_IMAGE_MAGIC_VAL  ((int)sizeof(IplImage))
#define CV_TYPE_NAME_IMAGE "opencv-image"

#define CV_IS_IMAGE_HDR(img) \
    ((img) != NULL && ((const IplImage*)(img))->nSize == sizeof(IplImage))

#define CV_IS_IMAGE(img) \
    (CV_IS_IMAGE_HDR(img) && ((IplImage*)img)->imageData != NULL)

/** for storing double-precision
   floating point data in IplImage's */
#define IPL_DEPTH_64F  64

/** get reference to pixel at (col,row),
   for multi-channel images (col) should be multiplied by number of channels */
#define CV_IMAGE_ELEM( image, elemtype, row, col )       \
    (((elemtype*)((image)->imageData + (image)->widthStep*(row)))[(col)])

/****************************************************************************************\
*                                  Matrix type (CvMat)                                   *
\****************************************************************************************/

#define CV_AUTO_STEP  0x7fffffff
#define CV_WHOLE_ARR  cvSlice( 0, 0x3fffffff )

#define CV_MAGIC_MASK       0xFFFF0000
#define CV_MAT_MAGIC_VAL    0x42420000
#define CV_TYPE_NAME_MAT    "opencv-matrix"

/** Matrix elements are stored row by row. Element (i, j) (i - 0-based row index, j - 0-based column
index) of a matrix can be retrieved or modified using CV_MAT_ELEM macro:

    uchar pixval = CV_MAT_ELEM(grayimg, uchar, i, j)
    CV_MAT_ELEM(cameraMatrix, float, 0, 2) = image.width*0.5f;

To access multip