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

To access multiple-channel matrices, you can use
CV_MAT_ELEM(matrix, type, i, j\*nchannels + channel_idx).

@deprecated CvMat is now obsolete; consider using Mat instead.
 */
typedef struct CvMat
{
    int type;
    int step;

    /* for internal use only */
    int* refcount;
    int hdr_refcount;

    union
    {
        uchar* ptr;
        short* s;
        int* i;
        float* fl;
        double* db;
    } data;

#ifdef __cplusplus
    union
    {
        int rows;
        int height;
    };

    union
    {
        int cols;
        int width;
    };
#else
    int rows;
    int cols;
#endif


#ifdef __cplusplus
    CvMat() {}
    CvMat(const CvMat& m) { memcpy(this, &m, sizeof(CvMat));}
    CvMat(const cv::Mat& m);
#endif

}
CvMat;


#define CV_IS_MAT_HDR(mat) \
    ((mat) != NULL && \
    (((const CvMat*)(mat))->type & CV_MAGIC_MASK) == CV_MAT_MAGIC_VAL && \
    ((const CvMat*)(mat))->cols > 0 && ((const CvMat*)(mat))->rows > 0)

#define CV_IS_MAT_HDR_Z(mat) \
    ((mat) != NULL && \
    (((const CvMat*)(mat))->type & CV_MAGIC_MASK) == CV_MAT_MAGIC_VAL && \
    ((const CvMat*)(mat))->cols >= 0 && ((const CvMat*)(mat))->rows >= 0)

#define CV_IS_MAT(mat) \
    (CV_IS_MAT_HDR(mat) && ((const CvMat*)(mat))->data.ptr != NULL)

#define CV_IS_MASK_ARR(mat) \
    (((mat)->type & (CV_MAT_TYPE_MASK & ~CV_8SC1)) == 0)

#define CV_ARE_TYPES_EQ(mat1, mat2) \
    ((((mat1)->type ^ (mat2)->type) & CV_MAT_TYPE_MASK) == 0)

#define CV_ARE_CNS_EQ(mat1, mat2) \
    ((((mat1)->type ^ (mat2)->type) & CV_MAT_CN_MASK) == 0)

#define CV_ARE_DEPTHS_EQ(mat1, mat2) \
    ((((mat1)->type ^ (mat2)->type) & CV_MAT_DEPTH_MASK) == 0)

#define CV_ARE_SIZES_EQ(mat1, mat2) \
    ((mat1)->rows == (mat2)->rows && (mat1)->cols == (mat2)->cols)

#define CV_IS_MAT_CONST(mat)  \
    (((mat)->rows|(mat)->cols) == 1)

#define IPL2CV_DEPTH(depth) \
    ((((CV_8U)+(CV_16U<<4)+(CV_32F<<8)+(CV_64F<<16)+(CV_8S<<20)+ \
    (CV_16S<<24)+(CV_32S<<28)) >> ((((depth) & 0xF0) >> 2) + \
    (((depth) & IPL_DEPTH_SIGN) ? 20 : 0))) & 15)

/** Inline constructor. No data is allocated internally!!!
 * (Use together with cvCreateData, or use cvCreateMat instead to
 * get a matrix with allocated data):
 */
CV_INLINE CvMat cvMat( int rows, int cols, int type, void* data CV_DEFAULT(NULL))
{
    CvMat m;

    assert( (unsigned)CV_MAT_DEPTH(type) <= CV_64F );
    type = CV_MAT_TYPE(type);
    m.type = CV_MAT_MAGIC_VAL | CV_MAT_CONT_FLAG | type;
    m.cols = cols;
    m.rows = rows;
    m.step = m.cols*CV_ELEM_SIZE(type);
    m.data.ptr = (uchar*)data;
    m.refcount = NULL;
    m.hdr_refcount = 0;

    return m;
}

#ifdef __cplusplus
inline CvMat::CvMat(const cv::Mat& m)
{
    CV_DbgAssert(m.dims <= 2);
    *this = cvMat(m.rows, m.dims == 1 ? 1 : m.cols, m.type(), m.data);
    step = (int)m.step[0];
    type = (type & ~cv::Mat::CONTINUOUS_FLAG) | (m.flags & cv::Mat::CONTINUOUS_FLAG);
}
#endif


#define CV_MAT_ELEM_PTR_FAST( mat, row, col, pix_size )  \
    (assert( (unsigned)(row) < (unsigned)(mat).rows &&   \
             (unsigned)(col) < (unsigned)(mat).cols ),   \
     (mat).data.ptr + (size_t)(mat).step*(row) + (pix_size)*(col))

#define CV_MAT_ELEM_PTR( mat, row, col )                 \
    CV_MAT_ELEM_PTR_FAST( mat, row, col, CV_ELEM_SIZE((mat).type) )

#define CV_MAT_ELEM( mat, elemtype, row, col )           \
    (*(elemtype*)CV_MAT_ELEM_PTR_FAST( mat, row, col, sizeof(elemtype)))

/** @brief Returns the particular element of single-channel floating-point matrix.

The function is a fast replacement for cvGetReal2D in the case of single-channel floating-point
matrices. It is faster because it is inline, it does fewer checks for array type and array element
type, and it checks for the row and column ranges only in debug mode.
@param mat Input matrix
@param row The zero-based index of row
@param col The zero-based index of column
 */
CV_INLINE  double  cvmGet( const CvMat* mat, int row, int col )
{
    int type;

    type = CV_MAT_TYPE(mat->type);
    assert( (unsigned)row < (unsigned)mat->rows &&
            (unsigned)col < (unsigned)mat->cols );

    if( type == CV_32FC1 )
        return ((float*)(void*)(mat->data.ptr + (size_t)mat->step*row))[col];
    else
    {
        assert( type == CV_64FC1 );
        return ((double*)(void*)(mat->data.ptr + (size_t)mat->step*row))[col];
    }
}

/** @brief Sets a specific element of a single-channel floating-point matrix.

The function is a fast replacement for cvSetReal2D in the case of single-channel floating-point
matrices. It is faster because it is inline, it does fewer checks for array type and array element
type, and it checks for the row and column ranges only in debug mode.
@param mat The matrix
@param row The zero-based index of row
@param col The zero-based index of column
@param value The new value of the matrix element
 */
CV_INLINE  void  cvmSet( CvMat* mat, int row, int col, double value )
{
    int type;
    type = CV_MAT_TYPE(mat->type);
    assert( (unsigned)row < (unsigned)mat->rows &&
            (unsigned)col < (unsigned)mat->cols );

    if( type == CV_32FC1 )
        ((float*)(void*)(mat->data.ptr + (size_t)mat->step*row))[col] = (float)value;
    else
    {
        assert( type == CV_64FC1 );
        ((double*)(void*)(mat->data.ptr + (size_t)mat->step*row))[col] = value;
    }
}


CV_INLINE int cvIplDepth( int type )
{
    int depth = CV_MAT_DEPTH(type);
    return CV_ELEM_SIZE1(depth)*8 | (depth == CV_8S || depth == CV_16S ||
           depth == CV_32S ? IPL_DEPTH_SIGN : 0);
}


/****************************************************************************************\
*                       Multi-dimensional dense array (CvMatND)                          *
\****************************************************************************************/

#define CV_MATND_MAGIC_VAL    0x42430000
#define CV_TYPE_NAME_MATND    "opencv-nd-matrix"

#define CV_MAX_DIM            32
#define CV_MAX_DIM_HEAP       1024

/**
  @deprecated consider using cv::Mat instead
  */
typedef struct
#ifdef __cplusplus
  CV_EXPORTS
#endif
CvMatND
{
    int type;
    int dims;

    int* refcount;
    int hdr_refcount;

    union
    {
        uchar* ptr;
        float* fl;
        double* db;
        int* i;
        short* s;
    } data;

    struct
    {
        int size;
        int step;
    }
    dim[CV_MAX_DIM];

#ifdef __cplusplus
    CvMatND() {}
    CvMatND(const cv::Mat& m);
#endif
}
CvMatND;

#define CV_IS_MATND_HDR(mat) \
    ((mat) != NULL && (((const CvMatND*)(mat))->type & CV_MAGIC_MASK) == CV_MATND_MAGIC_VAL)

#define CV_IS_MATND(mat) \
    (CV_IS_MATND_HDR(mat) && ((const CvMatND*)(mat))->data.ptr != NULL)


/****************************************************************************************\
*                      Multi-dimensional sparse array (CvSparseMat)                      *
\****************************************************************************************/

#define CV_SPARSE_MAT_MAGIC_VAL    0x42440000
#define CV_TYPE_NAME_SPARSE_MAT    "opencv-sparse-matrix"

struct CvSet;

typedef struct
#ifdef __cplusplus
  CV_EXPORTS
#endif
CvSparseMat
{
    int type;
    int dims;
    int* refcount;
    int hdr_refcount;

    struct CvSet* heap;
    void** hashtable;
    int hashsize;
    int valoffset;
    int idxoffset;
    int size[CV_MAX_DIM];

#ifdef __cplusplus
    void copyToSparseMat(cv::SparseMat& m) const;
#endif
}
CvSparseMat;

#ifdef __cplusplus
    CV_EXPORTS CvSparseMat* cvCreateSparseMat(const cv::SparseMat& m);
#endif

#define CV_IS_SPARSE_MAT_HDR(mat) \
    ((mat) != NULL && \
    (((const CvSparseMat*)(mat))->type & CV_MAGIC_MASK) == CV_SPARSE_MAT_MAGIC_VAL)

#define CV_IS_SPARSE_MAT(mat) \
    CV_IS_SPARSE_MAT_HDR(mat)

/**************** iteration through a sparse array *****************/

typedef struct CvSparseNode
{
    unsigned hashval;
    struct CvSparseNode* next;
}
CvSparseNode;

typedef struct CvSparseMatIterator
{
    CvSparseMat* mat;
    CvSparseNode* node;
    int curidx;
}
CvSparseMatIterator;

#define CV_NODE_VAL(mat,node)   ((void*)((uchar*)(node) + (mat)->valoffset))
#define CV_NODE_IDX(mat,node)   ((int*)((uchar*)(node) + (mat)->idxoffset))

/****************************************************************************************\
*                                         Histogram                                      *
\****************************************************************************************/

typedef int CvHistType;

#define CV_HIST_MAGIC_VAL     0x42450000
#define CV_HIST_UNIFORM_FLAG  (1 << 10)

/** indicates whether bin ranges are set already or not */
#define CV_HIST_RANGES_FLAG   (1 << 11)

#define CV_HIST_ARRAY         0
#define CV_HIST_SPARSE        1
#define CV_HIST_TREE          CV_HIST_SPARSE

/** should be used as a parameter only,
   it turns to CV_HIST_UNIFORM_FLAG of hist->type */
#define CV_HIST_UNIFORM       1

typedef struct CvHistogram
{
    int     type;
    CvArr*  bins;
    float   thresh[CV_MAX_DIM][2];  /**< For uniform histograms.                      */
    float** thresh2;                /**< For non-uniform histograms.                  */
    CvMatND mat;                    /**< Embedded matrix header for array histograms. */
}
CvHistogram;

#define CV_IS_HIST( hist ) \
    ((hist) != NULL  && \
     (((CvHistogram*)(hist))->type & CV_MAGIC_MASK) == CV_HIST_MAGIC_VAL && \
     (hist)->bins != NULL)

#define CV_IS_UNIFORM_HIST( hist ) \
    (((hist)->type & CV_HIST_UNIFORM_FLAG) != 0)

#define CV_IS_SPARSE_HIST( hist ) \
    CV_IS_SPARSE_MAT((hist)->bins)

#define CV_HIST_HAS_RANGES( hist ) \
    (((hist)->type & CV_HIST_RANGES_FLAG) != 0)

/****************************************************************************************\
*                      Other supplementary data type definitions                         *
\****************************************************************************************/

/*************************************** CvRect *****************************************/
/** @sa Rect_ */
typedef struct CvRect
{
    int x;
    int y;
    int width;
    int height;

#ifdef __cplusplus
    CvRect(int _x = 0, int _y = 0, int w = 0, int h = 0): x(_x), y(_y), width(w), height(h) {}
    template<typename _Tp>
    CvRect(const cv::Rect_<_Tp>& r): x(cv::saturate_cast<int>(r.x)), y(cv::saturate_cast<int>(r.y)), width(cv::saturate_cast<int>(r.width)), height(cv::saturate_cast<int>(r.height)) {}
    template<typename _Tp>
    operator cv::Rect_<_Tp>() const { return cv::Rect_<_Tp>((_Tp)x, (_Tp)y, (_Tp)width, (_Tp)height); }
#endif
}
CvRect;

/** constructs CvRect structure. */
CV_INLINE  CvRect  cvRect( int x, int y, int width, int height )
{
    CvRect r;

    r.x = x;
    r.y = y;
    r.width = width;
    r.height = height;

    return r;
}


CV_INLINE  IplROI  cvRectToROI( CvRect rect, int coi )
{
    IplROI roi;
    roi.xOffset = rect.x;
    roi.yOffset = rect.y;
    roi.width = rect.width;
    roi.height = rect.height;
    roi.coi = coi;

    return roi;
}


CV_INLINE  CvRect  cvROIToRect( IplROI roi )
{
    return cvRect( roi.xOffset, roi.yOffset, roi.width, roi.height );
}

/*********************************** CvTermCriteria *************************************/

#define CV_TERMCRIT_ITER    1
#define CV_TERMCRIT_NUMBER  CV_TERMCRIT_ITER
#define CV_TERMCRIT_EPS     2

/** @sa TermCriteria
 */
typedef struct CvTermCriteria
{
    int    type;  /**< may be combination of
                     CV_TERMCRIT_ITER
                     CV_TERMCRIT_EPS */
    int    max_iter;
    double epsilon;

#ifdef __cplusplus
    CvTermCriteria(int _type = 0, int _iter = 0, double _eps = 0) : type(_type), max_iter(_iter), epsilon(_eps)  {}
    CvTermCriteria(const cv::TermCriteria& t) : type(t.type), max_iter(t.maxCount), epsilon(t.epsilon)  {}
    operator cv::TermCriteria() const { return cv::TermCriteria(type, max_iter, epsilon); }
#endif

}
CvTermCriteria;

CV_INLINE  CvTermCriteria  cvTermCriteria( int type, int max_iter, double epsilon )
{
    CvTermCriteria t;

    t.type = type;
    t.max_iter = max_iter;
    t.epsilon = (float)epsilon;

    return t;
}


/******************************* CvPoint and variants ***********************************/

typedef struct CvPoint
{
    int x;
    int y;

#ifdef __cplusplus
    CvPoint(int _x = 0, int _y = 0): x(_x), y(_y) {}
    template<typename _Tp>
    CvPoint(const cv::Point_<_Tp>& pt): x((int)pt.x), y((int)pt.y) {}
    template<typename _Tp>
    operator cv::Point_<_Tp>() const { return cv::Point_<_Tp>(cv::saturate_cast<_Tp>(x), cv::saturate_cast<_Tp>(y)); }
#endif
}
CvPoint;

/** constructs CvPoint structure. */
CV_INLINE  CvPoint  cvPoint( int x, int y )
{
    CvPoint p;

    p.x = x;
    p.y = y;

    return p;
}


typedef struct CvPoint2D32f
{
    float x;
    float y;

#ifdef __cplusplus
    CvPoint2D32f(float _x = 0, float _y = 0): x(_x), y(_y) {}
    template<typename _Tp>
    CvPoint2D32f(const cv::Point_<_Tp>& pt): x((float)pt.x), y((float)pt.y) {}
    template<typename _Tp>
    operator cv::Point_<_Tp>() const { return cv::Point_<_Tp>(cv::saturate_cast<_Tp>(x), cv::saturate_cast<_Tp>(y)); }
#endif
}
CvPoint2D32f;

/** constructs CvPoint2D32f structure. */
CV_INLINE  CvPoint2D32f  cvPoint2D32f( double x, double y )
{
    CvPoint2D32f p;

    p.x = (float)x;
    p.y = (float)y;

    return p;
}

/** converts CvPoint to CvPoint2D32f. */
CV_INLINE  CvPoint2D32f  cvPointTo32f( CvPoint point )
{
    return cvPoint2D32f( (float)point.x, (float)point.y );
}

/** converts CvPoint2D32f to CvPoint. */
CV_INLINE  CvPoint  cvPointFrom32f( CvPoint2D32f point )
{
    CvPoint ipt;
    ipt.x = cvRound(point.x);
    ipt.y = cvRound(point.y);

    return ipt;
}


typedef struct CvPoint3D32f
{
    float x;
    float y;
    float z;

#ifdef __cplusplus
    CvPoint3D32f(float _x = 0, float _y = 0, float _z = 0): x(_x), y(_y), z(_z) {}
    template<typename _Tp>
    CvPoint3D32f(const cv::Point3_<_Tp>& pt): x((float)pt.x), y((float)pt.y), z((float)pt.z) {}
    template<typename _Tp>
    operator cv::Point3_<_Tp>() const { return cv::Point3_<_Tp>(cv::saturate_cast<_Tp>(x), cv::saturate_cast<_Tp>(y), cv::saturate_cast<_Tp>(z)); }
#endif
}
CvPoint3D32f;

/** constructs CvPoint3D32f structure. */
CV_INLINE  CvPoint3D32f  cvPoint3D32f( double x, double y, double z )
{
    CvPoint3D32f p;

    p.x = (float)x;
    p.y = (float)y;
    p.z = (float)z;

    return p;
}


typedef struct CvPoint2D64f
{
    double x;
    double y;
}
CvPoint2D64f;

/** constructs CvPoint2D64f structure.*/
CV_INLINE  CvPoint2D64f  cvPoint2D64f( double x, double y )
{
    CvPoint2D64f p;

    p.x = x;
    p.y = y;

    return p;
}


typedef struct CvPoint3D64f
{
    double x;
    double y;
    double z;
}
CvPoint3D64f;

/** constructs CvPoint3D64f structure. */
CV_INLINE  CvPoint3D64f  cvPoint3D64f( double x, double y, double z )
{
    CvPoint3D64f p;

    p.x = x;
    p.y = y;
    p.z = z;

    return p;
}


/******************************** CvSize's & CvBox **************************************/

typedef struct CvSize
{
    int width;
    int height;

#ifdef __cplusplus
    CvSize(int w = 0, int h = 0): width(w), height(h) {}
    template<typename _Tp>
    CvSize(const cv::Size_<_Tp>& sz): width(cv::saturate_cast<int>(sz.width)), height(cv::saturate_cast<int>(sz.height)) {}
    template<typename _Tp>
    operator cv::Size_<_Tp>() const { return cv::Size_<_Tp>(cv::saturate_cast<_Tp>(width), cv::saturate_cast<_Tp>(height)); }
#endif
}
CvSize;

/** constructs CvSize structure. */
CV_INLINE  CvSize  cvSize( int width, int height )
{
    CvSize s;

    s.width = width;
    s.height = height;

    return s;
}

typedef struct CvSize2D32f
{
    float width;
    float height;

#ifdef __cplusplus
    CvSize2D32f(float w = 0, float h = 0): width(w), height(h) {}
    template<typename _Tp>
    CvSize2D32f(const cv::Size_<_Tp>& sz): width(cv::saturate_cast<float>(sz.width)), height(cv::saturate_cast<float>(sz.height)) {}
    template<typename _Tp>
    operator cv::Size_<_Tp>() const { return cv::Size_<_Tp>(cv::saturate_cast<_Tp>(width), cv::saturate_cast<_Tp>(height)); }
#endif
}
CvSize2D32f;

/** constructs CvSize2D32f structure. */
CV_INLINE  CvSize2D32f  cvSize2D32f( double width, double height )
{
    CvSize2D32f s;

    s.width = (float)width;
    s.height = (float)height;

    return s;
}

/** @sa RotatedRect
 */
typedef struct CvBox2D
{
    CvPoint2D32f center;  /**< Center of the box.                          */
    CvSize2D32f  size;    /**< Box width and length.                       */
    float angle;          /**< Angle between the horizontal axis           */
                          /**< and the first side (i.e. length) in degrees */

#ifdef __cplusplus
    CvBox2D(CvPoint2D32f c = CvPoint2D32f(), CvSize2D32f s = CvSize2D32f(), f