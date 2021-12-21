/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2015, Intel Corporation, all rights reserved.
// Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
// Copyright (C) 2015, OpenCV Foundation, all rights reserved.
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

#ifndef OPENCV_CORE_HPP
#define OPENCV_CORE_HPP

#ifndef __cplusplus
#  error core.hpp header must be compiled as C++
#endif

#include "opencv2/core/cvdef.h"
#include "opencv2/core/version.hpp"
#include "opencv2/core/base.hpp"
#include "opencv2/core/cvstd.hpp"
#include "opencv2/core/traits.hpp"
#include "opencv2/core/matx.hpp"
#include "opencv2/core/types.hpp"
#include "opencv2/core/mat.hpp"
#include "opencv2/core/persistence.hpp"

/**
@defgroup core Core functionality
@{
    @defgroup core_basic Basic structures
    @defgroup core_c C structures and operations
    @{
        @defgroup core_c_glue Connections with C++
    @}
    @defgroup core_array Operations on arrays
    @defgroup core_xml XML/YAML Persistence
    @defgroup core_cluster Clustering
    @defgroup core_utils Utility and system functions and macros
    @{
        @defgroup core_utils_sse SSE utilities
        @defgroup core_utils_neon NEON utilities
        @defgroup core_utils_softfloat Softfloat support
    @}
    @defgroup core_opengl OpenGL interoperability
    @defgroup core_ipp Intel IPP Asynchronous C/C++ Converters
    @defgroup core_optim Optimization Algorithms
    @defgroup core_directx DirectX interoperability
    @defgroup core_eigen Eigen support
    @defgroup core_opencl OpenCL support
    @defgroup core_va_intel Intel VA-API/OpenCL (CL-VA) interoperability
    @defgroup core_hal Hardware Acceleration Layer
    @{
        @defgroup core_hal_functions Functions
        @defgroup core_hal_interface Interface
        @defgroup core_hal_intrin Universal intrinsics
        @{
            @defgroup core_hal_intrin_impl Private implementation helpers
        @}
    @}
@}
 */

namespace cv {

//! @addtogroup core_utils
//! @{

/*! @brief Class passed to an error.

This class encapsulates all or almost all necessary
information about the error happened in the program. The exception is
usually constructed and thrown implicitly via CV_Error and CV_Error_ macros.
@see error
 */
class CV_EXPORTS Exception : public std::exception
{
public:
    /*!
     Default constructor
     */
    Exception();
    /*!
     Full constructor. Normally the constructor is not called explicitly.
     Instead, the macros CV_Error(), CV_Error_() and CV_Assert() are used.
    */
    Exception(int _code, const String& _err, const String& _func, const String& _file, int _line);
    virtual ~Exception() throw();

    /*!
     \return the error description and the context as a text string.
    */
    virtual const char *what() const throw();
    void formatMessage();

    String msg; ///< the formatted error message

    int code; ///< error code @see CVStatus
    String err; ///< error description
    String func; ///< function name. Available only when the compiler supports getting it
    String file; ///< source file name where the error has occurred
    int line; ///< line number in the source file where the error has occurred
};

/*! @brief Signals an error and raises the exception.

By default the function prints information about the error to stderr,
then it either stops if cv::setBreakOnError() had been called before or raises the exception.
It is possible to alternate error processing by using #redirectError().
@param exc the exception raisen.
@deprecated drop this version
 */
CV_EXPORTS void error( const Exception& exc );

enum SortFlags { SORT_EVERY_ROW    = 0, //!< each matrix row is sorted independently
                 SORT_EVERY_COLUMN = 1, //!< each matrix column is sorted
                                        //!< independently; this flag and the previous one are
                                        //!< mutually exclusive.
                 SORT_ASCENDING    = 0, //!< each matrix row is sorted in the ascending
                                        //!< order.
                 SORT_DESCENDING   = 16 //!< each matrix row is sorted in the
                                        //!< descending order; this flag and the previous one are also
                                        //!< mutually exclusive.
               };

//! @} core_utils

//! @addtogroup core
//! @{

//! Covariation flags
enum CovarFlags {
    /** The output covariance matrix is calculated as:
       \f[\texttt{scale}   \cdot  [  \texttt{vects}  [0]-  \texttt{mean}  , \texttt{vects}  [1]-  \texttt{mean}  ,...]^T  \cdot  [ \texttt{vects}  [0]- \texttt{mean}  , \texttt{vects}  [1]- \texttt{mean}  ,...],\f]
       The covariance matrix will be nsamples x nsamples. Such an unusual covariance matrix is used
       for fast PCA of a set of very large vectors (see, for example, the EigenFaces technique for
       face recognition). Eigenvalues of this "scrambled" matrix match the eigenvalues of the true
       covariance matrix. The "true" eigenvectors can be easily calculated from the eigenvectors of
       the "scrambled" covariance matrix. */
    COVAR_SCRAMBLED = 0,
    /**The output covariance matrix is calculated as:
        \f[\texttt{scale}   \cdot  [  \texttt{vects}  [0]-  \texttt{mean}  , \texttt{vects}  [1]-  \texttt{mean}  ,...]  \cdot  [ \texttt{vects}  [0]- \texttt{mean}  , \texttt{vects}  [1]- \texttt{mean}  ,...]^T,\f]
        covar will be a square matrix of the same size as the total number of elements in each input
        vector. One and only one of #COVAR_SCRAMBLED and #COVAR_NORMAL must be specified.*/
    COVAR_NORMAL    = 1,
    /** If the flag is specified, the function does not calculate mean from
        the input vectors but, instead, uses the passed mean vector. This is useful if mean has been
        pre-calculated or known in advance, or if the covariance matrix is calculated by parts. In
        this case, mean is not a mean vector of the input sub-set of vectors but rather the mean
        vector of the whole set.*/
    COVAR_USE_AVG   = 2,
    /** If the flag is specified, the covariance matrix is scaled. In the
        "normal" mode, scale is 1./nsamples . In the "scrambled" mode, scale is the reciprocal of the
        total number of elements in each input vector. By default (if the flag is not specified), the
        covariance matrix is not scaled ( scale=1 ).*/
    COVAR_SCALE     = 4,
    /** If the flag is
        specified, all the input vectors are stored as rows of the samples matrix. mean should be a
        single-row vector in this case.*/
    COVAR_ROWS      = 8,
    /** If the flag is
        specified, all the input vectors are stored as columns of the samples matrix. mean should be a
        single-column vector in this case.*/
    COVAR_COLS      = 16
};

//! k-Means flags
enum KmeansFlags {
    /** Select random initial centers in each attempt.*/
    KMEANS_RANDOM_CENTERS     = 0,
    /** Use kmeans++ center initialization by Arthur and Vassilvitskii [Arthur2007].*/
    KMEANS_PP_CENTERS         = 2,
    /** During the first (and possibly the only) attempt, use the
        user-supplied labels instead of computing them from the initial centers. For the second and
        further attempts, use the random or semi-random centers. Use one of KMEANS_\*_CENTERS flag
        to specify the exact method.*/
    KMEANS_USE_INITIAL_LABELS = 1
};

//! type of line
enum LineTypes {
    FILLED  = -1,
    LINE_4  = 4, //!< 4-connected line
    LINE_8  = 8, //!< 8-connected line
    LINE_AA = 16 //!< antialiased line
};

//! Only a subset of Hershey fonts
//! <http://sources.isc.org/utils/misc/hershey-font.txt> are supported
enum HersheyFonts {
    FONT_HERSHEY_SIMPLEX        = 0, //!< normal size sans-serif font
    FONT_HERSHEY_PLAIN          = 1, //!< small size sans-serif font
    FONT_HERSHEY_DUPLEX         = 2, //!< normal size sans-serif font (more complex than FONT_HERSHEY_SIMPLEX)
    FONT_HERSHEY_COMPLEX        = 3, //!< normal size serif font
    FONT_HERSHEY_TRIPLEX        = 4, //!< normal size serif font (more complex than FONT_HERSHEY_COMPLEX)
    FONT_HERSHEY_COMPLEX_SMALL  = 5, //!< smaller version of FONT_HERSHEY_COMPLEX
    FONT_HERSHEY_SCRIPT_SIMPLEX = 6, //!< hand-writing style font
    FONT_HERSHEY_SCRIPT_COMPLEX = 7, //!< more complex variant of FONT_HERSHEY_SCRIPT_SIMPLEX
    FONT_ITALIC                 = 16 //!< flag for italic font
};

enum ReduceTypes { REDUCE_SUM = 0, //!< the output is the sum of all rows/columns of the matrix.
                   REDUCE_AVG = 1, //!< the output is the mean vector of all rows/columns of the matrix.
                   REDUCE_MAX = 2, //!< the output is the maximum (column/row-wise) of all rows/columns of the matrix.
                   REDUCE_MIN = 3  //!< the output is the minimum (column/row-wise) of all rows/columns of the matrix.
                 };


/** @brief Swaps two matrices
*/
CV_EXPORTS void swap(Mat& a, Mat& b);
/** @overload */
CV_EXPORTS void swap( UMat& a, UMat& b );

//! @} core

//! @addtogroup core_array
//! @{

/** @brief Computes the source location of an extrapolated pixel.

The function computes and returns the coordinate of a donor pixel corresponding to the specified
extrapolated pixel when using the specified extrapolation border mode. For example, if you use
cv::BORDER_WRAP mode in the horizontal direction, cv::BORDER_REFLECT_101 in the vertical direction and
want to compute value of the "virtual" pixel Point(-5, 100) in a floating-point image img , it
looks like:
@code{.cpp}
    float val = img.at<float>(borderInterpolate(100, img.rows, cv::BORDER_REFLECT_101),
                              borderInterpolate(-5, img.cols, cv::BORDER_WRAP));
@endcode
Normally, the function is not called directly. It is used inside filtering functions and also in
copyMakeBorder.
@param p 0-based coordinate of the extrapolated pixel along one of the axes, likely \<0 or \>= len
@param len Length of the array along the corresponding axis.
@param borderType Border type, one of the #BorderTypes, except for #BORDER_TRANSPARENT and
#BORDER_ISOLATED . When borderType==#BORDER_CONSTANT , the function always returns -1, regardless
of p and len.

@sa copyMakeBorder
*/
CV_EXPORTS_W int borderInterpolate(int p, int len, int borderType);

/** @example copyMakeBorder_demo.cpp
An example using copyMakeBorder function
 */
/** @brief Forms a border around an image.

The function copies the source image into the middle of the destination image. The areas to the
left, to the right, above and below the copied source image will be filled with extrapolated
pixels. This is not what filtering functions based on it do (they extrapolate pixels on-fly), but
what other more complex functions, including your own, may do to simplify image boundary handling.

The function supports the mode when src is already in the middle of dst . In this case, the
function does not copy src itself but simply constructs the border, for example:

@code{.cpp}
    // let border be the same in all directions
    int border=2;
    // constructs a larger image to fit both the image and the border
    Mat gray_buf(rgb.rows + border*2, rgb.cols + border*2, rgb.depth());
    // select the middle part of it w/o copying data
    Mat gray(gray_canvas, Rect(border, border, rgb.cols, rgb.rows));
    // convert image from RGB to grayscale
    cvtColor(rgb, gray, COLOR_RGB2GRAY);
    // form a border in-place
    copyMakeBorder(gray, gray_buf, border, border,
                   border, border, BORDER_REPLICATE);
    // now do some custom filtering ...
    ...
@endcode
@note When the source image is a part (ROI) of a bigger image, the function will try to use the
pixels outside of the ROI to form a border. To disable this feature and always do extrapolation, as
if src was not a ROI, use borderType | #BORDER_ISOLATED.

@param src Source image.
@param dst Destination image of the same type as src and the size Size(src.cols+left+right,
src.rows+top+bottom) .
@param top
@param bottom
@param left
@param right Parameter specifying how many pixels in each direction from the source image rectangle
to extrapolate. For example, top=1, bottom=1, left=1, right=1 mean that 1 pixel-wide border needs
to be built.
@param borderType Border type. See borderInterpolate for details.
@param value Border value if borderType==BORDER_CONSTANT .

@sa  borderInterpolate
*/
CV_EXPORTS_W void copyMakeBorder(InputArray src, OutputArray dst,
                                 int top, int bottom, int left, int right,
                                 int borderType, const Scalar& value = Scalar() );

/** @brief Calculates the per-element sum of two arrays or an array and a scalar.

The function add calculates:
- Sum of two arrays when both input arrays have the same size and the same number of channels:
\f[\texttt{dst}(I) =  \texttt{saturate} ( \texttt{src1}(I) +  \texttt{src2}(I)) \quad \texttt{if mask}(I) \ne0\f]
- Sum of an array and a scalar when src2 is constructed from Scalar or has the same number of
elements as `src1.channels()`:
\f[\texttt{dst}(I) =  \texttt{saturate} ( \texttt{src1}(I) +  \texttt{src2} ) \quad \texttt{if mask}(I) \ne0\f]
- Sum of a scalar and an array when src1 is constructed from Scalar or has the same number of
elements as `src2.channels()`:
\f[\texttt{dst}(I) =  \texttt{saturate} ( \texttt{src1} +  \texttt{src2}(I) ) \quad \texttt{if mask}(I) \ne0\f]
where `I` is a multi-dimensional index of array elements. In case of multi-channel arrays, each
channel is processed independently.

The first function in the list above can be replaced with matrix expressions:
@code{.cpp}
    dst = src1 + src2;
    dst += src1; // equivalent to add(dst, src1, dst);
@endcode
The input arrays and the output array can all have the same or different depths. For example, you
can add a 16-bit unsigned array to a 8-bit signed array and store the sum as a 32-bit
floating-point array. Depth of the output array is determined by the dtype parameter. In the second
and third cases above, as well as in the first case, when src1.depth() == src2.depth(), dtype can
be set to the default -1. In this case, the output array will have the same depth as the input
array, be it src1, src2 or both.
@note Saturation is not applied when the output array has the depth CV_32S. You may even get
result of an incorrect sign in the case of overflow.
@param src1 first input array or a scalar.
@param src2 second input array or a scalar.
@param dst output array that has the same size and number of channels as the input array(s); the
depth is defined by dtype or src1/src2.
@param mask optional operation mask - 8-bit single channel array, that specifies elements of the
output array to be changed.
@param dtype optional depth of the output array (see the discussion below).
@sa subtract, addWeighted, scaleAdd, Mat::convertTo
*/
CV_EXPORTS_W void add(InputArray src1, InputArray src2, OutputArray dst,
                      InputArray mask = noArray(), int dtype = -1);

/** @brief Calculates the per-element difference between two arrays or array and a scalar.

The function subtract calculates:
- Difference between two arrays, when both input arrays have the same size and the same number of
channels:
    \f[\texttt{dst}(I) =  \texttt{saturate} ( \texttt{src1}(I) -  \texttt{src2}(I)) \quad \texttt{if mask}(I) \ne0\f]
- Difference between an array and a scalar, when src2 is constructed from Scalar or has the same
number of elements as `src1.channels()`:
    \f[\texttt{dst}(I) =  \texttt{saturate} ( \texttt{src1}(I) -  \texttt{src2} ) \quad \texttt{if mask}(I) \ne0\f]
- Difference between a scalar and an array, when src1 is constructed from Scalar or has the same
number of elements as `src2.channels()`:
    \f[\texttt{dst}(I) =  \texttt{saturate} ( \texttt{src1} -  \texttt{src2}(I) ) \quad \texttt{if mask}(I) \ne0\f]
- The reverse difference between a scalar and an array in the case of `SubRS`:
    \f[\texttt{dst}(I) =  \texttt{saturate} ( \texttt{src2} -  \texttt{src1}(I) ) \quad \texttt{if mask}(I) \ne0\f]
where I is a multi-dimensional index of array elements. In case of multi-channel arrays, each
channel is processed independently.

The first function in the list above can be replaced with matrix expressions:
@code{.cpp}
    dst = src1 - src2;
    dst -= src1; // equivalent to subtract(dst, src1, dst);
@endcode
The input arrays and the output array can all have the same or different depths. For example, you
can subtract to 8-bit unsigned arrays and store the difference in a 16-bit signed array. Depth of
the output array is determined by dtype parameter. In the second and third cases above, as well as
in the first case, when src1.depth() == src2.depth(), dtype can be set to the default -1. In this
case the output array will have the same depth as the input array, be it src1, src2 or both.
@note Saturation is not applied when the output array has the depth CV_32S. You may even get
result of an incorrect sign in the case of overflow.
@param src1 first input array or a scalar.
@param src2 second input array or a scalar.
@param dst output array of the same size and the same number of channels as the input array.
@param mask optional operation mask; this is an 8-bit single channel array that specifies elements
of the output array to be changed.
@param dtype optional depth of the output array
@sa  add, addWeighted, scaleAdd, Mat::convertTo
  */
CV_EXPORTS_W void subtract(InputArray src1, InputArray src2, OutputArray dst,
                           InputArray mask = noArray(), int dtype = -1);


/** @brief Calculates the per-element scaled product of two arrays.

The function multiply calculates the per-element product of two arrays:

\f[\texttt{dst} (I)= \texttt{saturate} ( \texttt{scale} \cdot \texttt{src1} (I)  \cdot \texttt{src2} (I))\f]

There is also a @ref MatrixExpressions -friendly variant of the first function. See Mat::mul .

For a not-per-element matrix product, see gemm .

@note Saturation is not applied when the output array has the depth
CV_32S. You may even get result of an incorrect sign in the case of
overflow.
@param src1 first input array.
@param src2 second input array of the same size and the same type as src1.
@param dst output array of the same size and type as src1.
@param scale optional scale factor.
@param dtype optional depth of the output array
@sa add, subtract, divide, scaleAdd, addWeighted, accumulate, accumulateProduct, accumulateSquare,
Mat::convertTo
*/
CV_EXPORTS_W void multiply(InputArray src1, InputArray src2,
                           OutputArray dst, double scale = 1, int dtype = -1);

/** @brief Performs per-element division of two arrays or a scalar by an array.

The function cv::divide divides one array by another:
\f[\texttt{dst(I) = saturate(src1(I)*scale/src2(I))}\f]
or a scalar by an array when there is no src1 :
\f[\texttt{dst(I) = saturate(scale/src2(I))}\f]

When src2(I) is zero, dst(I) will also be zero. Different channels of
multi-channel arrays are processed independently.

@note Saturation is not applied when the output array has the depth CV_32S. You may even get
result of an incorrect sign in the case of overflow.
@param src1 first input array.
@param src2 second input array of the same size and type as src1.
@param scale scalar factor.
@param dst output array of the same size and type as src2.
@param dtype optional depth of the output array; if -1, dst will have depth src2.depth(), but in
case of an array-by-array division, you can only pass -1 when src1.depth()==src2.depth().
@sa  multiply, add, subtract
*/
CV_EXPORTS_W void divide(InputArray src1, InputArray src2, OutputArray dst,
                         double scale = 1, int dtype = -1);

/** @overload */
CV_EXPORTS_W void divide(double scale, InputArray src2,
                         OutputArray dst, int dtype = -1);

/** @brief Calculates the sum of a scaled array and another array.

The function scaleAdd is one of the classical primitive linear algebra operations, known as DAXPY
or SAXPY in [BLAS](http://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms). It calculates
the sum of a scaled array and another array:
\f[\texttt{dst} (I)= \texttt{scale} \cdot \texttt{src1} (I) +  \texttt{src2} (I)\f]
The function can also be emulated with a matrix expression, for example:
@code{.cpp}
    Mat A(3, 3, CV_64F);
    ...
    A.row(0) = A.row(1)*2 + A.row(2);
@endcode
@param src1 first input array.
@param alpha scale factor for the first array.
@param src2 second input array of the same size and type as src1.
@param dst output array of the same size and type as src1.
@sa add, addWeighted, subtract, Mat::dot, Mat::convertTo
*/
CV_EXPORTS_W void scaleAdd(InputArray src1, double alpha, InputArray src2, OutputArray dst);

/** @example AddingImagesTrackbar.cpp

 */
/** @brief Calculates the weighted sum of two arrays.

The function addWeighted calculates the weighted sum of two arrays as follows:
\f[\texttt{dst} (I)= \texttt{saturate} ( \texttt{src1} (I)* \texttt{alpha} +  \texttt{src2} (I)* \texttt{beta} +  \texttt{gamma} )\f]
where I is a multi-dimensional index of array elements. In case of multi-channel arrays, each
channel is processed independently.
The function can be replaced with a matrix expression:
@code{.cpp}
    dst = src1*alpha + src2*beta + gamma;
@endcode
@note Saturation is not applied when the output array has the depth CV_32S. You may even get
result of an incorrect sign in the case of overflow.
@param src1 first input array.
@param alpha weight of the first array elements.
@param src2 second input array of the same size and channel number as src1.
@param beta weight of the second array elements.
@param gamma scalar added to each sum.
@param dst output array that has the same size and number of channels as the input arrays.
@param dtype optional depth of the output array; when both input arrays have the same depth, dtype
can be set to -1, which will be equivalent to src1.depth().
@sa  add, subtract, scaleAdd, Mat::convertTo
*/
CV_EXPORTS_W void addWeighted(InputArray src1, double alpha, InputArray src2,
                              double beta, double gamma, OutputArray dst, int dtype = -1);

/** @brief Scales, calculates absolute values, and converts the result to 8-bit.

On each element of the input array, the function convertScaleAbs
performs three operations sequentially: scaling, taking an absolute
value, conversion to an unsigned 8-bit type:
\f[\texttt{dst} (I)= \texttt{saturate\_cast<uchar>} (| \texttt{src} (I)* \texttt{alpha} +  \texttt{beta} |)\f]
In case of multi-channel arrays, the function processes each channel
independently. When the output is not 8-bit, the operation can be
emulated by calling the Mat::convertTo method (or by using matrix
expressions) and then by calculating an absolute value of the result.
For example:
@code{.cpp}
    Mat_<float> A(30,30);
    randu(A, Scalar(-100), Scalar(100));
    Mat_<float> B = A*5 + 3;
    B = abs(B);
    // Mat_<float> B = abs(A*5+3) will also do the job,
    // but it will allocate a temporary matrix
@endcode
@param src input array.
@param dst output array.
@param alpha optional scale factor.
@param beta optional delta added to the scaled values.
@sa  Mat::convertTo, cv::abs(const Mat&)
*/
CV_EXPORTS_W void convertScaleAbs(InputArray src, OutputArray dst,
                                  double alpha = 1, double beta = 0);

/** @brief Converts an array to half precision floating number.

This function converts FP32 (single precision floating point) from/to FP16 (half precision floating point).  The input array has to have type of CV_32F or
CV_16S to represent the bit depth.  If the input array is neither of them, the function will raise an error.
The format of half precision floating point is defined in IEEE 754-2008.

@param src input array.
@param dst output array.
*/
CV_EXPORTS_W void convertFp16(InputArray src, OutputArray dst);

/** @brief Performs a look-up table transform of an array.

The function LUT fills the output array with values from the look-up table. Indices of the entries
are taken from the input array. That is, the function processes each element of src as follows:
\f[\texttt{dst} (I)  \leftarrow \texttt{lut(src(I) + d)}\f]
where
\f[d =  \fork{0}{if \(\texttt{src}\) has depth \(\texttt{CV_8U}\)}{128}{if \(\texttt{src}\) has depth \(\texttt{CV_8S}\)}\f]
@param src input array of 8-bit elements.
@param lut look-up table of 256 elements; in case of multi-channel input array, the table should
either have a single channel (in this case the same table is used for all channels) or the same
number of channels as in the input array.
@param dst output array of the same size and number of channels as src, and the same depth as lut.
@sa  convertScaleAbs, Mat::convertTo
*/
CV_EXPORTS_W void LUT(InputArray src, InputArray lut, OutputArray dst);

/** @brief Calculates the sum of array elements.

The function cv::sum calculates and returns the sum of array elements,
independently for each channel.
@param src input array that must have from 1 to 4 channels.
@sa  countNonZero, mean, meanStdDev, norm, minMaxLoc, reduce
*/
CV_EXPORTS_AS(sumElems) Scalar sum(InputArray src);

/** @brief Counts non-zero array elements.

The function returns the number of non-zero elements in src :
\f[\sum _{I: \; \texttt{src} (I) \ne0 } 1\f]
@param src single-channel array.
@sa  mean, meanStdDev, norm, minMaxLoc, calcCovarMatrix
*/
CV_EXPORTS_W int countNonZero( InputArray src );

/** @brief Returns the list of locations of non-zero pixels

Given a binary matrix (likely returned from an operation such
as threshold(), compare(), >, ==, etc, return all of
the non-zero indices as a cv::Mat or std::vector<cv::Point> (x,y)
For example:
@code{.cpp}
    cv::Mat binaryImage; // input, binary image
    cv::Mat locations;   // output, locations of non-zero pixels
    cv::findNonZero(binaryImage, locations);

    // access pixel coordinates
    Point pnt = locations.at<Point>(i);
@endcode
or
@code{.cpp}
    cv::Mat binaryImage; // input, binary image
    vector<Point> locations;   // output, locations of non-zero pixels
    cv::findNonZero(binaryImage, locations);

    // access pixel coordinates
    Point pnt = locations[i];
@endcode
@param src single-channel array (type CV_8UC1)
@param idx the output array, type of cv::Mat or std::vector<Point>, corresponding to non-zero indices in the input
*/
CV_EXPORTS_W void findNonZero( InputArray src, OutputArray idx );

/** @brief Calculates an average (mean) of array elements.

The function cv::mean calculates the mean value M of array elements,
independently for each channel, and return it:
\f[\begin{array}{l} N =  \sum _{I: \; \texttt{mask} (I) \ne 0} 1 \\ M_c =  \left ( \sum _{I: \; \texttt{mask} (I) \ne 0}{ \texttt{mtx} (I)_c} \right )/N \end{array}\f]
When all the mask elements are 0's, the function returns Scalar::all(0)
@param src input array that should have from 1 to 4 channels so that the result can be stored in
Scalar_ .
@param mask optional operation mask.
@sa  countNonZero, meanStdDev, norm, minMaxLoc
*/
CV_EXPORTS_W Scalar mean(InputArray src, InputArray mask = noArray());

/** Calculates a mean and standard deviation of array elements.

The function cv::meanStdDev calculates the mean and the standard deviation M
of array elements independently for each channel and returns it via the
output parameters:
\f[\begin{array}{l} N =  \sum _{I, \texttt{mask} (I)  \ne 0} 1 \\ \texttt{mean} _c =  \frac{\sum_{ I: \; \texttt{mask}(I) \ne 0} \texttt{src} (I)_c}{N} \\ \texttt{stddev} _c =  \sqrt{\frac{\sum_{ I: \; \texttt{mask}(I) \ne 0} \left ( \texttt{src} (I)_c -  \texttt{mean} _c \right )^2}{N}} \end{array}\f]
When all the mask elements are 0's, the function returns
mean=stddev=Scalar::all(0).
@note The calculated standard deviation is only the diagonal of the
complete normalized covariance matrix. If the full matrix is needed, you
can reshape the multi-channel array M x N to the single-channel array
M\*N x mtx.channels() (only possible when the matrix is continuous) and
then pass the matrix to calcCovarMatrix .
@param src input array that should have from 1 to 4 channels so that the results can be stored in
Scalar_ 's.
@param mean output parameter: calculated mean value.
@param stddev output parameter: calculated standard deviation.
@param mask optional operation mask.
@sa  countNonZero, mean, norm, minMaxLoc, calcCovarMatrix
*/
CV_EXPORTS_W void meanStdDev(InputArray src, OutputArray mean, OutputArray stddev,
                             InputArray mask=noArray());

/** @brief Calculates the  absolute norm of an array.

This version of #norm calculates the absolute norm of src1. The type of norm to calculate is specified using #NormTypes.

As example for one array consider the function \f$r(x)= \begin{pmatrix} x \\ 1-x \end{pmatrix}, x \in [-1;1]\f$.
The \f$ L_{1}, L_{2} \f$ and \f$ L_{\infty} \f$ norm for the sample value \f$r(-1) = \begin{pmatrix} -1 \\ 2 \end{pmatrix}\f$
is calculated as follows
\f{align*}
    \| r(-1) \|_{L_1} &= |-1| + |2| = 3 \\
    \| r(-1) \|_{L_2} &= \sqrt{(-1)^{2} + (2)^{2}} = \sqrt{5} \\
    \| r(-1) \|_{L_\infty} &= \max(|-1|,|2|) = 2
\f}
and for \f$r(0.5) = \begin{pmatrix} 0.5 \\ 0.5 \end{pmatrix}\f$ the calculation is
\f{align*}
    \| r(0.5) \|_{L_1} &= |0.5| + |0.5| = 1 \\
    \| r(0.5) \|_{L_2} &= \sqrt{(0.5)^{2} + (0.5)^{2}} = \sqrt{0.5} \\
    \| r(0.5) \|_{L_\infty} &= \max(|0.5|,|0.5|) = 0.5.
\f}
The following graphic shows all values for the three norm functions \f$\| r(x) \|_{L_1}, \| r(x) \|_{L_2}\f$ and \f$\| r(x) \|_{L_\infty}\f$.
It is notable that the \f$ L_{1} \f$ norm forms the upper and the \f$ L_{\infty} \f$ norm forms the lower border for the example function \f$ r(x) \f$.
![Graphs for the different norm functions from the above example](pics/NormTypes_OneArray_1-2-INF.png)

When the mask parameter is specified and it is not empty, the norm is

If normType is not specified, #NORM_L2 is used.
calculated only over the region specified by the mask.

Multi-channel input arrays are treated as single-channel arrays, that is,
the results for all channels are combined.

Hamming norms can only be calculated with CV_8U depth arrays.

@param src1 first input array.
@param normType type of the norm (see #NormTypes).
@param mask optional operation mask; it must have the same size as src1 and CV_8UC1 type.
*/
CV_EXPORTS_W double norm(InputArray src1, int normType = NORM_L2, InputArray mask = noArray());

/** @brief Calculates an absolute difference norm or a relative difference norm.

This version of cv::norm calculates the absolute difference norm
or the relative difference norm of arrays src1 and src2.
The type of norm to calculate is specified using #NormTypes.

@param src1 first input array.
@param src2 second input array of the same size and the same type as src1.
@param normType type of the norm (see #NormTypes).
@param mask optional operation mask; it must have the same size as src1 and CV_8UC1 type.
*/
CV_EXPORTS_W double norm(InputArray src1, InputArray src2,
                         int normType = NORM_L2, InputArray mask = noArray());
/** @overload
@param src first input array.
@param normType type of the norm (see #NormTypes).
*/
CV_EXPORTS double norm( const SparseMat& src, int normType );

/** @brief Computes the Peak Signal-to-Noise Ratio (PSNR) image quality metric.

This function calculates the Peak Signal-to-Noise Ratio (PSNR) image quality metric in decibels (dB), between two input arrays src1 and src2. Arrays must have depth CV_8U.

The PSNR is calculated as follows:

\f[
\texttt{PSNR} = 10 \cdot \log_{10}{\left( \frac{R^2}{MSE} \right) }
\f]

where R is the maximum integer value of depth CV_8U (255) and MSE is the mean squared error between the two arrays.

@param src1 first input array.
@param src2 second input array of the same size as src1.

  */
CV_EXPORTS_W double PSNR(InputArray src1, InputArray src2);

/** @brief naive nearest neighbor finder

see http://en.wikipedia.org/wiki/Nearest_neighbor_search
@todo document
  */
CV_EXPORTS_W void batchDistance(InputArray src1, InputArray src2,
                                OutputArray dist, int dtype, OutputArray nidx,
                                int normType = NORM_L2, int K = 0,
                                InputArray mask = noArray(), int update = 0,
                                bool crosscheck = false);

/** @brief Normalizes the norm or value range of an array.

The function cv::normalize normalizes scale and shift the input array elements so that
\f[\| \texttt{dst} \| _{L_p}= \texttt{alpha}\f]
(where p=Inf, 1 or 2) when normType=NORM_INF, NORM_L1, or NORM_L2, respectively; or so that
\f[\min _I  \texttt{dst} (I)= \texttt{alpha} , \, \, \max _I  \texttt{dst} (I)= \texttt{beta}\f]

when normType=NORM_MINMAX (for dense arrays only). The optional mask specifies a sub-array to be
normalized. This means that the norm or min-n-max are calculated over the sub-array, and then this
sub-array is modified to be normalized. If you want to only use the mask to calculate the norm or
min-max but modify the whole array, you can use norm and Mat::convertTo.

In case of sparse matrices, only the non-zero values are analyzed and transformed. Because of this,
the range transformation for sparse matrices is not allowed since it can shift the zero level.

Possible usage with some positive example data:
@code{.cpp}
    vector<double> positiveData = { 2.0, 8.0, 10.0 };
    vector<double> normalizedData_l1, normalizedData_l2, normalizedData_inf, normalizedData_minmax;

    // Norm to probability (total count)
    // sum(numbers) = 20.0
    // 2.0      0.1     (2.0/20.0)
    // 8.0      0.4     (8.0/20.0)
    // 10.0     0.5     (10.0/20.0)
    normalize(positiveData, normalizedData_l1, 1.0, 0.0, NORM_L1);

    // Norm to unit vector: ||positiveData|| = 1.0
    // 2.0      0.15
    // 8.0      0.62
    // 10.0     0.77
    normalize(positiveData, normalizedData_l2, 1.0, 0.0, NORM_L2);

    // Norm to max element
    // 2.0      0.2     (2.0/10.0)
    // 8.0      0.8     (8.0/10.0)
    // 10.0     1.0     (10.0/10.0)
    normalize(positiveData, normalizedData_inf, 1.0, 0.0, NORM_INF);

    // Norm to range [0.0;1.0]
    // 2.0      0.0     (shift to left border)
    // 8.0      0.75    (6.0/8.0)
    // 10.0     1.0     (shift to right border)
    normalize(positiveData, normalizedData_minmax, 1.0, 0.0, NORM_MINMAX);
@endcode

@param src input array.
@param dst output array of the same size as src .
@param alpha norm value to normalize to or the lower range boundary in case of the range
normalization.
@param beta upper range boundary in case of the range normalization; it is not used for the norm
normalization.
@param norm_type normalization type (see cv::NormTypes).
@param dtype when negative, the output array has the same type as src; otherwise, it has the same
number of channels as src and the depth =CV_MAT_DEPTH(dtype).
@param mask optional operation mask.
@sa norm, Mat::convertTo, SparseMat::convertTo
*/
CV_EXPORTS_W void normalize( InputArray src, InputOutputArray dst, double alpha = 1, double beta = 0,
                             int norm_type = NORM_L2, int dtype = -1, InputArray mask = noArray());

/** @overload
@param src input array.
@param dst output array of the same size as src .
@param alpha norm value to normalize to or the lower range boundary in case of the range
normalization.
@param normType normalization type (see cv::NormTypes).
*/
CV_EXPORTS void normalize( const SparseMat& src, SparseMat& dst, double alpha, int normType );

/** @brief Finds the global minimum and maximum in an array.

The function cv::minMaxLoc finds the minimum and maximum element values and their positions. The
extremums are searched across the whole array or, if mask is not an empty array, in the specified
array region.

The function do not work with multi-channel arrays. If you need to find minimum or maximum
elements across all the channels, use Mat::reshape first to reinterpret the array as
single-channel. Or you may extract the particular channel using either extractImageCOI , or
mixChannels , or split .
@param src input single-channel array.
@param minVal pointer to the returned minimum value; NULL is used if not required.
@param maxVal pointer to the returned maximum value; NULL is used if not required.
@param minLoc pointer to the returned minimum location (in 2D case); NULL is used if not required.
@param maxLoc pointer to the returned maximum location (in 2D case); NULL is used if not required.
@param mask optional mask used to select a sub-array.
@sa max, min, compare, inRange, extractImageCOI, mixChannels, split, Mat::reshape
*/
CV_EXPORTS_W void minMaxLoc(InputArray src, CV_OUT double* minVal,
                            CV_OUT double* maxVal = 0, CV_OUT Point* minLoc = 0,
                            CV_OUT Point* maxLoc = 0, InputArray mask = noArray());


/** @brief Finds the global minimum and maximum in an array

The function cv::minMaxIdx finds the minimum and maximum element values and their positions. The
extremums are searched across the whole array or, if mask is not an empty array, in the specified
array region. The function does not work with multi-channel arrays. If you need to find minimum or
maximum elements across all the channels, use Mat::reshape first to reinterpret the array as
single-channel. Or you may extract the particular channel using either extractImageCOI , or
mixChannels , or split . In case of a sparse matrix, the minimum is found among non-zero elements
only.
@note When minIdx is not NULL, it must have at least 2 elements (as well as maxIdx), even if src is
a single-row or single-column matrix. In OpenCV (following MATLAB) each array has at least 2
dimensions, i.e. single-column matrix is Mx1 matrix (and therefore minIdx/maxIdx will be
(i1,0)/(i2,0)) and single-row matrix is 1xN matrix (and therefore minIdx/maxIdx will be
(0,j1)/(0,j2)).
@param src input single-channel array.
@param minVal pointer to the returned minimum value; NULL is used if not required.
@param maxVal pointer to the returned maximum value; NULL is used if not required.
@param minIdx pointer to the returned minimum location (in nD case); NULL is used if not required;
Otherwise, it must point to an array of src.dims elements, the coordinates of the minimum element
in each dimension are stored there sequentially.
@param maxIdx pointer to the returned maximum location (in nD case). NULL is used if not required.
@param mask specified array region
*/
CV_EXPORTS void minMaxIdx(InputArray src, double* minVal, double* maxVal = 0,
                          int* minIdx = 0, int* maxIdx = 0, InputArray mask = noArray());

/** @overload
@param a input single-channel array.
@param minVal pointer to the returned minimum value; NULL is used if not required.
@param maxVal pointer to the returned maximum value; NULL is used if not required.
@param minIdx pointer to the returned minimum location (in nD case); NULL is used if not required;
Otherwise, it must point to an array of src.dims elements, the coordinates of the minimum element
in each dimension are stored there sequentially.
@param maxIdx pointer to the returned maximum location (in nD case). NULL is used if not required.
*/
CV_EXPORTS void minMaxLoc(const SparseMat& a, double* minVal,
                          double* maxVal, int* minIdx = 0, int* maxIdx = 0);

/** @brief Reduces a matrix to a vector.

The function #reduce reduces the matrix to a vector by treating the matrix rows/columns as a set of
1D vectors and performing the specified operation on the vectors until a single row/column is
obtained. For example, the function can be used to compute horizontal and vertical projections of a
raster image. In case of #REDUCE_MAX and #REDUCE_MIN , the output image should have the same type as the source one.
In case of #REDUCE_SUM and #REDUCE_AVG , the output may have a larger element bit-depth to preserve accuracy.
And multi-channel arrays are also supported in these two reduction modes.

The following code demonstrates its usage for a single channel matrix.
@snippet snippets/core_reduce.cpp example

And the following code demonstrates its usage for a two-channel matrix.
@snippet snippets/core_reduce.cpp example2

@param src input 2D matrix.
@param dst output vector. Its size and type is defined by dim and dtype parameters.
@param dim dimension index along which the matrix is reduced. 0 means that the matrix is reduced to
a single row. 1 means that the matrix is reduced to a single column.
@param rtype reduction operation that could be one of #ReduceTypes
@param dtype when negative, the output vector will have the same type as the input matrix,
otherwise, its type will be CV_MAKE_TYPE(CV_MAT_DEPTH(dtype), src.channels()).
@sa repeat
*/
CV_EXPORTS_W void reduce(InputArray src, OutputArray dst, int dim, int rtype, int dtype = -1);

/** @brief Creates one multi-channel array out of several single-channel ones.

The function cv::merge merges several arrays to make a single multi-channel array. That is, each
element of the output array will be a concatenation of the elements of the input arrays, where
elements of i-th input array are treated as mv[i].channels()-element vectors.

The function cv::split does the reverse operation. If you need to shuffle channels in some other
advanced way, use cv::mixChannels.

The following example shows how to merge 3 single channel matrices into a single 3-channel matrix.
@snippet snippets/core_merge.cpp example

@param mv input array of matrices to be merged; all the matrices in mv must have the same
size and the same depth.
@param count number of input matrices when mv is a plain C array; it must be greater than zero.
@param dst output array of the same size and the same depth as mv[0]; The number of channels will
be equal to the parameter count.
@sa  mixChannels, split, Mat::reshape
*/
CV_EXPORTS void merge(const Mat* mv, size_t count, OutputArray dst);

/** @overload
@param mv input vector of matrices to be merged; all the matrices in mv must have the same
size and the same depth.
@param dst output array of the same size and the same depth as mv[0]; The number of channels will
be the total number of channels in the matrix array.
  */
CV_EXPORTS_W void merge(InputArrayOfArrays mv, OutputArray dst);

/** @brief Divides a multi-channel array into several single-channel arrays.

The function cv::split splits a multi-channel array into separate single-channel arrays:
\f[\texttt{mv} [c](I) =  \texttt{src} (I)_c\f]
If you need to extract a single channel or do some other sophisticated channel permutation, use
mixChannels .

The following example demonstrates how to split a 3-channel matrix into 3 single channel matrices.
@snippet snippets/core_split.cpp example

@param src input multi-channel array.
@param mvbegin output array; the number of arrays must match src.channels(); the arrays themselves are
reallocated, if needed.
@sa merge, mixChannels, cvtColor
*/
CV_EXPORTS void split(const Mat& src, Mat* mvbegin);

/** @overload
@param m input multi-channel array.
@param mv output vector of arrays; the arrays themselves are reallocated, if needed.
*/
CV_EXPORTS_W void split(InputArray m, OutputArrayOfArrays mv);

/** @brief Copies specified channels from input arrays to the specified channels of
output arrays.

The function cv::mixChannels provides an advanced mechanism for shuffling image channels.

cv::split,cv::merge,cv::extractChannel,cv::insertChannel and some forms of cv::cvtColor are partial cases of cv::mixChannels.

In the example below, the code splits a 4-channel BGRA image into a 3-channel BGR (with B and R
channels swapped) and a separate alpha-channel image:
@code{.cpp}
    Mat bgra( 100, 100, CV_8UC4, Scalar(255,0,0,255) );
    Mat bgr( bgra.rows, bgra.cols, CV_8UC3 );
    Mat alpha( bgra.rows, bgra.cols, CV_8UC1 );

    // forming an array of matrices is a quite efficient operation,
    // because the matrix data is not copied, only the headers
    Mat out[] = { bgr, alpha };
    // bgra[0] -> bgr[2], bgra[1] -> bgr[1],
    // bgra[2] -> bgr[0], bgra[3] -> alpha[0]
    int from_to[] = { 0,2, 1,1, 2,0, 3,3 };
    mixChannels( &bgra, 1, out, 2, from_to, 4 );
@endcode
@note Unlike many other new-style C++ functions in OpenCV (see the introduction section and
Mat::create ), cv::mixChannels requires the output arrays to be pre-allocated before calling the
function.
@param src input array or vector of matrices; all of the matrices must have the same size and the
same depth.
@param nsrcs number of matrices in `src`.
@param dst output array or vector of matrices; all the matrices **must be allocated**; their size and
depth must be the same as in `src[0]`.
@param ndsts number of matrices in `dst`.
@param fromTo array of index pairs specifying which channels are copied and where; fromTo[k\*2] is
a 0-based index of the input channel in src, fromTo[k\*2+1] is an index of the output channel in
dst; the continuous channel numbering is used: the first input image channels are indexed from 0 to
src[0].channels()-1, the second input image channels are indexed from src[0].channels() to
src[0].channels() + src[1].channels()-1, and so on, the same scheme is used for the output image
channels; as a special case, when fromTo[k\*2] is negative, the corresponding output channel is
filled with zero .
@param npairs number of index pairs in `fromTo`.
@sa split, merge, extractChannel, insertChannel, cvtColor
*/
CV_EXPORTS void mixChannels(const Mat* src, size_t nsrcs, Mat* dst, size_t ndsts,
                            const int* fromTo, size_t npairs);

/** @overload
@param src input array or vector of matrices; all of the matrices must have the same size and the
same depth.
@param dst output array or vector of matrices; all the matrices **must be allocated**; their size and
depth must be the same as in src[0].
@param fromTo array of index pairs specifying which channels are copied and where; fromTo[k\*2] is
a 0-based index of the input channel in src, fromTo[k\*2+1] is an index of the output channel in
dst; the continuous channel numbering is used: the first input image channels are indexed from 0 to
src[0].channels()-1, the second input image channels are indexed from src[0].channels() to
src[0].channels() + src[1].channels()-1, and so on, the same scheme is used for the output image
channels; as a special case, when fromTo[k\*2] is negative, the corresponding output channel is
filled with zero .
@param npairs number of index pairs in fromTo.
*/
CV_EXPORTS void mixChannels(InputArrayOfArrays src, InputOutputArrayOfArrays dst,
                            const int* fromTo, size_t npairs);

/** @overload
@param src input array or vector of matrices; all of the matrices must have the same size and the
same depth.
@param dst output array or vector of matrices; all the matrices **must be allocated**; their size and
depth must be the same as in src[0].
@param fromTo array of index pairs specifying which channels are copied and where; fromTo[k\*2] is
a 0-based index of the input channel in src, fromTo[k\*2+1] is an index of the output channel in
dst; the continuous channel numbering is used: the first input image channels are indexed from 0 to
src[0].channels()-1, the second input image channels are indexed from src[0].channels() to
src[0].channels() + src[1].channels()-1, and so on, the same scheme is used for the output image
channels; as a special case, when fromTo[k\*2] is negative, the corresponding output channel is
filled with zero .
*/
CV_EXPORTS_W void mixChannels(InputArrayOfArrays src, InputOutputArrayOfArrays dst,
                              const std::vector<int>& fromTo);

/** @brief Extracts a single channel from src (coi is 0-based index)
@param src input array
@param dst output array
@param coi index of channel to extract
@sa mixChannels, split
*/
CV_EXPORTS_W void extractChannel(InputArray src, OutputArray dst, int coi);

/** @brief Inserts a single channel to dst (coi is 0-based index)
@param src input array
@param dst output array
@param coi index of channel for insertion
@sa mixChannels, merge
*/
CV_EXPORTS_W void insertChannel(InputArray src, InputOutputArray dst, int coi);

/** @brief Flips a 2D array around vertical, horizontal, or both axes.

The function cv::flip flips the array in one of three different ways (row
and column indices are 0-based):
\f[\texttt{dst} _{ij} =
\left\{
\begin{array}{l l}
\texttt{src} _{\texttt{src.rows}-i-1,j} & if\;  \texttt{flipCode} = 0 \\
\texttt{src} _{i, \texttt{src.cols} -j-1} & if\;  \texttt{flipCode} > 0 \\
\texttt{src} _{ \texttt{src.rows} -i-1, \texttt{src.cols} -j-1} & if\; \texttt{flipCode} < 0 \\
\end{array}
\right.\f]
The example scenarios of using the function are the following:
*   Vertical flipping of the image (flipCode == 0) to switch between
    top-left and bottom-left image origin. This is a typical operation
    in video processing on Microsoft Windows\* OS.
*   Horizontal flipping of the image with the subsequent horizontal
    shift and absolute difference calculation to check for a
    vertical-axis symmetry (flipCode \> 0).
*   Simultaneous horizontal and vertical flipping of the image with
    the subsequent shift and absolute difference calculation to check
    for a central symmetry (flipCode \< 0).
*   Reversing the order of point arrays (flipCode \> 0 or
    flipCode == 0).
@param src input array.
@param dst output array of the same size and type as src.
@param flipCode a flag to specify how to flip the array; 0 means
flipping around the x-axis and positive value (for example, 1) means
flipping around y-axis. Negative value (for example, -1) means flipping
around both axes.
@sa transpose , repeat , completeSymm
*/
CV_EXPORTS_W void flip(InputArray src, OutputArray dst, int flipCode);

enum RotateFlags {
    ROTATE_90_CLOCKWISE = 0, //Rotate 90 degrees clockwise
    ROTATE_180 = 1, //Rotate 180 degrees clockwise
    ROTATE_90_COUNTERCLOCKWISE = 2, //Rotate 270 degrees clockwise
};
/** @brief Rotates a 2D array in multiples of 90 degrees.
The function rotate rotates the array in one of three different ways:
*   Rotate by 90 degrees clockwise (rotateCode = ROTATE_90).
*   Rotate by 180 degrees clockwise (rotateCode = ROTATE_180).
*   Rotate by 270 degrees clockwise (rotateCode = ROTATE_270).
@param src input array.
@param dst output array of the same type as src.  The size is the same with ROTATE_180,
and the rows and cols are switched for ROTATE_90 and ROTATE_270.
@param rotateCode an enum to specify how to rotate the array; see the enum RotateFlags
@sa transpose , repeat , completeSymm, flip, RotateFlags
*/
CV_EXPORTS_W void rotate(InputArray src, OutputArray dst, int rotateCode);

/** @brief Fills the output array with repeated copies of the input array.

The function cv::repeat duplicates the input array one or more times along each of the two axes:
\f[\texttt{dst} _{ij}= \texttt{src} _{i\mod src.rows, \; j\mod src.cols }\f]
The second variant of the function is more convenient to use with @ref MatrixExpressions.
@param src input array to replicate.
@param ny Flag to specify how many times the `src` is repeated along the
vertical axis.
@param nx Flag to specify how many times the `src` is repeated along the
horizontal axis.
@param dst output array of the same type as `src`.
@sa cv::reduce
*/
CV_EXPORTS_W void repeat(InputArray src, int ny, int nx, OutputArray dst);

/** @overload
@param src input array to replicate.
@param ny Flag to specify how many times the `src` is repeated along the
vertical axis.
@param nx Flag to specify how many times the `src` is repeated along the
horizontal axis.
  */
CV_EXPORTS Mat repeat(const Mat& src, int ny, int nx);

/** @brief Applies horizontal concatenation to given matrices.

The function horizontally concatenates two or more cv::Mat matrices (with the same number of rows).
@code{.cpp}
    cv::Mat matArray[] = { cv::Mat(4, 1, CV_8UC1, cv::Scalar(1)),
                           cv::Mat(4, 1, CV_8UC1, cv::Scalar(2)),
                           cv::Mat(4, 1, CV_8UC1, cv::Scalar(3)),};

    cv::Mat out;
    cv::hconcat( matArray, 3, out );
    //out:
    //[1, 2, 3;
    // 1, 2, 3;
    // 1, 2, 3;
    // 1, 2, 3]
@endcode
@param src input array or vector of matrices. all of the matrices must have the same number of rows and the same depth.
@param nsrc number of matrices in src.
@param dst output array. It has the same number of rows and depth as the src, and the sum of cols of the src.
@sa cv::vconcat(const Mat*, size_t, OutputArray), @sa cv::vconcat(InputArrayOfArrays, OutputArray) and @sa cv::vconcat(InputArray, InputArray, OutputArray)
*/
CV_EXPORTS void hconcat(const Mat* src, size_t nsrc, OutputArray dst);
/** @overload
 @code{.cpp}
    cv::Mat_<float> A = (cv::Mat_<float>(3, 2) << 1, 4,
                                                  2, 5,
                                                  3, 6);
    cv::Mat_<float> B = (cv::Mat_<float>(3, 2) << 7, 10,
                                                  8, 11,
                                                  9, 12);

    cv::Mat C;
    cv::hconcat(A, B, C);
    //C:
    //[1, 4, 7, 10;
    // 2, 5, 8, 11;
    // 3, 6, 9, 12]
 @endcode
 @param src1 first input array to be considered for horizontal concatenation.
 @param src2 second input array to be considered for horizontal concatenation.
 @param dst output array. It has the same number of rows and depth as the src1 and src2, and the sum of cols of the src1 and src2.
 */
CV_EXPORTS void hconcat(InputArray src1, InputArray src2, OutputArray dst);
/** @overload
 @code{.cpp}
    std::vector<cv::Mat> matrices = { cv::Mat(4, 1, CV_8UC1, cv::Scalar(1)),
                                      cv::Mat(4, 1, CV_8UC1, cv::Scalar(2)),
                                      cv::Mat(4, 1, CV_8UC1, cv::Scalar(3)),};

    cv::Mat out;
    cv::hconcat( matrices, out );
    //out:
    //[1, 2, 3;
    // 1, 2, 3;
    // 1, 2, 3;
    // 1, 2, 3]
 @endcode
 @param src input array or vector of matrices. all of the matrices must have the same number of rows and the same depth.
 @param dst output array. It has the same number of rows and depth as the src, and the sum of cols of the src.
same depth.
 */
CV_EXPORTS_W void hconcat(InputArrayOfArrays src, OutputArray dst);

/** @brief Applies vertical concatenation to given matrices.

The function vertically concatenates two or more cv::Mat matrices (with the same number of cols).
@code{.cpp}
    cv::Mat matArray[] = { cv::Mat(1, 4, CV_8UC1, cv::Scalar(1)),
                           cv::Mat(1, 4, CV_8UC1, cv::Scalar(2)),
                           cv::Mat(1, 4, CV_8UC1, cv::Scalar(3)),};

    cv::Mat out;
    cv::vconcat( matArray, 3, out );
    //out:
    //[1,   1,   1,   1;
    // 2,   2,   2,   2;
    // 3,   3,   3,   3]
@endcode
@param src input array or vector of matrices. all of the matrices must have the same number of cols and the same depth.
@param nsrc number of matrices in src.
@param dst output array. It has the same number of cols and depth as the src, and the sum of rows of the src.
@sa cv::hconcat(const Mat*, size_t, OutputArray), @sa cv::hconcat(InputArrayOfArrays, OutputArray) and @sa cv::hconcat(InputArray, InputArray, OutputArray)
*/
CV_EXPORTS void vconcat(const Mat* src, size_t nsrc, OutputArray dst);
/** @overload
 @code{.cpp}
    cv::Mat_<float> A = (cv::Mat_<float>(3, 2) << 1, 7,
                                                  2, 8,
                                                  3, 9);
    cv::Mat_<float> B = (cv::Mat_<float>(3, 2) << 4, 10,
                                                  5, 11,
                                                  6, 12);

    cv::Mat C;
    cv::vconcat(A, B, C);
    //C:
    //[1, 7;
    // 2, 8;
    // 3, 9;
    // 4, 10;
    // 5, 11;
    // 6, 12]
 @endcode
 @param src1 first input array to be considered for vertical concatenation.
 @param src2 second input array to be considered for vertical concatenation.
 @param dst output array. It has the same number of cols and depth as the src1 and src2, and the sum of rows of the src1 and src2.
 */
CV_EXPORTS void vconcat(InputArray src1, InputArray src2, OutputArray dst);
/** @overload
 @code{.cpp}
    std::vector<cv::Mat> matrices = { cv::Mat(1, 4, CV_8UC1, cv::Scalar(1)),
                                      cv::Mat(1, 4, CV_8UC1, cv::Scalar(2)),
                                      cv::Mat(1, 4, CV_8UC1, cv::Scalar(3)),};

    cv::Mat out;
    cv::vconcat( matrices, out );
    //out:
    //[1,   1,   1,   1;
    // 2,   2,   2,   2;
    // 3,   3,   3,   3]
 @endcode
 @param src input array or vector of matrices. all of the matrices must have the same number of cols and the same depth
 @param dst output array. It has the same number of cols and depth as the src, and the sum of rows of the src.
same depth.
 */
CV_EXPORTS_W void vconcat(InputArrayOfArrays src, OutputArray dst);

/** @brief computes bitwise conjunction of the two arrays (dst = src1 & src2)
Calculates the per-element bit-wise conjunction of two arrays or an
array and a scalar.

The function cv::bitwise_and calculates the per-element bit-wise logical conjunction for:
*   Two arrays when src1 and src2 have the same size:
    \f[\texttt{dst} (I) =  \texttt{src1} (I)  \wedge \texttt{src2} (I) \quad \texttt{if mask} (I) \ne0\f]
*   An array and a scalar when src2 is constructed from Scalar or has
    the same number of elements as `src1.channels()`:
    \f[\texttt{dst} (I) =  \texttt{src1} (I)  \wedge \texttt{src2} \quad \texttt{if mask} (I) \ne0\f]
*   A scalar and an array when src1 is constructed from Scalar or has
    the same number of elements as `src2.channels()`:
    \f[\texttt{dst} (I) =  \texttt{src1}  \wedge \texttt{src2} (I) \quad \texttt{if mask} (I) \ne0\f]
In case of floating-point arrays, their machine-specific bit
representations (usually IEEE754-compliant) are used for the operation.
In case of multi-channel arrays, each channel is processed
independently. In the second and third cases above, the scalar is first
converted to the array type.
@param src1 first input array or a scalar.
@param src2 second input array or a scalar.
@param dst output array that has the same size and type as the input
arrays.
@param mask optional operation mask, 8-bit single channel array, that
specifies elements of the output array to be changed.
*/
CV_EXPORTS_W void bitwise_and(InputArray src1, InputArray src2,
                              OutputArray dst, InputArray mask = noArray());

/** @brief Calculates the per-element bit-wise disjunction of two arrays or an
array and a scalar.

The function cv::bitwise_or calculates the per-element bit-wise logical disjunction for:
*   Two arrays when src1 and src2 have the same size:
    \f[\texttt{dst} (I) =  \texttt{src1} (I)  \vee \texttt{src2} (I) \quad \texttt{if mask} (I) \ne0\f]
*   An array and a scalar when src2 is constructed from Scalar or has
    the same number of elements as `src1.channels()`:
    \f[\texttt{dst} (I) =  \texttt{src1} (I)  \vee \texttt{src2} \quad \texttt{if mask} (I) \ne0\f]
*   A scalar and an array when src1 is constructed from Scalar or has
    the same number of elements as `src2.channels()`:
    \f[\texttt{dst} (I) =  \texttt{src1}  \vee \texttt{src2} (I) \quad \texttt{if mask} (I) \ne0\f]
In case of floating-point arrays, their machine-specific bit
representations (usually IEEE754-compliant) are used for the operation.
In case of multi-channel arrays, each channel is processed
independently. In the second and third cases above, the scalar is first
converted to the array type.
@param src1 first input array or a scalar.
@param src2 second input array or a scalar.
@param dst output array that has the same size and type as the input
arrays.
@param mask optional operation mask, 8-bit single channel array, that
specifies elements of the output array to be changed.
*/
CV_EXPORTS_W void bitwise_or(InputArray src1, InputArray src2,
                             OutputArray dst, InputArray mask = noArray());

/** @brief Calculates the per-element bit-wise "exclusive or" operation on two
arrays or an array and a scalar.

The function cv::bitwise_xor calculates the per-element bit-wise logical "exclusive-or"
operation for:
*   Two arrays when src1 and src2 have the same size:
    \f[\texttt{dst} (I) =  \texttt{src1} (I)  \oplus \texttt{src2} (I) \quad \texttt{if mask} (I) \ne0\f]
*   An array and a scalar when src2 is constructed from Scalar or has
    the same number of elements as `src1.channels()`:
    \f[\texttt{dst} (I) =  \texttt{src1} (I)  \oplus \texttt{src2} \quad \texttt{if mask} (I) \ne0\f]
*   A scalar and an array when src1 is constructed from Scalar or has
    the same number of elements as `src2.channels()`:
    \f[\texttt{dst} (I) =  \texttt{src1}  \oplus \texttt{src2} (I) \quad \texttt{if mask} (I) \ne0\f]
In case of floating-point arrays, their machine-specific bit
representations (usually IEEE754-compliant) are used for the operation.
In case of multi-channel arrays, each channel is processed
independently. In the 2nd and 3rd cases above, the scalar is first
converted to the array type.
@param src1 first input array or a scalar.
@param src2 second input array or a scalar.
@param dst output array that has the same size and type as the input
arrays.
@param mask optional operation mask, 8-bit single channel array, that
specifies elements of the output array to be changed.
*/
CV_EXPORTS_W void bitwise_xor(InputArray src1, InputArray src2,
                              OutputArray dst, InputArray mask = noArray());

/** @brief  Inverts every bit of an array.

The function cv::bitwise_not calculates per-element bit-wise inversion of the input
array:
\f[\texttt{dst} (I) =  \neg \texttt{src} (I)\f]
In case of a floating-point input array, its machine-specific bit
representation (usually IEEE754-compliant) is used for the operation. In
case of multi-channel arrays, each channel is processed independently.
@param src input array.
@param dst output array that has the same size and type as the input
array.
@param mask optional operation mask, 8-bit single channel array, that
specifies elements of the output array to be changed.
*/
CV_EXPORTS_W void bitwise_not(InputArray src, OutputArray dst,
                              InputArray mask = noArray());

/** @brief Calculates the per-element absolute difference between two arrays or between an array and a scalar.

The function cv::absdiff calculates:
*   Absolute difference between two arrays when they have the same
    size and type:
    \f[\texttt{dst}(I) =  \texttt{saturate} (| \texttt{src1}(I) -  \texttt{src2}(I)|)\f]
*   Absolute difference between an array and a scalar when the second
    array is constructed from Scalar or has as many elements as the
    number of channels in `src1`:
    \f[\texttt{dst}(I) =  \texttt{saturate} (| \texttt{src1}(I) -  \texttt{src2} |)\f]
*   Absolute difference between a scalar and an array when the first
    array is constructed from Scalar or has as many elements as the
    number of channels in `src2`:
    \f[\texttt{dst}(I) =  \texttt{saturate} (| \texttt{src1} -  \texttt{src2}(I) |)\f]
    where I is a multi-dimensional index of array elements. In case of
    multi-channel arrays, each channel is processed independently.
@note Saturation is not applied when the arrays have the depth CV_32S.
You may even get a negative value in the case of overflow.
@param src1 first input array or a scalar.
@param src2 second input array or a scalar.
@param dst output array that has the same size and type as input arrays.
@sa cv::abs(const Mat&)
*/
CV_EXPORTS_W void absdiff(InputArray src1, InputArray src2, OutputArray dst);

/** @brief  Checks if array elements lie between the elements of two other arrays.

The function checks the range as follows:
-   For every element of a single-channel input array:
    \f[\texttt{dst} (I)= \texttt{lowerb} (I)_0  \leq \texttt{src} (I)_0 \leq  \texttt{upperb} (I)_0\f]
-   For two-channel arrays:
    \f[\texttt{dst} (I)= \texttt{lowerb} (I)_0  \leq \texttt{src} (I)_0 \leq  \texttt{upperb} (I)_0  \land \texttt{lowerb} (I)_1  \leq \texttt{src} (I)_1 \leq  \texttt{upperb} (I)_1\f]
-   and so forth.

That is, dst (I) is set to 255 (all 1 -bits) if src (I) is within the
specified 1D, 2D, 3D, ... box and 0 otherwise.

When the lower and/or upper boundary parameters are scalars, the indexes
(I) at lowerb and upperb in the above formulas should be omitted.
@param src first input array.
@param lowerb inclusive lower boundary array or a scalar.
@param upperb inclusive upper boundary array or a scalar.
@param dst output array of the same size as src and CV_8U type.
*/
CV_EXPORTS_W void inRange(InputArray src, InputArray lowerb,
                          InputArray upperb, OutputArray dst);

/** @brief Performs the per-element comparison of two arrays or an array and scalar value.

The function compares:
*   Elements of two arrays when src1 and src2 have the same size:
    \f[\texttt{dst} (I) =  \texttt{src1} (I)  \,\texttt{cmpop}\, \texttt{src2} (I)\f]
*   Elements of src1 with a scalar src2 when src2 is constructed from
    Scalar or has a single element:
    \f[\texttt{dst} (I) =  \texttt{src1}(I) \,\texttt{cmpop}\,  \texttt{src2}\f]
*   src1 with elements of src2 when src1 is constructed from Scalar or
    has a single element:
    \f[\texttt{dst} (I) =  \texttt{src1}  \,\texttt{cmpop}\, \texttt{src2} (I)\f]
When the comparison result is true, the corresponding element of output
array is set to 255. The comparison operations can be replaced with the
equivalent matrix expressions:
@code{.cpp}
    Mat dst1 = src1 >= src2;
    Mat dst2 = src1 < 8;
    ...
@endcode
@param src1 first input array or a scalar; when it is an array, it must have a single channel.
@param src2 second input array or a scalar; when it is an array, it must have a single channel.
@param dst output array of type ref CV_8U that has the same size and the same number of channels as
    the input arrays.
@param cmpop a flag, that specifies correspondence between the arrays (cv::CmpTypes)
@sa checkRange, min, max, threshold
*/
CV_EXPORTS_W void compare(InputArray src1, InputArray src2, OutputArray dst, int cmpop);

/** @brief Calculates per-element minimum of two arrays or an array and a scalar.

The function cv::min calculates the per-element minimum of two arrays:
\f[\texttt{dst} (I)= \min ( \texttt{src1} (I), \texttt{src2} (I))\f]
or array and a scalar:
\f[\texttt{dst} (I)= \min ( \texttt{src1} (I), \texttt{value} )\f]
@param src1 first input array.
@param src2 second input array of the same size and type as src1.
@param dst output array of the same size and type as src1.
@sa max, compare, inRange, minMaxLoc
*/
CV_EXPORTS_W void min(InputArray src1, InputArray src2, OutputArray dst);
/** @overload
needed to avoid conflicts with const _Tp& std::min(const _Tp&, const _Tp&, _Compare)
*/
CV_EXPORTS void min(const Mat& src1, const Mat& src2, Mat& dst);
/** @overload
needed to avoid conflicts with const _Tp& std::min(const _Tp&, const _Tp&, _Compare)
*/
CV_EXPORTS void min(const UMat& src1, const UMat& src2, UMat& dst);

/** @brief Calculates per-element maximum of two arrays or an array and a scalar.

The function cv::max calculates the per-element maximum of two arrays:
\f[\texttt{dst} (I)= \max ( \texttt{src1} (I), \texttt{src2} (I))\f]
or array and a scalar:
\f[\texttt{dst} (I)= \max ( \texttt{src1} (I), \texttt{value} )\f]
@param src1 first input array.
@param src2 second input array of the same size and type as src1 .
@param dst output array of the same size and type as src1.
@sa  min, compare, inRange, minMaxLoc, @ref MatrixExpressions
*/
CV_EXPORTS_W void max(InputArray src1, InputArray src2, OutputArray dst);
/** @overload
needed to avoid conflicts with const _Tp& std::min(const _Tp&, const _Tp&, _Compare)
*/
CV_EXPORTS void max(const Mat& src1, const Mat& src2, Mat& dst);
/** @overload
needed to avoid conflicts with const _Tp& std::min(const _Tp&, const _Tp&, _Compare)
*/
CV_EXPORTS void max(const UMat& src1, const UMat& src2, UMat& dst);

/** @brief Calculates a square root of array elements.

The function cv::sqrt calculates a square root of each input array element.
In case of multi-channel arrays, each channel is processed
independently. The accuracy is approximately the same as of the built-in
std::sqrt .
@param src input floating-point array.
@param dst output array of the same size and type as src.
*/
CV_EXPORTS_W void sqrt(InputArray src, OutputArray dst);

/** @brief Raises every array element to a power.

The function cv::pow raises every element of the input array to power :
\f[\texttt{dst} (I) =  \fork{\texttt{src}(I)^{power}}{if \(\texttt{power}\) is integer}{|\texttt{src}(I)|^{power}}{otherwise}\f]

So, for a non-integer power exponent, the absolute values of input array
elements are used. However, it is possible to get true values for
negative values using some extra operations. In the example below,
computing the 5th root of array src shows:
@code{.cpp}
    Mat mask = src < 0;
    pow(src, 1./5, dst);
    subtract(Scalar::all(0), dst, dst, mask);
@endcode
For some values of power, such as integer values, 0.5 and -0.5,
specialized faster algorithms are used.

Special values (NaN, Inf) are not handled.
@param src input array.
@param power exponent of power.
@param dst output array of the same size and type as src.
@sa sqrt, exp, log, cartToPolar, polarToCart
*/
CV_EXPORTS_W void pow(InputArray src, double power, OutputArray dst);

/** @brief Calculates the exponent of every array element.

The function cv::exp calculates the exponent of every element of the input
array:
\f[\texttt{dst} [I] = e^{ src(I) }\f]

The maximum relative error is about 7e-6 for single-precision input and
less than 1e-10 for double-precision input. Currently, the function
converts denormalized values to zeros on output. Special values (NaN,
Inf) are not handled.
@param src input array.
@param dst output array of the same size and type as src.
@sa log , cartToPolar , polarToCart , phase , pow , sqrt , magnitude
*/
CV_EXPORTS_W void exp(InputArray src, OutputArray dst);

/** @brief Calculates the natural logarithm of every array element.

The function cv::log calculates the natural logarithm of every element of the input array:
\f[\texttt{dst} (I) =  \log (\texttt{src}(I)) \f]

Output on zero, negative and special (NaN, Inf) values is undefined.

@param src input array.
@param dst output array of the same size and type as src .
@sa exp, cartToPolar, polarToCart, phase, pow, sqrt, magnitude
*/
CV_EXPORTS_W void log(InputArray src, OutputArray dst);

/** @brief Calculates x and y coordinates of 2D vectors from their magnitude and angle.

The function cv::polarToCart calculates the Cartesian coordinates of each 2D
vector represented by the corresponding elements of magnitude and angle:
\f[\begin{array}{l} \texttt{x} (I) =  \texttt{magnitude} (I) \cos ( \texttt{angle} (I)) \\ \texttt{y} (I) =  \texttt{magnitude} (I) \sin ( \texttt{angle} (I)) \\ \end{array}\f]

The relative accuracy of the estimated coordinates is about 1e-6.
@param magnitude input floating-point array of magnitudes of 2D vectors;
it can be an empty matrix (=Mat()), in this case, the function assumes
that all the magnitudes are =1; if it is not empty, it must have the
same size and type as angle.
@param angle input floating-point array of angles of 2D vectors.
@param x output array of x-coordinates of 2D vectors; it has the same
size and type as angle.
@param y output array of y-coordinates of 2D vectors; it has the same
size and type as angle.
@param angleInDegrees when true, the input angles are measured in
degrees, otherwise, they are measured in radians.
@sa cartToPolar, magnitude, phase, exp, log, pow, sqrt
*/
CV_EXPORTS_W void polarToCart(InputArray magnitude, InputArray angle,
                              OutputArray x, OutputArray y, bool angleInDegrees = false);

/** @brief Calculates the magnitude and angle of 2D vectors.

The function cv::cartToPolar calculates either the magnitude, angle, or both
for every 2D vector (x(I),y(I)):
\f[\begin{array}{l} \texttt{magnitude} (I)= \sqrt{\texttt{x}(I)^2+\texttt{y}(I)^2}