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

#ifndef OPENCV_CORE_MAT_HPP
#define OPENCV_CORE_MAT_HPP

#ifndef __cplusplus
#  error mat.hpp header must be compiled as C++
#endif

#include "opencv2/core/matx.hpp"
#include "opencv2/core/types.hpp"

#include "opencv2/core/bufferpool.hpp"

#ifdef CV_CXX11
#include <type_traits>
#endif

namespace cv
{

//! @addtogroup core_basic
//! @{

enum { ACCESS_READ=1<<24, ACCESS_WRITE=1<<25,
    ACCESS_RW=3<<24, ACCESS_MASK=ACCESS_RW, ACCESS_FAST=1<<26 };

CV__DEBUG_NS_BEGIN

class CV_EXPORTS _OutputArray;

//////////////////////// Input/Output Array Arguments /////////////////////////////////

/** @brief This is the proxy class for passing read-only input arrays into OpenCV functions.

It is defined as:
@code
    typedef const _InputArray& InputArray;
@endcode
where _InputArray is a class that can be constructed from `Mat`, `Mat_<T>`, `Matx<T, m, n>`,
`std::vector<T>`, `std::vector<std::vector<T> >`, `std::vector<Mat>`, `std::vector<Mat_<T> >`,
`UMat`, `std::vector<UMat>` or `double`. It can also be constructed from a matrix expression.

Since this is mostly implementation-level class, and its interface may change in future versions, we
do not describe it in details. There are a few key things, though, that should be kept in mind:

-   When you see in the reference manual or in OpenCV source code a function that takes
    InputArray, it means that you can actually pass `Mat`, `Matx`, `vector<T>` etc. (see above the
    complete list).
-   Optional input arguments: If some of the input arrays may be empty, pass cv::noArray() (or
    simply cv::Mat() as you probably did before).
-   The class is designed solely for passing parameters. That is, normally you *should not*
    declare class members, local and global variables of this type.
-   If you want to design your own function or a class method that can operate of arrays of
    multiple types, you can use InputArray (or OutputArray) for the respective parameters. Inside
    a function you should use _InputArray::getMat() method to construct a matrix header for the
    array (without copying data). _InputArray::kind() can be used to distinguish Mat from
    `vector<>` etc., but normally it is not needed.

Here is how you can use a function that takes InputArray :
@code
    std::vector<Point2f> vec;
    // points or a circle
    for( int i = 0; i < 30; i++ )
        vec.push_back(Point2f((float)(100 + 30*cos(i*CV_PI*2/5)),
                              (float)(100 - 30*sin(i*CV_PI*2/5))));
    cv::transform(vec, vec, cv::Matx23f(0.707, -0.707, 10, 0.707, 0.707, 20));
@endcode
That is, we form an STL vector containing points, and apply in-place affine transformation to the
vector using the 2x3 matrix created inline as `Matx<float, 2, 3>` instance.

Here is how such a function can be implemented (for simplicity, we implement a very specific case of
it, according to the assertion statement inside) :
@code
    void myAffineTransform(InputArray _src, OutputArray _dst, InputArray _m)
    {
        // get Mat headers for input arrays. This is O(1) operation,
        // unless _src and/or _m are matrix expressions.
        Mat src = _src.getMat(), m = _m.getMat();
        CV_Assert( src.type() == CV_32FC2 && m.type() == CV_32F && m.size() == Size(3, 2) );

        // [re]create the output array so that it has the proper size and type.
        // In case of Mat it calls Mat::create, in case of STL vector it calls vector::resize.
        _dst.create(src.size(), src.type());
        Mat dst = _dst.getMat();

        for( int i = 0; i < src.rows; i++ )
            for( int j = 0; j < src.cols; j++ )
            {
                Point2f pt = src.at<Point2f>(i, j);
                dst.at<Point2f>(i, j) = Point2f(m.at<float>(0, 0)*pt.x +
                                                m.at<float>(0, 1)*pt.y +
                                                m.at<float>(0, 2),
                                                m.at<float>(1, 0)*pt.x +
                                                m.at<float>(1, 1)*pt.y +
                                                m.at<float>(1, 2));
            }
    }
@endcode
There is another related type, InputArrayOfArrays, which is currently defined as a synonym for
InputArray:
@code
    typedef InputArray InputArrayOfArrays;
@endcode
It denotes function arguments that are either vectors of vectors or vectors of matrices. A separate
synonym is needed to generate Python/Java etc. wrappers properly. At the function implementation
level their use is similar, but _InputArray::getMat(idx) should be used to get header for the
idx-th component of the outer vector and _InputArray::size().area() should be used to find the
number of components (vectors/matrices) of the outer vector.
 */
class CV_EXPORTS _InputArray
{
public:
    enum {
        KIND_SHIFT = 16,
        FIXED_TYPE = 0x8000 << KIND_SHIFT,
        FIXED_SIZE = 0x4000 << KIND_SHIFT,
        KIND_MASK = 31 << KIND_SHIFT,

        NONE              = 0 << KIND_SHIFT,
        MAT               = 1 << KIND_SHIFT,
        MATX              = 2 << KIND_SHIFT,
        STD_VECTOR        = 3 << KIND_SHIFT,
        STD_VECTOR_VECTOR = 4 << KIND_SHIFT,
        STD_VECTOR_MAT    = 5 << KIND_SHIFT,
        EXPR              = 6 << KIND_SHIFT,
        OPENGL_BUFFER     = 7 << KIND_SHIFT,
        CUDA_HOST_MEM     = 8 << KIND_SHIFT,
        CUDA_GPU_MAT      = 9 << KIND_SHIFT,
        UMAT              =10 << KIND_SHIFT,
        STD_VECTOR_UMAT   =11 << KIND_SHIFT,
        STD_BOOL_VECTOR   =12 << KIND_SHIFT,
        STD_VECTOR_CUDA_GPU_MAT = 13 << KIND_SHIFT,
        STD_ARRAY         =14 << KIND_SHIFT,
        STD_ARR