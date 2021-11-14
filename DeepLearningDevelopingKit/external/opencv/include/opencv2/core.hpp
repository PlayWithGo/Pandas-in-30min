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
        \f[\t