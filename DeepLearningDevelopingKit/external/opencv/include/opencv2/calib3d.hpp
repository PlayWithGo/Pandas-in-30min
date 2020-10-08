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

#ifndef OPENCV_CALIB3D_HPP
#define OPENCV_CALIB3D_HPP

#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/core/affine.hpp"

/**
  @defgroup calib3d Camera Calibration and 3D Reconstruction

The functions in this section use a so-called pinhole camera model. In this model, a scene view is
formed by projecting 3D points into the image plane using a perspective transformation.

\f[s  \; m' = A [R|t] M'\f]

or

\f[s  \vecthree{u}{v}{1} = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}
\begin{bmatrix}
r_{11} & r_{12} & r_{13} & t_1  \\
r_{21} & r_{22} & r_{23} & t_2  \\
r_{31} & r_{32} & r_{33} & t_3
\end{bmatrix}
\begin{bmatrix}
X \\
Y \\
Z \\
1
\end{bmatrix}\f]

where:

-   \f$(X, Y, Z)\f$ are the coordinates of a 3D point in the world coordinate space
-   \f$(u, v)\f$ are the coordinates of the projection point in pixels
-   \f$A\f$ is a camera matrix, or a matrix of intrinsic parameters
-   \f$(cx, cy)\f$ is a principal point that is usually at the image center
-   \f$fx, fy\f$ are the focal lengths expressed in pixel units.

Thus, if an image from the camera is scaled by a factor, all of these parameters should be scaled
(multiplied/divided, respectively) by the same factor. The matrix of intrinsic parameters does not
depend on the scene viewed. So, once estimated, it can be re-used as long as the focal length is
fixed (in case of zoom lens). The joint rotation-translation matrix \f$[R|t]\f$ is called a matrix of
extrinsic parameters. It is used to describe the camera motion around a static scene, or vice versa,
rigid motion of an object in front of a still camera. That is, \f$[R|t]\f$ translates coordinates of a
point \f$(X, Y, Z)\f$ to a coordinate system, fixed with respect to the camera. The transformation above
is equivalent to the following (when \f$z \ne 0\f$ ):

\f[\begin{array}{l}
\vecthree{x}{y}{z} = R  \vecthree{X}{Y}{Z} + t \\
x' = x/z \\
y' = y/z \\
u = f_x*x' + c_x \\
v = f_y*y' + c_y
\end{array}\f]

The following figure illustrates the pinhole camera model.

![Pinhole camera model](pics/pinhole_camera_model.png)

Real lenses usually have some distortion, mostly radial distortion and slight tangential distortion.
So, the above model is extended as:

\f[\begin{array}{l}
\vecthree{x}{y}{z} = R  \vecthree{X}{Y}{Z} + t \\
x' = x/z \\
y' = y/z \\
x'' = x'  \frac{1 + k_1 r^2 + k_2 r^4 + k_3 r^6}{1 + k_4 r^2 + k_5 r^4 + k_6 r^6} + 2 p_1 x' y' + p_2(r^2 + 2 x'^2) + s_1 r^2 + s_2 r^4 \\
y'' = y'  \frac{1 + k_1 r^2 + k_2 r^4 + k_3 r^6}{1 + k_4 r^2 + k_5 r^4 + k_6 r^6} + p_1 (r^2 + 2 y'^2) + 2 p_2 x' y' + s_3 r^2 + s_4 r^4 \\
\text{where} \quad r^2 = x'^2 + y'^2  \\
u = f_x*x'' + c_x \\
v = f_y*y'' + c_y
\end{array}\f]

\f$k_1\f$, \f$k_2\f$, \f$k_3\f$, \f$k_4\f$, \f$k_5\f$, and \f$k_6\f$ are radial distortion coefficients. \f$p_1\f$ and \f$p_2\f$ are
tangential distortion coefficients. \f$s_1\f$, \f$s_2\f$, \f$s_3\f$, and \f$s_4\f$, are the thin prism distortion
coefficients. Higher-order coefficients are not considered in OpenCV.

The next figure shows two common types of radial distortion: barrel distortion (typically \f$ k_1 > 0 \f$ and pincushion distortion (typically \f$ k_1 < 0 \f$).

![](pics/distortion_examples.png)

In some cases the image sensor may be tilted in order to focus an oblique plane in front of the
camera (Scheimpfug condition). This can be useful for particle image velocimetry (PIV) or
triangulation with a laser fan. The tilt causes a perspective distortion of \f$x''\f$ and
\f$y''\f$. This distortion can be modelled in the following way, see e.g. @cite Louhichi07.

\f[\begin{array}{l}
s\vecthree{x'''}{y'''}{1} =
\vecthreethree{R_{33}(\tau_x, \tau_y)}{0}{-R_{13}(\tau_x, \tau_y)}
{0}{R_{33}(\tau_x, \tau_y)}{-R_{23}(\tau_x, \tau_y)}
{0}{0}{1} R(\tau_x, \tau_y) \vecthree{x''}{y''}{1}\\
u = f_x*x''' + c_x \\
v = f_y*y''' + c_y
\end{array}\f]

where the matrix \f$R(\tau_x, \tau_y)\f$ is defined by two rotations with angular parameter \f$\tau_x\f$
and \f$\tau_y\f$, respectively,

\f[
R(\tau_x, \tau_y) =
\vecthreethree{\cos(\tau_y)}{0}{-\sin(\tau_y)}{0}{1}{0}{\sin(\tau_y)}{0}{\cos(\tau_y)}
\vecthreethree{1}{0}{0}{0}{\cos(\tau_x)}{\sin(\tau_x)}{0}{-\sin(\tau_x)}{\cos(\tau_x)} =
\vecthreethree{\cos(\tau_y)}{\sin(\tau_y)\sin(\tau_x)}{-\sin(\tau_y)\cos(\tau_x)}
{0}{\cos(\tau_x)}{\sin(\tau_x)}
{\sin(\tau_y)}{-\cos(\tau_y)\sin(\tau_x)}{\cos(\tau_y)\cos(\tau_x)}.
\f]

In the functions below the coefficients are passed or returned as

\f[(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6 [, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\f]

vector. That is, if the vector contains four elements, it means that \f$k_3=0\f$ . The distortion
coefficients do not depend on the scene viewed. Thus, they also belong to the intrinsic camera
parameters. And they remain the same regardless of the captured image resolution. If, for example, a
camera has been calibrated on images of 320 x 240 resolution, absolutely the same distortion
coefficients can be used for 640 x 480 images from the same camera while \f$f_x\f$, \f$f_y\f$, \f$c_x\f$, and
\f$c_y\f$ need to be scaled appropriately.

The functions below use the above model to do the following:

-   Project 3D points to the image plane given intrinsic and extrinsic parameters.
-   Compute extrinsic parameters given intrinsic parameters, a few 3D points, and their
projections.
-   Estimate intrinsic and extrinsic camera parameters from several views of a known calibration
pattern (every view is described by several 3D-2D point correspondences).
-   Estimate the relative position and orientation of the stereo camera "heads" and compute the
*rectification* transformation that makes the camera optical axes parallel.

@note
   -   A calibration sample for 3 cameras in horizontal position can be found at
        opencv_source_code/samples/cpp/3calibration.cpp
    -   A calibration sample based on a sequence of images can be found at
        opencv_source_code/samples/cpp/calibration.cpp
    -   A calibration sample in order to do 3D reconstruction can be found at
        opencv_source_code/samples/cpp/build3dmodel.cpp
    -   A calibration sample of an artificially generated camera and chessboard patterns can be
        found at opencv_source_code/samples/cpp/calibration_artificial.cpp
    -   A calibration example on stereo calibration can be found at
        opencv_source_code/samples/cpp/stereo_calib.cpp
    -   A calibration example on stereo matching can be found at
        opencv_source_code/samples/cpp/stereo_match.cpp
    -   (Python) A camera calibration sample can be found at
        opencv_source_code/samples/python/calibrate.py

  @{
    @defgroup calib3d_fisheye Fisheye camera model

    Definitions: Let P be a point in 3D of coordinates X in the world reference frame (stored in the
    matrix X) The coordinate vector of P in the camera reference frame is:

    \f[Xc = R X + T\f]

    where R is the rotation matrix corresponding to the rotation vector om: R = rodrigues(om); call x, y
    and z the 3 coordinates of Xc:

    \f[x = Xc_1 \\ y = Xc_2 \\ z = Xc_3\f]

    The pinhole projection coordinates of P is [a; b] where

    \f[a = x / z \ and \ b = y / z \\ r^2 = a^2 + b^2 \\ \theta = atan(r)\f]

    Fisheye distortion:

    \f[\theta_d = \theta (1 + k_1 \theta^2 + k_2 \theta^4 + k_3 \theta^6 + k_4 \theta^8)\f]

    The distorted point coordinates are [x'; y'] where

    \f[x' = (\theta_d / r) a \\ y' = (\theta_d / r) b \f]

    Finally, conversion into pixel coordinates: The final pixel coordinates vector [u; v] where:

    \f[u = f_x (x' + \alpha y') + c_x \\
    v = f_y y' + c_y\f]

    @defgroup calib3d_c C API

  @}
 */

namespace cv
{

//! @addtogroup calib3d
//! @{

//! type of the robust estimation algorithm
enum { LMEDS  = 4, //!< least-median of squares algorithm
       RANSAC = 8, //!< RANSAC algorithm
       RHO    = 16 //!< RHO algorithm
     };

enum { SOLVEPNP_ITERATIVE = 0,
       SOLVEPNP_EPNP      = 1, //!< EPnP: Efficient Perspective-n-Point Camera Pose Estimation @cite lepetit2009epnp
       SOLVEPNP_P3P       = 2, //!< Complete Solution Classification for the Perspective-Three-Point Problem @cite gao2003complete
       SOLVEPNP_DLS       = 3, //!< A Direct Least-Squares (DLS) Method for PnP  @cite hesch2011direct
       SOLVEPNP_UPNP      = 4, //!< Exhaustive Linearization for Robust Camera Pose and Focal Length Estimation @cite penate2013exhaustive
       SOLVEPNP_AP3P      = 5, //!< An Efficient Algebraic Solution to the Perspective-Three-Point Problem @cite Ke17
       SOLVEPNP_MAX_COUNT      //!< Used for count
};

enum { CALIB_CB_ADAPTIVE_THRESH = 1,
       CALIB_CB_NORMALIZE_IMAGE = 2,
       CALIB_CB_FILTER_QUADS    = 4,
       CALIB_CB_FAST_CHECK      = 8
     };

enum { CALIB_CB_SYMMETRIC_GRID  = 1,
       CALIB_CB_ASYMMETRIC_GRID = 2,
       CALIB_CB_CLUSTERING      = 4
     };

enum { CALIB_USE_INTRINSIC_GUESS = 0x00001,
       CALIB_FIX_ASPECT_RATIO    = 0x00002,
       CALIB_FIX_PRINCIPAL_POINT = 0x00004,
       CALIB_ZERO_TANGENT_DIST   = 0x00008,
       CALIB_FIX_FOCAL_LENGTH    = 0x00010,
       CALIB_FIX_K1              = 0x00020,
       CALIB_FIX_K2              = 0x00040,
       CALIB_FIX_K3              = 0x00080,
       CALIB_FIX_K4              = 0x00800,
       CALIB_FIX_K5              = 0x01000,
       CALIB_FIX_K6              = 0x02000,
       CALIB_RATIONAL_MODEL      = 0x04000,
       CALIB_THIN_PRISM_MODEL    = 0x08000,
       CALIB_FIX_S1_S2_S3_S4     = 0x10000,
       CALIB_TILTED_MODEL        = 0x40000,
       CALIB_FIX_TAUX_TAUY       = 0x80000,
       CALIB_USE_QR              = 0x100000, //!< use QR instead of SVD decomposition for solving. Faster but potentially less precise
       CALIB_FIX_TANGENT_DIST    = 0x200000,
       // only for stereo
       CALIB_FIX_INTRINSIC       = 0x00100,
       CALIB_SAME_FOCAL_LENGTH   = 0x00200,
       // for stereo rectification
       CALIB_ZERO_DISPARITY      = 0x00400,
       CALIB_USE_LU              = (1 << 17), //!< use LU instead of SVD decomposition for solving. much faster but potentially less precise
       CALIB_USE_EXTRINSIC_GUESS = (1 << 22), //!< for stereoCalibrate
     };

//! the algorithm for finding fundamental matrix
enum { FM_7POINT = 1, //!< 7-point algorithm
       FM_8POINT = 2, //!< 8-point algorithm
       FM_LMEDS  = 4, //!< least-median algorithm. 7-point algorithm is used.
       FM_RANSAC = 8  //!< RANSAC algorithm. It needs at least 15 points. 7-point algorithm is used.
     };



/** @brief Converts a rotation matrix to a rotation vector or vice versa.

@param src Input rotation vector (3x1 or 1x3) or rotation matrix (3x3).
@param dst Output rotation matrix (3x3) or rotation vector (3x1 or 1x3), respectively.
@param jacobian Optional output Jacobian matrix, 3x9 or 9x3, which is a matrix of partial
derivatives of the output array components with respect to the input array components.

\f[\begin{array}{l} \theta \leftarrow norm(r) \\ r  \leftarrow r/ \theta \\ R =  \cos{\theta} I + (1- \cos{\theta} ) r r^T +  \sin{\theta} \vecthreethree{0}{-r_z}{r_y}{r_z}{0}{-r_x}{-r_y}{r_x}{0} \end{array}\f]

Inverse transformation can be also done easily, since

\f[\sin ( \theta ) \vecthreethree{0}{-r_z}{r_y}{r_z}{0}{-r_x}{-r_y}{r_x}{0} = \frac{R - R^T}{2}\f]

A rotation vector is a convenient and most compact representation of a rotation matrix (since any
rotation matrix has just 3 degrees of freedom). The representation is used in the global 3D geometry
optimization procedures like calibrateCamera, stereoCalibrate, or solvePnP .
 */
CV_EXPORTS_W void Rodrigues( InputArray src, OutputArray dst, OutputArray jacobian = noArray() );

/** @example pose_from_homography.cpp
  An example program about pose estimation from coplanar points

  Check @ref tutorial_homography "the corresponding tutorial" for more details
 */

/** @brief Finds a perspective transformation between two planes.

@param srcPoints Coordinates of the points in the original plane, a matrix of the type CV_32FC2
or vector\<Point2f\> .
@param dstPoints Coordinates of the points in the target plane, a matrix of the type CV_32FC2 or
a vector\<Point2f\> .
@param method Method used to compute a homography matrix. The following methods are possible:
-   **0** - a regular method using all the points, i.e., the least squares method
-   **RANSAC** - RANSAC-based robust method
-   **LMEDS** - Least-Median robust method
-   **RHO** - PROSAC-based robust method
@param ransacReprojThreshold Maximum allowed reprojection error to treat a point pair as an inlier
(used in the RANSAC and RHO methods only). That is, if
\f[\| \texttt{dstPoints} _i -  \texttt{convertPointsHomogeneous} ( \texttt{H} * \texttt{srcPoints} _i) \|_2  >  \texttt{ransacReprojThreshold