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
\f[\| \texttt{dstPoints} _i -  \texttt{convertPointsHomogeneous} ( \texttt{H} * \texttt{srcPoints} _i) \|_2  >  \texttt{ransacReprojThreshold}\f]
then the point \f$i\f$ is considered as an outlier. If srcPoints and dstPoints are measured in pixels,
it usually makes sense to set this parameter somewhere in the range of 1 to 10.
@param mask Optional output mask set by a robust method ( RANSAC or LMEDS ). Note that the input
mask values are ignored.
@param maxIters The maximum number of RANSAC iterations.
@param confidence Confidence level, between 0 and 1.

The function finds and returns the perspective transformation \f$H\f$ between the source and the
destination planes:

\f[s_i  \vecthree{x'_i}{y'_i}{1} \sim H  \vecthree{x_i}{y_i}{1}\f]

so that the back-projection error

\f[\sum _i \left ( x'_i- \frac{h_{11} x_i + h_{12} y_i + h_{13}}{h_{31} x_i + h_{32} y_i + h_{33}} \right )^2+ \left ( y'_i- \frac{h_{21} x_i + h_{22} y_i + h_{23}}{h_{31} x_i + h_{32} y_i + h_{33}} \right )^2\f]

is minimized. If the parameter method is set to the default value 0, the function uses all the point
pairs to compute an initial homography estimate with a simple least-squares scheme.

However, if not all of the point pairs ( \f$srcPoints_i\f$, \f$dstPoints_i\f$ ) fit the rigid perspective
transformation (that is, there are some outliers), this initial estimate will be poor. In this case,
you can use one of the three robust methods. The methods RANSAC, LMeDS and RHO try many different
random subsets of the corresponding point pairs (of four pairs each, collinear pairs are discarded), estimate the homography matrix
using this subset and a simple least-squares algorithm, and then compute the quality/goodness of the
computed homography (which is the number of inliers for RANSAC or the least median re-projection error for
LMeDS). The best subset is then used to produce the initial estimate of the homography matrix and
the mask of inliers/outliers.

Regardless of the method, robust or not, the computed homography matrix is refined further (using
inliers only in case of a robust method) with the Levenberg-Marquardt method to reduce the
re-projection error even more.

The methods RANSAC and RHO can handle practically any ratio of outliers but need a threshold to
distinguish inliers from outliers. The method LMeDS does not need any threshold but it works
correctly only when there are more than 50% of inliers. Finally, if there are no outliers and the
noise is rather small, use the default method (method=0).

The function is used to find initial intrinsic and extrinsic matrices. Homography matrix is
determined up to a scale. Thus, it is normalized so that \f$h_{33}=1\f$. Note that whenever an \f$H\f$ matrix
cannot be estimated, an empty one will be returned.

@sa
getAffineTransform, estimateAffine2D, estimateAffinePartial2D, getPerspectiveTransform, warpPerspective,
perspectiveTransform
 */
CV_EXPORTS_W Mat findHomography( InputArray srcPoints, InputArray dstPoints,
                                 int method = 0, double ransacReprojThreshold = 3,
                                 OutputArray mask=noArray(), const int maxIters = 2000,
                                 const double confidence = 0.995);

/** @overload */
CV_EXPORTS Mat findHomography( InputArray srcPoints, InputArray dstPoints,
                               OutputArray mask, int method = 0, double ransacReprojThreshold = 3 );

/** @brief Computes an RQ decomposition of 3x3 matrices.

@param src 3x3 input matrix.
@param mtxR Output 3x3 upper-triangular matrix.
@param mtxQ Output 3x3 orthogonal matrix.
@param Qx Optional output 3x3 rotation matrix around x-axis.
@param Qy Optional output 3x3 rotation matrix around y-axis.
@param Qz Optional output 3x3 rotation matrix around z-axis.

The function computes a RQ decomposition using the given rotations. This function is used in
decomposeProjectionMatrix to decompose the left 3x3 submatrix of a projection matrix into a camera
and a rotation matrix.

It optionally returns three rotation matrices, one for each axis, and the three Euler angles in
degrees (as the return value) that could be used in OpenGL. Note, there is always more than one
sequence of rotations about the three principal axes that results in the same orientation of an
object, e.g. see @cite Slabaugh . Returned tree rotation matrices and corresponding three Euler angles
are only one of the possible solutions.
 */
CV_EXPORTS_W Vec3d RQDecomp3x3( InputArray src, OutputArray mtxR, OutputArray mtxQ,
                                OutputArray Qx = noArray(),
                                OutputArray Qy = noArray(),
                                OutputArray Qz = noArray());

/** @brief Decomposes a projection matrix into a rotation matrix and a camera matrix.

@param projMatrix 3x4 input projection matrix P.
@param cameraMatrix Output 3x3 camera matrix K.
@param rotMatrix Output 3x3 external rotation matrix R.
@param transVect Output 4x1 translation vector T.
@param rotMatrixX Optional 3x3 rotation matrix around x-axis.
@param rotMatrixY Optional 3x3 rotation matrix around y-axis.
@param rotMatrixZ Optional 3x3 rotation matrix around z-axis.
@param eulerAngles Optional three-element vector containing three Euler angles of rotation in
degrees.

The function computes a decomposition of a projection matrix into a calibration and a rotation
matrix and the position of a camera.

It optionally returns three rotation matrices, one for each axis, and three Euler angles that could
be used in OpenGL. Note, there is always more than one sequence of rotations about the three
principal axes that results in the same orientation of an object, e.g. see @cite Slabaugh . Returned
tree rotation matrices and corresponding three Euler angles are only one of the possible solutions.

The function is based on RQDecomp3x3 .
 */
CV_EXPORTS_W void decomposeProjectionMatrix( InputArray projMatrix, OutputArray cameraMatrix,
                                             OutputArray rotMatrix, OutputArray transVect,
                                             OutputArray rotMatrixX = noArray(),
                                             OutputArray rotMatrixY = noArray(),
                                             OutputArray rotMatrixZ = noArray(),
                                             OutputArray eulerAngles =noArray() );

/** @brief Computes partial derivatives of the matrix product for each multiplied matrix.

@param A First multiplied matrix.
@param B Second multiplied matrix.
@param dABdA First output derivative matrix d(A\*B)/dA of size
\f$\texttt{A.rows*B.cols} \times {A.rows*A.cols}\f$ .
@param dABdB Second output derivative matrix d(A\*B)/dB of size
\f$\texttt{A.rows*B.cols} \times {B.rows*B.cols}\f$ .

The function computes partial derivatives of the elements of the matrix product \f$A*B\f$ with regard to
the elements of each of the two input matrices. The function is used to compute the Jacobian
matrices in stereoCalibrate but can also be used in any other similar optimization function.
 */
CV_EXPORTS_W void matMulDeriv( InputArray A, InputArray B, OutputArray dABdA, OutputArray dABdB );

/** @brief Combines two rotation-and-shift transformations.

@param rvec1 First rotation vector.
@param tvec1 First translation vector.
@param rvec2 Second rotation vector.
@param tvec2 Second translation vector.
@param rvec3 Output rotation vector of the superposition.
@param tvec3 Output translation vector of the superposition.
@param dr3dr1
@param dr3dt1
@param dr3dr2
@param dr3dt2
@param dt3dr1
@param dt3dt1
@param dt3dr2
@param dt3dt2 Optional output derivatives of rvec3 or tvec3 with regard to rvec1, rvec2, tvec1 and
tvec2, respectively.

The functions compute:

\f[\begin{array}{l} \texttt{rvec3} =  \mathrm{rodrigues} ^{-1} \left ( \mathrm{rodrigues} ( \texttt{rvec2} )  \cdot \mathrm{rodrigues} ( \texttt{rvec1} ) \right )  \\ \texttt{tvec3} =  \mathrm{rodrigues} ( \texttt{rvec2} )  \cdot \texttt{tvec1} +  \texttt{tvec2} \end{array} ,\f]

where \f$\mathrm{rodrigues}\f$ denotes a rotation vector to a rotation matrix transformation, and
\f$\mathrm{rodrigues}^{-1}\f$ denotes the inverse transformation. See Rodrigues for details.

Also, the functions can compute the derivatives of the output vectors with regards to the input
vectors (see matMulDeriv ). The functions are used inside stereoCalibrate but can also be used in
your own code where Levenberg-Marquardt or another gradient-based solver is used to optimize a
function that contains a matrix multiplication.
 */
CV_EXPORTS_W void composeRT( InputArray rvec1, InputArray tvec1,
                             InputArray rvec2, InputArray tvec2,
                             OutputArray rvec3, OutputArray tvec3,
                             OutputArray dr3dr1 = noArray(), OutputArray dr3dt1 = noArray(),
                             OutputArray dr3dr2 = noArray(), OutputArray dr3dt2 = noArray(),
                             OutputArray dt3dr1 = noArray(), OutputArray dt3dt1 = noArray(),
                             OutputArray dt3dr2 = noArray(), OutputArray dt3dt2 = noArray() );

/** @brief Projects 3D points to an image plane.

@param objectPoints Array of object points, 3xN/Nx3 1-channel or 1xN/Nx1 3-channel (or
vector\<Point3f\> ), where N is the number of points in the view.
@param rvec Rotation vector. See Rodrigues for details.
@param tvec Translation vector.
@param cameraMatrix Camera matrix \f$A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{_1}\f$ .
@param distCoeffs Input vector of distortion coefficients
\f$(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6 [, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\f$ of
4, 5, 8, 12 or 14 elements. If the vector is empty, the zero distortion coefficients are assumed.
@param imagePoints Output array of image points, 2xN/Nx2 1-channel or 1xN/Nx1 2-channel, or
vector\<Point2f\> .
@param jacobian Optional output 2Nx(10+\<numDistCoeffs\>) jacobian matrix of derivatives of image
points with respect to components of the rotation vector, translation vector, focal lengths,
coordinates of the principal point and the distortion coefficients. In the old interface different
components of the jacobian are returned via different output parameters.
@param aspectRatio Optional "fixed aspect ratio" parameter. If the parameter is not 0, the
function assumes that the aspect ratio (*fx/fy*) is fixed and correspondingly adjusts the jacobian
matrix.

The function computes projections of 3D points to the image plane given intrinsic and extrinsic
camera parameters. Optionally, the function computes Jacobians - matrices of partial derivatives of
image points coordinates (as functions of all the input parameters) with respect to the particular
parameters, intrinsic and/or extrinsic. The Jacobians are used during the global optimization in
calibrateCamera, solvePnP, and stereoCalibrate . The function itself can also be used to compute a
re-projection error given the current intrinsic and extrinsic parameters.

@note By setting rvec=tvec=(0,0,0) or by setting cameraMatrix to a 3x3 identity matrix, or by
passing zero distortion coefficients, you can get various useful partial cases of the function. This
means that you can compute the distorted coordinates for a sparse set of points or apply a
perspective transformation (and also compute the derivatives) in the ideal zero-distortion setup.
 */
CV_EXPORTS_W void projectPoints( InputArray objectPoints,
                                 InputArray rvec, InputArray tvec,
                                 InputArray cameraMatrix, InputArray distCoeffs,
                                 OutputArray imagePoints,
                                 OutputArray jacobian = noArray(),
                                 double aspectRatio = 0 );

/** @example homography_from_camera_displacement.cpp
  An example program about homography from the camera displacement

  Check @ref tutorial_homography "the corresponding tutorial" for more details
 */

/** @brief Finds an object pose from 3D-2D point correspondences.

@param objectPoints Array of object points in the object coordinate space, Nx3 1-channel or
1xN/Nx1 3-channel, where N is the number of points. vector\<Point3f\> can be also passed here.
@param imagePoints Array of corresponding image points, Nx2 1-channel or 1xN/Nx1 2-channel,
where N is the number of points. vector\<Point2f\> can be also passed here.
@param cameraMatrix Input camera matrix \f$A = \vecthreethree{fx}{0}{cx}{0}{fy}{cy}{0}{0}{1}\f$ .
@param distCoeffs Input vector of distortion coefficients
\f$(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6 [, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\f$ of
4, 5, 8, 12 or 14 elements. If the vector is NULL/empty, the zero distortion coefficients are
assumed.
@param rvec Output rotation vector (see @ref Rodrigues ) that, together with tvec , brings points from
the model coordinate system to the camera coordinate system.
@param tvec Output translation vector.
@param useExtrinsicGuess Parameter used for #SOLVEPNP_ITERATIVE. If true (1), the function uses
the provided rvec and tvec values as initial approximations of the rotation and translation
vectors, respectively, and further optimizes them.
@param flags Method for solving a PnP problem:
-   **SOLVEPNP_ITERATIVE** Iterative method is based on Levenberg-Marquardt optimization. In
this case the function finds such a pose that minimizes reprojection error, that is the sum
of squared distances between the observed projections imagePoints and the projected (using
projectPoints ) objectPoints .
-   **SOLVEPNP_P3P** Method is based on the paper of X.S. Gao, X.-R. Hou, J. Tang, H.-F. Chang
"Complete Solution Classification for the Perspective-Three-Point Problem" (@cite gao2003complete).
In this case the function requires exactly four object and image points.
-   **SOLVEPNP_AP3P** Method is based on the paper of T. Ke, S. Roumeliotis
"An Efficient Algebraic Solution to the Perspective-Three-Point Problem" (@cite Ke17).
In this case the function requires exactly four object and image points.
-   **SOLVEPNP_EPNP** Method has been introduced by F.Moreno-Noguer, V.Lepetit and P.Fua in the
paper "EPnP: Efficient Perspective-n-Point Camera Pose Estimation" (@cite lepetit2009epnp).
-   **SOLVEPNP_DLS** Method is based on the paper of Joel A. Hesch and Stergios I. Roumeliotis.
"A Direct Least-Squares (DLS) Method for PnP" (@cite hesch2011direct).
-   **SOLVEPNP_UPNP** Method is based on the paper of A.Penate-Sanchez, J.Andrade-Cetto,
F.Moreno-Noguer. "Exhaustive Linearization for Robust Camera Pose and Focal Length
Estimation" (@cite penate2013exhaustive). In this case the function also estimates the parameters \f$f_x\f$ and \f$f_y\f$
assuming that both have the same value. Then the cameraMatrix is updated with the estimated
focal length.
-   **SOLVEPNP_AP3P** Method is based on the paper of Tong Ke and Stergios I. Roumeliotis.
"An Efficient Algebraic Solution to the Perspective-Three-Point Problem" (@cite Ke17). In this case the
function requires exactly four object and image points.

The function estimates the object pose given a set of object points, their corresponding image
projections, as well as the camera matrix and the distortion coefficients, see the figure below
(more precisely, the X-axis of the camera frame is pointing to the right, the Y-axis downward
and the Z-axis forward).

![](pnp.jpg)

Points expressed in the world frame \f$ \bf{X}_w \f$ are projected into the image plane \f$ \left[ u, v \right] \f$
using the perspective projection model \f$ \Pi \f$ and the camera intrinsic parameters matrix \f$ \bf{A} \f$:

\f[
  \begin{align*}
  \begin{bmatrix}
  u \\
  v \\
  1
  \end{bmatrix} &=
  \bf{A} \hspace{0.1em} \Pi \hspace{0.2em} ^{c}\bf{M}_w
  \begin{bmatrix}
  X_{w} \\
  Y_{w} \\
  Z_{w} \\
  1
  \end{bmatrix} \\
  \begin{bmatrix}
  u \\
  v \\
  1
  \end{bmatrix} &=
  \begin{bmatrix}
  f_x & 0 & c_x \\
  0 & f_y & c_y \\
  0 & 0 & 1
  \end{bmatrix}
  \begin{bmatrix}
  1 & 0 & 0 & 0 \\
  0 & 1 & 0 & 0 \\
  0 & 0 & 1 & 0
  \end{bmatrix}
  \begin{bmatrix}
  r_{11} & r_{12} & r_{13} & t_x \\
  r_{21} & r_{22} & r_{23} & t_y \\
  r_{31} & r_{32} & r_{33} & t_z \\
  0 & 0 & 0 & 1
  \end{bmatrix}
  \begin{bmatrix}
  X_{w} \\
  Y_{w} \\
  Z_{w} \\
  1
  \end{bmatrix}
  \end{align*}
\f]

The estimated pose is thus the rotation (`rvec`) and the translation (`tvec`) vectors that allow to transform
a 3D point expressed in the world frame into the camera frame:

\f[
  \begin{align*}
  \begin{bmatrix}
  X_c \\
  Y_c \\
  Z_c \\
  1
  \end{bmatrix} &=
  \hspace{0.2em} ^{c}\bf{M}_w
  \begin{bmatrix}
  X_{w} \\
  Y_{w} \\
  Z_{w} \\
  1
  \end{bmatrix} \\
  \begin{bmatrix}
  X_c \\
  Y_c \\
  Z_c \\
  1
  \end{bmatrix} &=
  \begin{bmatrix}
  r_{11} & r_{12} & r_{13} & t_x \\
  r_{21} & r_{22} & r_{23} & t_y \\
  r_{31} & r_{32} & r_{33} & t_z \\
  0 & 0 & 0 & 1
  \end{bmatrix}
  \begin{bmatrix}
  X_{w} \\
  Y_{w} \\
  Z_{w} \\
  1
  \end{bmatrix}
  \end{align*}
\f]

@note
   -   An example of how to use solvePnP for planar augmented reality can be found at
        opencv_source_code/samples/python/plane_ar.py
   -   If you are using Python:
        - Numpy array slices won't work as input because solvePnP requires contiguous
        arrays (enforced by the assertion using cv::Mat::checkVector() around line 55 of
        modules/calib3d/src/solvepnp.cpp version 2.4.9)
        - The P3P algorithm requires image points to be in an array of shape (N,1,2) due
        to its calling of cv::undistortPoints (around line 75 of modules/calib3d/src/solvepnp.cpp version 2.4.9)
        which requires 2-channel information.
        - Thus, given some data D = np.array(...) where D.shape = (N,M), in order to use a subset of
        it as, e.g., imagePoints, one must effectively copy it into a new array: imagePoints =
        np.ascontiguousarray(D[:,:2]).reshape((N,1,2))
   -   The methods **SOLVEPNP_DLS** and **SOLVEPNP_UPNP** cannot be used as the current implementations are
       unstable and sometimes give completely wrong results. If you pass one of these two
       flags, **SOLVEPNP_EPNP** method will be used instead.
   -   The minimum number of points is 4 in the general case. In the case of **SOLVEPNP_P3P** and **SOLVEPNP_AP3P**
       methods, it is required to use exactly 4 points (the first 3 points are used to estimate all the solutions
       of the P3P problem, the last one is used to retain the best solution that minimizes the reprojection error).
   -   With **SOLVEPNP_ITERATIVE** method and `useExtrinsicGuess=true`, the minimum number of points is 3 (3 points
       are sufficient to compute a pose but there are up to 4 solutions). The initial solution should be close to the
       global solution to converge.
 */
CV_EXPORTS_W bool solvePnP( InputArray objectPoints, InputArray imagePoints,
                            InputArray cameraMatrix, InputArray distCoeffs,
                            OutputArray rvec, OutputArray tvec,
                            bool useExtrinsicGuess = false, int flags = SOLVEPNP_ITERATIVE );

/** @brief Finds an object pose from 3D-2D point correspondences using the RANSAC scheme.

@param objectPoints Array of object points in the object coordinate space, Nx3 1-channel or
1xN/Nx1 3-channel, where N is the number of points. vector\<Point3f\> can be also passed here.
@param imagePoints Array of corresponding image points, Nx2 1-channel or 1xN/Nx1 2-channel,
where N is the number of points. vector\<Point2f\> can be also passed here.
@param cameraMatrix Input camera matrix \f$A = \vecthreethree{fx}{0}{cx}{0}{fy}{cy}{0}{0}{1}\f$ .
@param distCoeffs Input vector of distortion coefficients
\f$(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6 [, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\f$ of
4, 5, 8, 12 or 14 elements. If the vector is NULL/empty, the zero distortion coefficients are
assumed.
@param rvec Output rotation vector (see Rodrigues ) that, together with tvec , brings points from
the model coordinate system to the camera coordinate system.
@param tvec Output translation vector.
@param useExtrinsicGuess Parameter used for SOLVEPNP_ITERATIVE. If true (1), the function uses
the provided rvec and tvec values as initial approximations of the rotation and translation
vectors, respectively, and further optimizes them.
@param iterationsCount Number of iterations.
@param reprojectionError Inlier threshold value used by the RANSAC procedure. The parameter value
is the maximum allowed distance between the observed and computed point projections to consider it
an inlier.
@param confidence The probability that the algorithm produces a useful result.
@param inliers Output vector that contains indices of inliers in objectPoints and imagePoints .
@param flags Method for solving a PnP problem (see solvePnP ).

The function estimates an object pose given a set of object points, their corresponding image
projections, as well as the camera matrix and the distortion coefficients. This function finds such
a pose that minimizes reprojection error, that is, the sum of squared distances between the observed
projections imagePoints and the projected (using projectPoints ) objectPoints. The use of RANSAC
makes the function resistant to outliers.

@note
   -   An example of how to use solvePNPRansac for object detection can be found at
        opencv_source_code/samples/cpp/tutorial_code/calib3d/real_time_pose_estimation/
   -   The default method used to estimate the camera pose for the Minimal Sample Sets step
       is #SOLVEPNP_EPNP. Exceptions are:
         - if you choose #SOLVEPNP_P3P or #SOLVEPNP_AP3P, these methods will be used.
         - if the number of input points is equal to 4, #SOLVEPNP_P3P is used.
   -   The method used to estimate the camera pose using all the inliers is defined by the
       flags parameters unless it is equal to #SOLVEPNP_P3P or #SOLVEPNP_AP3P. In this case,
       the method #SOLVEPNP_EPNP will be used instead.
 */
CV_EXPORTS_W bool solvePnPRansac( InputArray objectPoints, InputArray imagePoints,
                                  InputArray cameraMatrix, InputArray distCoeffs,
                                  OutputArray rvec, OutputArray tvec,
                                  bool useExtrinsicGuess = false, int iterationsCount = 100,
                                  float reprojectionError = 8.0, double confidence = 0.99,
                                  OutputArray inliers = noArray(), int flags = SOLVEPNP_ITERATIVE );
/** @brief Finds an object pose from 3 3D-2D point correspondences.

@param objectPoints Array of object points in the object coordinate space, 3x3 1-channel or
1x3/3x1 3-channel. vector\<Point3f\> can be also passed here.
@param imagePoints Array of corresponding image points, 3x2 1-channel or 1x3/3x1 2-channel.
 vector\<Point2f\> can be also passed here.
@param cameraMatrix Input camera matrix \f$A = \vecthreethree{fx}{0}{cx}{0}{fy}{cy}{0}{0}{1}\f$ .
@param distCoeffs Input vector of distortion coefficients
\f$(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6 [, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\f$ of
4, 5, 8, 12 or 14 elements. If the vector is NULL/empty, the zero distortion coefficients are
assumed.
@param rvecs Output rotation vectors (see Rodrigues ) that, together with tvecs , brings points from
the model coordinate system to the camera coordinate system. A P3P problem has up to 4 solutions.
@param tvecs Output translation vectors.
@param flags Method for solving a P3P problem:
-   **SOLVEPNP_P3P** Method is based on the paper of X.S. Gao, X.-R. Hou, J. Tang, H.-F. Chang
"Complete Solution Classification for the Perspective-Three-Point Problem" (@cite gao2003complete).
-   **SOLVEPNP_AP3P** Method is based on the paper of Tong Ke and Stergios I. Roumeliotis.
"An Efficient Algebraic Solution to the Perspective-Three-Point Problem" (@cite Ke17).

The function estimates the object pose given 3 object points, their corresponding image
projections, as well as the camera matrix and the distortion coefficients.
 */
CV_EXPORTS_W int solveP3P( InputArray objectPoints, InputArray imagePoints,
                           InputArray cameraMatrix, InputArray distCoeffs,
                           OutputArrayOfArrays rvecs, OutputArrayOfArrays tvecs,
                           int flags );

/** @brief Finds an initial camera matrix from 3D-2D point correspondences.

@param objectPoints Vector of vectors of the calibration pattern points in the calibration pattern
coordinate space. In the old interface all the per-view vectors are concatenated. See
calibrateCamera for details.
@param imagePoints Vector of vectors of the projections of the calibration pattern points. In the
old interface all the per-view vectors are concatenated.
@param imageSize Image size in pixels used to initialize the principal point.
@param aspectRatio If it is zero or negative, both \f$f_x\f$ and \f$f_y\f$ are estimated independently.
Otherwise, \f$f_x = f_y * \texttt{aspectRatio}\f$ .

The function estimates and returns an initial camera matrix for the camera calibration process.
Currently, the function only supports planar calibration patterns, which are patterns where each
object point has z-coordinate =0.
 */
CV_EXPORTS_W Mat initCameraMatrix2D( InputArrayOfArrays objectPoints,
                                     InputArrayOfArrays imagePoints,
                                     Size imageSize, double aspectRatio = 1.0 );

/** @brief Finds the positions of internal corners of the chessboard.

@param image Source chessboard view. It must be an 8-bit grayscale or color image.
@param patternSize Number of inner corners per a chessboard row and column
( patternSize = cvSize(points_per_row,points_per_colum) = cvSize(columns,rows) ).
@param corners Output array of detected corners.
@param flags Various operation flags that can be zero or a combination of the following values:
-   **CALIB_CB_ADAPTIVE_THRESH** Use adaptive thresholding to convert the image to black
and white, rather than a fixed threshold level (computed from the average image brightness).
-   **CALIB_CB_NORMALIZE_IMAGE** Normalize the image gamma with equalizeHist before
applying fixed or adaptive thresholding.
-   **CALIB_CB_FILTER_QUADS** Use additional criteria (like contour area, perimeter,
square-like shape) to filter out false quads extracted at the contour retrieval stage.
-   **CALIB_CB_FAST_CHECK** Run a fast check on the image that looks for chessboard corners,
and shortcut the call if none is found. This can drastically speed up the call in the
degenerate condition when no chessboard is observed.

The function attempts to determine whether the input image is a view of the chessboard pattern and
locate the internal chessboard corners. The function returns a non-zero value if all of the corners
are found and they are placed in a certain order (row by row, left to right in every row).
Otherwise, if the function fails to find all the corners or reorder them, it returns 0. For example,
a regular chessboard has 8 x 8 squares and 7 x 7 internal corners, that is, points where the black
squares touch each other. The detected coordinates are approximate, and to determine their positions
more accurately, the function calls cornerSubPix. You also may use the function cornerSubPix with
different parameters if returned coordinates are not accurate enough.

Sample usage of detecting and drawing chessboard corners: :
@code
    Size patternsize(8,6); //interior number of corners
    Mat gray = ....; //source image
    vector<Point2f> corners; //this will be filled by the detected corners

    //CALIB_CB_FAST_CHECK saves a lot of time on images
    //that do not contain any chessboard corners
    bool patternfound = findChessboardCorners(gray, patternsize, corners,
            CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE
            + CALIB_CB_FAST_CHECK);

    if(patternfound)
      cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1),
        TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));

    drawChessboardCorners(img, patternsize, Mat(corners), patternfound);
@endcode
@note The function requires white space (like a square-thick border, the wider the better) around
the board to make the detection more robust in various environments. Otherwise, if there is no
border and the background is dark, the outer black squares cannot be segmented properly and so the
square grouping and ordering algorithm fails.
 */
CV_EXPORTS_W bool findChessboardCorners( InputArray image, Size patternSize, OutputArray corners,
                                         int flags = CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE );

//! finds subpixel-accurate positions of the chessboard corners
CV_EXPORTS bool find4QuadCornerSubpix( InputArray img, InputOutputArray corners, Size region_size );

/** @brief Renders the detected chessboard corners.

@param image Destination image. It must be an 8-bit color image.
@param patternSize Number of inner corners per a chessboard row and column
(patternSize = cv::Size(points_per_row,points_per_column)).
@param corners Array of detected corners, the output of findChessboardCorners.
@param patternWasFound Parameter indicating whether the complete board was found or not. The
return value of findChessboardCorners should be passed here.

The function draws individual chessboard corners detected either as red circles if the board was not
found, or as colored corners connected with lines if the board was found.
 */
CV_EXPORTS_W void drawChessboardCorners( InputOutputArray image, Size patternSize,
                                         InputArray corners, bool patternWasFound );

struct CV_EXPORTS_W_SIMPLE CirclesGridFinderParameters
{
    CV_WRAP CirclesGridFinderParameters();
    CV_PROP_RW cv::Size2f densityNeighborhoodSize;
    CV_PROP_RW float minDensity;
    CV_PROP_RW int kmeansAttempts;
    CV_PROP_RW int minDistanceToAddKeypoint;
    CV_PROP_RW int keypointScale;
    CV_PROP_RW float minGraphConfidence;
    CV_PROP_RW float vertexGain;
    CV_PROP_RW float vertexPenalty;
    CV_PROP_RW float existingVertexGain;
    CV_PROP_RW float edgeGain;
    CV_PROP_RW float edgePenalty;
    CV_PROP_RW float convexHullFactor;
    CV_PROP_RW float minRNGEdgeSwitchDist;

    enum GridType
    {
      SYMMETRIC_GRID, ASYMMETRIC_GRID
    };
    GridType gridType;
};

struct CV_EXPORTS_W_SIMPLE CirclesGridFinderParameters2 : public CirclesGridFinderParameters
{
    CV_WRAP CirclesGridFinderParameters2();

    CV_PROP_RW float squareSize; //!< Distance between two adjacent points. Used by CALIB_CB_CLUSTERING.
    CV_PROP_RW float maxRectifiedDistance; //!< Max deviation from predicion. Used by CALIB_CB_CLUSTERING.
};

/** @brief Finds centers in the grid of circles.

@param image grid view of input circles; it must be an 8-bit grayscale or color image.
@param patternSize number of circles per row and column
( patternSize = Size(points_per_row, points_per_colum) ).
@param centers output array of detected centers.
@param flags various operation flags that can be one of the following values:
-   **CALIB_CB_SYMMETRIC_GRID** uses symmetric pattern of circles.
-   **CALIB_CB_ASYMMETRIC_GRID** uses asymmetric pattern of circles.
-   **CALIB_CB_CLUSTERING** uses a special algorithm for grid detection. It is more robust to
perspective distortions but much more sensitive to background clutter.
@param blobDetector feature detector that finds blobs like dark circles on light background.
@param parameters struct for finding circles in a grid pattern.

The function attempts to determine whether the input image contains a grid of circles. If it is, the
function locates centers of the circles. The function returns a non-zero value if all of the centers
have been found and they have been placed in a certain order (row by row, left to right in every
row). Otherwise, if the function fails to find all the corners or reorder them, it returns 0.

Sample usage of detecting and drawing the centers of circles: :
@code
    Size patternsize(7,7); //number of centers
    Mat gray = ....; //source image
    vector<Point2f> centers; //this will be filled by the detected centers

    bool patternfound = findCirclesGrid(gray, patternsize, centers);

    drawChessboardCorners(img, patternsize, Mat(centers), patternfound);
@endcode
@note The function requires white space (like a square-thick border, the wider the better) around
the board to make the detection more robust in various environments.
 */
CV_EXPORTS_W bool findCirclesGrid( InputArray image, Size patternSize,
                                   OutputArray centers, int flags,
                                   const Ptr<FeatureDetector> &blobDetector,
                                   CirclesGridFinderParameters parameters);

/** @overload */
CV_EXPORTS_W bool findCirclesGrid2( InputArray image, Size patternSize,
                                   OutputArray centers, int flags,
                                   const Ptr<FeatureDetector> &blobDetector,
                                   CirclesGridFinderParameters2 parameters);

/** @overload */
CV_EXPORTS_W bool findCirclesGrid( InputArray image, Size patternSize,
                                   OutputArray centers, int flags = CALIB_CB_SYMMETRIC_GRID,
                                   const Ptr<FeatureDetector> &blobDetector = SimpleBlobDetector::create());

/** @brief Finds the camera intrinsic and extrinsic parameters from several views of a calibration pattern.

@param objectPoints In the new interface it is a vector of vectors of calibration pattern points in
the calibration pattern coordinate space (e.g. std::vector<std::vector<cv::Vec3f>>). The outer
vector contains as many elements as the number of the pattern views. If the same calibration pattern
is shown in each view and it is fully visible, all the vectors will be the same. Although, it is
possible to use partially occluded patterns, or even different patterns in different views. Then,
the vectors will be different. The points are 3D, but since they are in a pattern coordinate system,
then, if the rig is planar, it may make sense to put the model to a XY coordinate plane so that
Z-coordinate of each input object point is 0.
In the old interface all the vectors of object points from different views are concatenated
together.
@param imagePoints In the new interface it is a vector of vectors of the projections of calibration
pattern points (e.g. std::vector<std::vector<cv::Vec2f>>). imagePoints.size() and
objectPoints.size() and imagePoints[i].size() must be equal to objectPoints[i].size() for each i.
In the old interface all the vectors of object points from different views are concatenated
together.
@param imageSize Size of the image used only to initialize the intrinsic camera matrix.
@param cameraMatrix Output 3x3 floating-point camera matrix
\f$A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\f$ . If CV\_CALIB\_USE\_INTRINSIC\_GUESS
and/or CALIB_FIX_ASPECT_RATIO are specified, some or all of fx, fy, cx, cy must be
initialized before calling the function.
@param distCoeffs Output vector of distortion coefficients
\f$(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6 [, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\f$ of
4, 5, 8, 12 or 14 elements.
@param rvecs Output vector of rotation vectors (see Rodrigues ) estimated for each pattern view
(e.g. std::vector<cv::Mat>>). That is, each k-th rotation vector together with the corresponding
k-th translation vector (see the next output parameter description) brings the calibration pattern
from the model coordinate space (in which object points are specified) to the world coordinate
space, that is, a real position of the calibration pattern in the k-th pattern view (k=0.. *M* -1).
@param tvecs Output vector of translation vectors estimated for each pattern view.
@param stdDeviationsIntrinsics Output vector of standard deviations estimated for intrinsic parameters.
 Order of deviations values:
\f$(f_x, f_y, c_x, c_y, k_1, k_2, p_1, p_2, k_3, k_4, k_5, k_6 , s_1, s_2, s_3,
 s_4, \tau_x, \tau_y)\f$ If one of parameters is not estimated, it's deviation is equals to zero.
@param stdDeviationsExtrinsics Output vector of standard deviations estimated for extrinsic parameters.
 Order of deviations values: \f$(R_1, T_1, \dotsc , R_M, T_M)\f$ where M is number of pattern views,
 \f$R_i, T_i\f$ are concatenated 1x3 vectors.
 @param perViewErrors Output vector of the RMS re-projection error estimated for each pattern view.
@param flags Different flags that may be zero or a combination of the following values:
-   **CALIB_USE_INTRINSIC_GUESS** cameraMatrix contains valid initial values of
fx, fy, cx, cy that are optimized further. Otherwise, (cx, cy) is initially set to the image
center ( imageSize is used), and focal distances are computed in a least-squares fashion.
Note, that if intrinsic parameters are known, there is no need to use this function just to
estimate extrinsic parameters. Use solvePnP instead.
-   **CALIB_FIX_PRINCIPAL_POINT** The principal point is not changed during the global
optimization. It stays at the center or at a different location specified when
CALIB_USE_INTRINSIC_GUESS is set too.
-   **CALIB_FIX_ASPECT_RATIO** The functions considers only fy as a free parameter. The
ratio fx/fy stays the same as in the input cameraMatrix . When
CALIB_USE_INTRINSIC_GUESS is not set, the actual input values of fx and fy are
ignored, only their ratio is computed and used further.
-   **CALIB_ZERO_TANGENT_DIST** Tangential distortion coefficients \f$(p_1, p_2)\f$ are set
to zeros and stay zero.
-   **CALIB_FIX_K1,...,CALIB_FIX_K6** The corresponding radial distortion
coefficient is not changed during the optimization. If CALIB_USE_INTRINSIC_GUESS is
set, the coefficient from the supplied distCoeffs matrix is used. Otherwise, it is set to 0.
-   **CALIB_RATIONAL_MODEL** Coefficients k4, k5, and k6 are enabled. To provide the
backward compatibility, this extra flag should be explicitly specified to make the
calibration function use the rational model and return 8 coefficients. If the flag is not
set, the function computes and returns only 5 distortion coefficients.
-   **CALIB_THIN_PRISM_MODEL** Coefficients s1, s2, s3 and s4 are enabled. To provide the
backward compatibility, this extra flag should be explicitly specified to make the
calibration function use the thin prism model and return 12 coefficients. If the flag is not
set, the function computes and returns only 5 distortion coefficients.
-   **CALIB_FIX_S1_S2_S3_S4** The thin prism distortion coefficients are not changed during
the optimization. If CALIB_USE_INTRINSIC_GUESS is set, the coefficient from the
supplied distCoeffs matrix is used. Otherwise, it is set to 0.
-   **CALIB_TILTED_MODEL** Coefficients tauX and tauY are enabled. To provide the
backward compatibility, this extra flag should be explicitly specified to make the
calibration function use the tilted sensor model and return 14 coefficients. If the flag is not
set, the function computes and returns only 5 distortion coefficients.
-   **CALIB_FIX_TAUX_TAUY** The coefficients of the tilted sensor model are not changed during
the optimization. If CALIB_USE_INTRINSIC_GUESS is set, the coefficient from the
supplied distCoeffs matrix is used. Otherwise, it is set to 0.
@param criteria Termination criteria for the iterative optimization algorithm.

@return the overall RMS re-projection error.

The function estimates the intrinsic camera parameters and extrinsic parameters for each of the
views. The algorithm is based on @cite Zhang2000 and @cite BouguetMCT . The coordinates of 3D object
points and their corresponding 2D projections in each view must be specified. That may be achieved
by using an object with a known geometry and easily detectable feature points. Such an object is
called a calibration rig or calibration pattern, and OpenCV has built-in support for a chessboard as
a calibration rig (see findChessboardCorners ). Currently, initialization of intrinsic parameters
(when CALIB_USE_INTRINSIC_GUESS is not set) is only implemented for planar calibration
patterns (where Z-coordinates of the object points must be all zeros). 3D calibration rigs can also
be used as long as initial cameraMatrix is provided.

The algorithm performs the following steps:

-   Compute the initial intrinsic parameters (the option only available for planar calibration
    patterns) or read them from the input parameters. The distortion coefficients are all set to
    zeros initially unless some of CALIB_FIX_K? are specified.

-   Estimate the initial camera pose as if the intrinsic parameters have been already known. This is
    done using solvePnP .

-   Run the global Levenberg-Marquardt optimization algorithm to minimize the reprojection error,
    that is, the total sum of squared distances between the observed feature points imagePoints and
    the projected (using the current estimates for camera parameters and the poses) object points
    objectPoints. See projectPoints for details.

@note
   If you use a non-square (=non-NxN) grid and findChessboardCorners for calibration, and
    calibrateCamera returns bad values (zero distortion coefficients, an image center very far from
    (w/2-0.5,h/2-0.5), and/or large differences between \f$f_x\f$ and \f$f_y\f$ (ratios of 10:1 or more)),
    then you have probably used patternSize=cvSize(rows,cols) instead of using
    patternSize=cvSize(cols,rows) in findChessboardCorners .

@sa
   findChessboardCorners, solvePnP, initCameraMatrix2D, stereoCalibrate, undistort
 */
CV_EXPORTS_AS(calibrateCameraExtended) double calibrateCamera( InputArrayOfArrays objectPoints,
                                     InputArrayOfArrays imagePoints, Size imageSize,
                                     InputOutputArray cameraMatrix, InputOutputArray distCoeffs,
                                     OutputArrayOfArrays rvecs, OutputArrayOfArrays tvecs,
                                     OutputArray stdDeviationsIntrinsics,
                                     OutputArray stdDeviationsExtrinsics,
                                     OutputArray perViewErrors,
                                     int flags = 0, TermCriteria criteria = TermCriteria(
                                        TermCriteria::COUNT + TermCriteria::EPS, 30, DBL_EPSILON) );

/** @overload double calibrateCamera( InputArrayOfArrays objectPoints,
                                     InputArrayOfArrays imagePoints, Size imageSize,
                                     InputOutputArray cameraMatrix, InputOutputArray distCoeffs,
                                     OutputArrayOfArrays rvecs, OutputArrayOfArrays tvecs,
                                     OutputArray stdDeviations, OutputArray perViewErrors,
                                     int flags = 0, TermCriteria criteria = TermCriteria(
                                        TermCriteria::COUNT + TermCriteria::EPS, 30, DBL_EPSILON) )
 */
CV_EXPORTS_W double calibrateCamera( InputArrayOfArrays objectPoints,
                                     InputArrayOfArrays imagePoints, Size imageSize,
                                     InputOutputArray cameraMatrix, InputOutputArray distCoeffs,
                                     OutputArrayOfArrays rvecs, OutputArrayOfArrays tvecs,
                                     int flags = 0, TermCriteria criteria = TermCriteria(
                                        TermCriteria::COUNT + TermCriteria::EPS, 30, DBL_EPSILON) );

/** @brief Computes useful camera characteristics from the camera matrix.

@param cameraMatrix Input camera matrix that can be estimated by calibrateCamera or
stereoCalibrate .
@param imageSize Input image size in pixels.
@param apertureWidth Physical width in mm of the sensor.
@param apertureHeight Physical height in mm of the sensor.
@param fovx Output field of view in degrees along the horizontal sensor axis.
@param fovy Output field of view in degrees along the vertical sensor axis.
@param focalLength Focal length of the lens in mm.
@param principalPoint Principal point in mm.
@param aspectRatio \f$f_y/f_x\f$

The function computes various useful camera characteristics from the previously estimated camera
matrix.

@note
   Do keep in mind that the unity measure 'mm' stands for whatever unit of measure one chooses for
    the chessboard pitch (it can thus be any value).
 */
CV_EXPORTS_W void calibrationMatrixValues( InputArray cameraMatrix, Size imageSize,
                                           double apertureWidth, double apertureHeight,
                                           CV_OUT double& fovx, CV_OUT double& fovy,
                                           CV_OUT double& focalLength, CV_OUT Point2d& principalPoint,
                                           CV_OUT double& aspectRatio );

/** @brief Calibrates the stereo camera.

@param objectPoints Vector of vectors of the calibration pattern points.
@param imagePoints1 Vector of vectors of the projections of the calibration pattern points,
observed by the first camera.
@param imagePoints2 Vector of vectors of the projections of the calibration pattern points,
observed by the second camera.
@param cameraMatrix1 Input/output first camera matrix:
\f$\vecthreethree{f_x^{(j)}}{0}{c_x^{(j)}}{0}{f_y^{(j)}}{c_y^{(j)}}{0}{0}{1}\f$ , \f$j = 0,\, 1\f$ . If
any of CALIB_USE_INTRINSIC_GUESS , CALIB_FIX_ASPECT_RATIO ,
CALIB_FIX_INTRINSIC , or CALIB_FIX_FOCAL_LENGTH are specified, some or all of the
matrix components must be initialized. See the flags description for details.
@param distCoeffs1 Input/output vector of distortion coefficients
\f$(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6 [, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\f$ of
4, 5, 8, 12 or 14 elements. The output vector length depends on the flags.
@param cameraMatrix2 Input/output second camera matrix. The parameter is similar to cameraMatrix1
@param distCoeffs2 Input/output lens distortion coefficients for the second camera. The parameter
is similar to distCoeffs1 .
@param imageSize Size of the image used only to initialize intrinsic camera matrix.
@param R Output rotation matrix between the 1st and the 2nd camera coordinate systems.
@param T Output translation vector between the coordinate systems of the cameras.
@param E Output essential matrix.
@param F Output fundamental matrix.
@param perViewErrors Output vector of the RMS re-projection error estimated for each pattern view.
@param flags Different flags that may be zero or a combination of the following values:
-   **CALIB_FIX_INTRINSIC** Fix cameraMatrix? and distCoeffs? so that only R, T, E , and F
matrices are estimated.
-   **CALIB_USE_INTRINSIC_GUESS** Optimize some or all of the intrinsic parameters
according to the specified flags. Initial values are provided by the user.
-   **CALIB_USE_EXTRINSIC_GUESS** R, T contain valid initial values that are optimized further.
Otherwise R, T are initialized to the median value of the pattern views (each dimension separately).
-   **CALIB_FIX_PRINCIPAL_POINT** Fix the principal points during the optimization.
-   **CALIB_FIX_FOCAL_LENGTH** Fix \f$f^{(j)}_x\f$ and \f$f^{(j)}_y\f$ .
-   **CALIB_FIX_ASPECT_RATIO** Optimize \f$f^{(j)}_y\f$ . Fix the ratio \f$f^{(j)}_x/f^{(j)}_y\f$
.
-   **CALIB_SAME_FOCAL_LENGTH** Enforce \f$f^{(0)}_x=f^{(1)}_x\f$ and \f$f^{(0)}_y=f^{(1)}_y\f$ .
-   **CALIB_ZERO_TANGENT_DIST** Set tangential distortion coefficients for each camera to
zeros and fix there.
-   **CALIB_FIX_K1,...,CALIB_FIX_K6** Do not change the corresponding radial
distortion coefficient during the optimization. If CALIB_USE_INTRINSIC_GUESS is set,
the coefficient from the supplied distCoeffs matrix is used. Otherwise, it is set to 0.
-   **CALIB_RATIONAL_MODEL** Enable coefficients k4, k5, and k6. To provide the backward
compatibility, this extra flag should be explicitly specified to make the calibration
function use the rational model and return 8 coefficients. If the flag is not set, the
function computes and returns only 5 distortion coefficients.
-   **CALIB_THIN_PRISM_MODEL** Coefficients s1, s2, s3 and s4 are enabled. To provide the
backward compatibility, this extra flag should be explicitly specified to make the
calibration function use the thin prism model and return 12 coefficients. If the flag is not
set, the function computes and returns only 5 distortion coefficients.
-   **CALIB_FIX_S1_S2_S3_S4** The thin prism distortion coefficients are not changed during
the optimization. If CALIB_USE_INTRINSIC_GUESS is set, the coefficient from the
supplied distCoeffs matrix is used. Otherwise, it is set to 0.
-   **CALIB_TILTED_MODEL** Coefficients tauX and tauY are enabled. To provide the
backward compatibility, this extra flag should be explicitly specified to make the
calibration function use the tilted sensor model and return 14 coefficients. If the flag is not
set, the function computes and returns only 5 distortion coefficients.
-   **CALIB_FIX_TAUX_TAUY** The coefficients of the tilted sensor model are not changed during
the optimization. If CALIB_USE_INTRINSIC_GUESS is set, the coefficient from the
supplied distCoeffs matrix is used. Otherwise, it is set to 0.
@param criteria Termination criteria for the iterative optimization algorithm.

The function estimates transformation between two cameras making a stereo pair. If you have a stereo
camera where the relative position and orientation of two cameras is fixed, and if you computed
poses of an object relative to the first camera and to the second camera, (R1, T1) and (R2, T2),
respectively (this can be done with solvePnP ), then those poses definitely relate to each other.
This means that, given ( \f$R_1\f$,\f$T_1\f$ ), it should be possible to compute ( \f$R_2\f$,\f$T_2\f$ ). You only
need to know the position and orientation of the second camera relative to the first camera. This is
what the described function does. It computes ( \f$R\f$,\f$T\f$ ) so that:

\f[R_2=R*R_1\f]
\f[T_2=R*T_1 + T,\f]

Optionally, it computes the essential matrix E:

\f[E= \vecthreethree{0}{-T_2}{T_1}{T_2}{0}{-T_0}{-T_1}{T_0}{0} *R\f]

where \f$T_i\f$ are components of the translation vector \f$T\f$ : \f$T=[T_0, T_1, T_2]^T\f$ . And the function
can also compute the fundamental matrix F:

\f[F = cameraMatrix2^{-T} E cameraMatrix1^{-1}\f]

Besides the stereo-related information, the function can also perform a full calibration of each of
two cameras. However, due to the high dimensionality of the parameter space and noise in the input
data, the function can diverge from the correct solution. If the intrinsic parameters can be
estimated with high accuracy for each of the cameras individually (for example, using
calibrateCamera ), you are recommended to do so and then pass CALIB_FIX_INTRINSIC flag to the
function along with the computed intrinsic parameters. Otherwise, if all the parameters are
estimated at once, it makes sense to restrict some parameters, for example, pass
CALIB_SAME_FOCAL_LENGTH and CALIB_ZERO_TANGENT_DIST flags, which is usually a
reasonable assumption.

Similarly to calibrateCamera , the function minimizes the total re-projection error for all the
points in all the available views from both cameras. The function returns the final value of the
re-projection error.
 */
CV_EXPORTS_AS(stereoCalibrateExtended) double stereoCalibrate( InputArrayOfArrays objectPoints,
                                     InputArrayOfArrays imagePoints1, InputArrayOfArrays imagePoints2,
                                     InputOutputArray cameraMatrix1, InputOutputArray distCoeffs1,
                                     InputOutputArray cameraMatrix2, InputOutputArray distCoeffs2,
                                     Size imageSize, InputOutputArray R,InputOutputArray T, OutputArray E, OutputArray F,
                                     OutputArray perViewErrors, int flags = CALIB_FIX_INTRINSIC,
                                     TermCriteria criteria = TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 1e-6) );

/// @overload
CV_EXPORTS_W double stereoCalibrate( InputArrayOfArrays objectPoints,
                                     InputArrayOfArrays imagePoints1, InputArrayOfArrays imagePoints2,
                                     InputOutputArray cameraMatrix1, InputOutputArray distCoeffs1,
                                     InputOutputArray cameraMatrix2, InputOutputArray distCoeffs2,
                                     Size imageSize, OutputArray R,OutputArray T, OutputArray E, OutputArray F,
                                     int flags = CALIB_FIX_INTRINSIC,
                                     TermCriteria criteria = TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 1e-6) );

/** @brief Computes rectification transforms for each head of a calibrated stereo camera.

@param cameraMatrix1 First camera matrix.
@param distCoeffs1 First camera distortion parameters.
@param cameraMatrix2 Second camera matrix.
@param distCoeffs2 Second camera distortion parameters.
@param imageSize Size of the image used for stereo calibration.
@param R Rotation matrix between the coordinate systems of the first and the second cameras.
@param T Translation vector between coordinate systems of the cameras.
@param R1 Output 3x3 rectification transform (rotation matrix) for the first camera.
@param R2 Output 3x3 rectification transform (rotation matrix) for the second camera.
@param P1 Output 3x4 projection matrix in the new (rectified) coordinate systems for the first
camera.
@param P2 Output 3x4 projection matrix in the new (rectified) coordinate systems for the second
camera.
@param Q Output \f$4 \times 4\f$ disparity-to-depth mapping matrix (see reprojectImageTo3D ).
@param flags Operation flags that may be zero or CALIB_ZERO_DISPARITY . If the flag is set,
the function makes the principal points of each camera have the same pixel coordinates in the
rectified views. And if the flag is not set, the function may still shift the images in the
horizontal or vertical direction (depending on the orientation of epipolar lines) to maximize the
useful image area.
@param alpha Free scaling parameter. If it is -1 or absent, the function performs the default
scaling. Otherwise, the parameter should be between 0 and 1. alpha=0 means that the rectified
images are zoomed and shifted so that only valid pixels are visible (no black areas after
rectification). alpha=1 means that the rectified image is decimated and shifted so that all the
pixels from the original images from the cameras are retained in the rectified images (no source
image pixels are lost). Obviously, any intermediate value yields an intermediate result between
those two extreme cases.
@param newImageSize New image resolution after rectification. The same size should be passed to
initUndistortRectifyMap (see the stereo_calib.cpp sample in OpenCV samples directory). When (0,0)
is passed (default), it is set to the original imageSize . Setting it to larger value can help you
preserve details in the original image, especially when there is a big radial distortion.
@param validPixROI1 Optional output rectangles inside the rectified images where all the pixels
are valid. If alpha=0 , the ROIs cover the whole images. Otherwise, they are likely to be smaller
(see the picture below).
@param validPixROI2 Optional output rectangles inside the rectified images where all the pixels
are valid. If alpha=0 , the ROIs cover the whole images. Otherwise, they are likely to be smaller
(see the picture below).

The function computes the rotation matrices for each camera that (virtually) make both camera image
planes the same plane. Consequently, this makes all the epipolar lines parallel and thus simplifies
the dense stereo correspondence problem. The function takes the matrices computed by stereoCalibrate
as input. As output, it provides two rotation matrices and also two projection matrices in the new
coordinates. The function distinguishes the following two cases:

-   **Horizontal stereo**: the first and the second camera views are shifted relative to each other
    mainly along the x axis (with possible small vertical shift). In the rectified images, the
    corresponding epipolar lines in the left and right cameras are horizontal and have the same
    y-coordinate. P1 and P2 look like:

    \f[\texttt{P1} = \begin{bmatrix} f & 0 & cx_1 & 0 \\ 0 & f & cy & 0 \\ 0 & 0 & 1 & 0 \end{bmatrix}\f]

    \f[\texttt{P2} = \begin{bmatrix} f & 0 & cx_2 & T_x*f \\ 0 & f & cy & 0 \\ 0 & 0 & 1 & 0 \end{bmatrix} ,\f]

    where \f$T_x\f$ is a horizontal shift between the cameras and \f$cx_1=cx_2\f$ if
    CALIB_ZERO_DISPARITY is set.

-   **Vertical stereo**: the first and the second camera views are shifted relative to each other
    mainly in vertical direction (and probably a bit in the horizontal direction too). The epipolar
    lines in the rectified images are vertical and have the same x-coordinate. P1 and P2 look like:

    \f[\texttt{P1} = \begin{bmatrix} f & 0 & cx & 0 \\ 0 & f & cy_1 & 0 \\ 0 & 0 & 1 & 0 \end{bmatrix}\f]

    \f[\texttt{P2} = \begin{bmatrix} f & 0 & cx & 0 \\ 0 & f & cy_2 & T_y*f \\ 0 & 0 & 1 & 0 \end{bmatrix} ,\f]

    where \f$T_y\f$ is a vertical shift between the cameras and \f$cy_1=cy_2\f$ if CALIB_ZERO_DISPARITY is
    set.

As you can see, the first three columns of P1 and P2 will effectively be the new "rectified" camera
matrices. The matrices, together with R1 and R2 , can then be passed to initUndistortRectifyMap to
initialize the rectification map for each camera.

See below the screenshot from the stereo_calib.cpp sample. Some red horizontal lines pass through
the corresponding image regions. This means that the images are well rectified, which is what most
stereo correspondence algorithms rely on. The green rectangles are roi1 and roi2 . You see that
their interiors are all valid pixels.

![image](pics/stereo_undistort.jpg)
 */
CV_EXPORTS_W void stereoRectify( InputArray cameraMatrix1, InputArray distCoeffs1,
                                 InputArray cameraMatrix2, InputArray distCoeffs2,
                                 Size imageSize, InputArray R, InputArray T,
                                 OutputArray R1, OutputArray R2,
                                 OutputArray P1, OutputArray P2,
                                 OutputArray Q, int flags = CALIB_ZERO_DISPARITY,
                                 double alpha = -1, Size newImageSize = Size(),