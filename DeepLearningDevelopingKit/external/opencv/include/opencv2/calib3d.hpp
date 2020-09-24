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

Real lenses usually have some distortion, mostly radial disto