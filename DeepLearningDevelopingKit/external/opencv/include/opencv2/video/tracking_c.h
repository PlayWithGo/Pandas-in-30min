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

#ifndef OPENCV_TRACKING_C_H
#define OPENCV_TRACKING_C_H

#include "opencv2/imgproc/types_c.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @addtogroup video_c
  @{
*/

/****************************************************************************************\
*                                  Motion Analysis                                       *
\****************************************************************************************/

/************************************ optical flow ***************************************/

#define CV_LKFLOW_PYR_A_READY       1
#define CV_LKFLOW_PYR_B_READY       2
#define CV_LKFLOW_INITIAL_GUESSES   4
#define CV_LKFLOW_GET_MIN_EIGENVALS 8

/* It is Lucas & Kanade method, modified to use pyramids.
   Also it does several iterations to get optical flow for
   every point at every pyramid level.
   Calculates optical flow between two images for certain set of points (i.e.
   it is a "sparse" optical flow, which is opposite to the previous 3 methods) */
CVAPI(void)  cvCalcOpticalFlowPyrLK( const CvArr*  prev, const CvArr*  curr,
                                     CvArr*  prev_pyr, CvArr*  curr_pyr,
                                     const CvPoint2D32f* prev_features,
                                     CvPoint2D32f* curr_features,
                                     int       count,
                                     CvSize    win_size,
                                     int       level,
                                     char*     status,
                                     float*    track_error,
                                     CvTermCriteria criteria,
                                     int       flags );


/* Modification of a previous sparse optical flow algorithm to calculate
   affine flow */
CVAPI(void)  cvCalcAffineFlowPyrLK( const CvArr*  prev, const CvArr*  curr,
                                    CvArr*  prev_pyr, CvArr*  curr_pyr,
                                    const CvPoint2D32f* prev_features,
                                    CvPoint2D32f* curr_features,
                                    float* matrices, int  count,
                                    CvSize win_size, int  level,
                                    char* status, float* track_error,
                                    CvTermCriteria criteria, int flags );

/* Estimate rigid transformation between 2 images or 2 point sets */
CVAPI(int)  cvEstimateRigidTransform( const CvArr* A, const CvArr* B,
                                      CvMat* M, int full_affine );

/* Estimate optical flow for each pixel using the two-frame G. Farneback algorithm */
CVAPI(void) cvCalcOpticalFlowFarneback( const CvArr* prev, const CvArr* next,
                                        CvArr* flow, double pyr_scale, int levels,
                                        int winsize, int iterations, int poly_n,
                                        double poly_sigma, int flags );

/********************************* motion templates *************************************/

/****************************************************************************************\
*        All the motion template functions