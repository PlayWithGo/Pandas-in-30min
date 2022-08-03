/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
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
//   * The name of Intel Corporation may not be used to endorse or promote products
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

#ifndef OPENCV_VIDEOIO_H
#define OPENCV_VIDEOIO_H

#include "opencv2/core/core_c.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/**
  @addtogroup videoio_c
  @{
*/

/****************************************************************************************\
*                         Working with Video Files and Cameras                           *
\****************************************************************************************/

/** @brief "black box" capture structure

In C++ use cv::VideoCapture
*/
typedef struct CvCapture CvCapture;

/** @brief start capturing frames from video file
*/
CVAPI(CvCapture*) cvCreateFileCapture( const char* filename );

/** @brief start capturing frames from video file. allows specifying a preferred API to use
*/
CVAPI(CvCapture*) cvCreateFileCaptureWithPreference( const char* filename , int apiPreference);

enum
{
    CV_CAP_ANY      =0,     // autodetect

    CV_CAP_MIL      =100,   // MIL proprietary drivers

    CV_CAP_VFW      =200,   // platform native
    CV_CAP_V4L      =200,
    CV_CAP_V4L2     =200,

    CV_CAP_FIREWARE =300,   // IEEE 1394 drivers
    CV_CAP_FIREWIRE =300,
    CV_CAP_IEEE1394 =300,
    CV_CAP_DC1394   =300,
    CV_CAP_CMU1394  =300,

    CV_CAP_STEREO   =400,   // TYZX proprietary drivers
    CV_CAP_TYZX     =400,
    CV_TYZX_LEFT    =400,
    CV_TYZX_RIGHT   =401,
    CV_TYZX_COLOR   =402,
    CV_TYZX_Z       =403,

    CV_CAP_QT       =500,   // QuickTime

    CV_CAP_UNICAP   =600,   // Unicap drivers

    CV_CAP_DSHOW    =700,   // DirectShow (via videoInput)
    CV_CAP_MSMF     =1400,  // Microsoft Media Foundation (via videoInput)

    CV_CAP_PVAPI    =800,   // PvAPI, Prosilica GigE SDK

    CV_CAP_OPENNI   =900,   // OpenNI (for Kinect)
    CV_CAP_OPENNI_ASUS =910,   // OpenNI (for Asus Xtion)

    CV_CAP_ANDROID  =1000,  // Android - not used
    CV_CAP_ANDROID_BACK =CV_CAP_ANDROID+99, // Android back camera - not used
    CV_CAP_ANDROID_FRONT =CV_CAP_ANDROID+98, // Android front camera - not used

    CV_CAP_XIAPI    =1100,   // XIMEA Camera API

    CV_CAP_AVFOUNDATION = 1200,  // AVFoundation framework for iOS (OS X Lion will have the same API)

    CV_CAP_GIGANETIX = 1300,  // Smartek Giganetix GigEVisionSDK

    CV_CAP_INTELPERC = 1500, // Intel Perceptual Computing

    CV_CAP_OPENNI2 = 1600,   // OpenNI2 (for Kinect)
    CV_CAP_GPHOTO2 = 1700,
    CV_CAP_GSTREAMER = 1800, // GStreamer
    CV_CAP_FFMPEG = 1900,    // FFMPEG
    CV_CAP_IMAGES = 2000,    // OpenCV Image Sequence (e.g. img_%02d.jpg)

    CV_CAP_ARAVIS = 2100     // Aravis GigE SDK
};

/** @brief start capturing frames from camera: index = camera_index + domain_offset (CV_CAP_*)
*/
CVAPI(CvCapture*) cvCreateCameraCapture( int index );

/** @brief grab a frame, return 1 on success, 0 on fail.

  this function is thought to be fast
*/
CVAPI(int) cvGrabFrame( CvCapture* capture );

/** @brief get the frame grabbed with cvGrabFrame(..)

  This function may apply some frame processing like
  frame decompression, flipping etc.
  @warning !!!DO NOT RELEASE or MODIFY the retrieved frame!!!
*/
CVAPI(IplImage*) cvRetrieveFrame( CvCapture* capture, int streamIdx CV_DEFAULT(0) );

/** @brief Just a combination of cvGrabFrame and cvRetrieveFrame

  @warning !!!DO NOT RELEASE or MODIFY the retrieved frame!!!
*/
CVAPI(IplImage*) cvQueryFrame( CvCapture* capture );

/** @brief stop capturing/reading and free resources
*/
CVAPI(void) cvReleaseCapture( CvCapture** ca