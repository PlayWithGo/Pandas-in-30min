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
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2008-2012, Willow Garage Inc., all rights reserved.
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

#ifndef OPENCV_PHOTO_HPP
#define OPENCV_PHOTO_HPP

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

/**
@defgroup photo Computational Photography
@{
    @defgroup photo_denoise Denoising
    @defgroup photo_hdr HDR imaging

This section describes high dynamic range imaging algorithms namely tonemapping, exposure alignment,
camera calibration with multiple exposures and exposure fusion.

    @defgroup photo_clone Seamless Cloning
    @defgroup photo_render Non-Photorealistic Rendering
    @defgroup photo_c C API
@}
  */

namespace cv
{

//! @addtogroup photo
//! @{

//! the inpainting algorithm
enum
{
    INPAINT_NS    = 0, // Navier-Stokes algorithm
    INPAINT_TELEA = 1 // A. Telea algorithm
};

enum
{
    NORMAL_CLONE = 1,
    MIXED_CLONE  = 2,
    MONOCHROME_TRANSFER = 3
};

enum
{
    RECURS_FILTER = 1,
    NORMCONV_FILTER = 2
};

/** @brief Restores the selected region in an image using the region neighborhood.

@param src Input 8-bit, 16-bit unsigned or 32-bit float 1-channel or 8-bit 3-channel image.
@param inpaintMask Inpainting mask, 8-bit 1-channel image. Non-zero pixels indicate the area that
needs to be inpainted.
@param dst Output image with the same size and type as src .
@param inpaintRadius Radius of a circular neighborhood of each point inpainted that is considered
by the algorithm.
@param flags Inpainting method that could be one of the following:
-   **INPAINT_NS** Navier-Stokes based method [Navier01]
-   **INPAINT_TELEA** Method by Alexandru Telea @cite Telea04 .

The function reconstructs the selected image area from the pixel near the area boundary. The
function may be used to remove dust and scratches from a scanned photo, or to remove undesirable
objects from still images or video. See <http://en.wikipedia.org/wiki/Inpainting> for more details.

@note
   -   An example using the inpainting technique can be found at
        opencv_source_code/sampl