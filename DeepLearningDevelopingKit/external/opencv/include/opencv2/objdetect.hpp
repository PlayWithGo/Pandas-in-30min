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

#ifndef OPENCV_OBJDETECT_HPP
#define OPENCV_OBJDETECT_HPP

#include "opencv2/core.hpp"

/**
@defgroup objdetect Object Detection

Haar Feature-based Cascade Classifier for Object Detection
----------------------------------------------------------

The object detector described below has been initially proposed by Paul Viola @cite Viola01 and
improved by Rainer Lienhart @cite Lienhart02 .

First, a classifier (namely a *cascade of boosted classifiers working with haar-like features*) is
trained with a few hundred sample views of a particular object (i.e., a face or a car), called
positive examples, that are scaled to the same size (say, 20x20), and negative examples - arbitrary
images of the same size.

After a classifier is trained, it can be applied to a region of interest (of the same size as used
during the training) in an input image. The classifier outputs a "1" if the region is likely to show
the object (i.e., face/car), and "0" otherwise. To search for the object in the whole image one can
move the search window across the image and check every location using the classifier. The
classifier is designed so that it can be easily "resized" in order to be able to find the objects of
interest at different sizes, which is more efficient than resizing the image itself. So, to find an
object of an unknown size in the image the scan procedure should be done several times at different
scales.

The word "cascade" in the classifier name means that the resultant classifier consists of several
simpler classifiers (*stages*) that are applied subsequently to a region of interest until at some
stage the candidate is rejected or all the stages are passed. The word "boosted" means that the
classifiers at every stage of the cascade are complex themselves and they are built out of basic
classifiers using one of four different boosting techniques (weighted voting). Currently Discrete
Adaboost, Real Adaboost, Gentle Adaboost and Logitboost are supported. The basic classifiers are
decision-tree classifiers with at least 2 leaves. Haar-like features are the input to the basic
classifiers, and are calculated as described below. The current algorithm uses the following
Haar-like features:

![image](pics/haarfeatures.png)

The feature used in a particular classifier is specified by its shape (1a, 2b etc.), position within
the region of interest and the scale (this scale is not the same as the scale used at the detection
stage, though these two scales are multiplied). For example, in the case of the third line feature
(2c) the response is calculated as the difference between the sum of image pixels under the
rectangle covering the whole feature (including the two white stripes and the black stripe in the
middle) and the sum of the image pixels under the black stripe multiplied by 3 in order to
compensate for the differences in the size of areas. The sums of pixel values over a rectangular
regions are calculated rapidly using integral images (see below and the integral description).

To see the object detector at work, have a look at the facedetect demo:
<https://github.com/opencv/opencv/tree/master/samples/cpp/dbt_face_detection.cpp>

The following reference is for the detection part only. There is a separate application called
opencv_traincascade that can train a cascade of boosted classifiers from a set of samples.

@note In the new C++ interface it is also possible to use LBP (local binary pattern) features in
addition to Haar-like features. .. [Viola01] Paul Viola and Michael J. Jones. Rapid Object Detection
using a Boosted Cascade of Simple Features. IEEE CVPR, 2001. The paper is available online at
<http://research.microsoft.com/en-us/um/people/viola/Pubs/Detect/violaJones_CVPR2001.pdf>

@{
    @defgroup objdetect_c C API
@}
 */

typedef struct CvHaarClassifierCascade CvHaarClassifierCascade;

namespace cv
{

//! @addtogroup objdetect
//! @{

///////////////////////////// Object Detection ////////////////////////////

//! class for grouping object candidates, detected by Cascade Classifier, HOG etc.
//! instance of the class is to be passed to cv::partition (see cxoperations.hpp)
class CV_EXPORTS SimilarRects
{
public:
    SimilarRects(double _eps) : eps(_eps) {}
    inline bool operator()(const Rect& r1, const Rect& r2) const
    {
        double delta = eps * ((std::min)(r1.width, r2.width) + (std::min)(r1.height, r2.height)) * 0.5;
        return std::abs(r1.x - r2.x) <= delta &&
            std::abs(r1.y - r2.y) <= delta &&
            std::abs(r1.x + r1.width - r2.x - r2.width) <= delta &&
            std::abs(r1.y + r1.height - r2.y - r2.height) <= delta;
    }
    double eps;
};

/** @brief Groups the object candidate rectangles.

@param rectList Input/output vector of rectangles. Output vector includes retained and grouped
rectangles. (The Python list is not modified in place.)
@param groupThreshold Minimum possible number of rectangles minus 1. The threshold is used in a
group of rectangles to retain it.
@param eps Relative difference between sides of the rectangles to merge them into a group.

The function is a wrapper for the generic function partition . It clusters all the input rectangles
using the rectangle equivalence criteria that combines rectangles with similar sizes and similar
locations. The similarity is defined by eps. When eps=0 , no clustering is done at all. If
\f$\texttt{eps}\rightarrow +\inf\f$ , all the rectangles are put in one cluster. Then, the small
clusters containing less than or equal to groupThreshold rectangles are rejected. In each other
cluster, the average rectangle is computed and put into the output rectangle list.
 */
CV_EXPORTS   void groupRectangles(std::vector<Rect>& rectList, int groupThreshold, double eps = 0.2);
/** @overload */
CV_EXPORTS_W void groupRectangles(CV_IN_OUT std::vector<Rect>& rectList, CV_OUT std::vector<int>& weights,
                                  int groupThreshold, double eps = 0.2);
/** @overload */
CV_EXPORTS   void groupRectangles(std::vector<Rect>& rectList, int groupThreshold,
                                  double eps, std::vector<int>* weights, std::vector<double>* levelWeights );
/** @overload */
CV_EXPORTS   void groupRectangles(std::vector<Rect>& rectList, std::vector<int>& rejectLevels,
                                  std::vector<double>& levelWeights, int groupThreshold, double eps = 0.2);
/** @overload */
CV_EXPORTS   void groupRectangles_meanshift(std::vector<Rect>& rectList, std::vector<double>& foundWeights,
                                            std::vector<double>& foundScales,
                                            double detectThreshold = 0.0, Size winDetSize = Size(64, 128));

template<> CV_EXPORTS void DefaultDeleter<CvHaarClassifierCascade>::operator ()(CvHaarClassifierCascade* obj) const;

enum { CASCADE_DO_CANNY_PRUNING    = 1,
       CASCADE_SCALE_IMAGE         = 2,
       CASCADE_FIND_BIGGEST_OBJECT = 4,
       CASCADE_DO_ROUGH_SEARCH     = 8
     };

class CV_EXPORTS_W BaseCascadeClassifier : public Algorithm
{
public:
    virtual ~BaseCascadeClassifier();
    virtual bool empty() const = 0;
    virtual bool load( const String& filename ) = 0;
    virtual void detectMultiScale( InputArray image,
                           CV_OUT std::vector<Rect>& objects,
                           double scaleFactor,
                           int minNeighbors, int flags,
                           Size minSize, Size maxSize ) = 0;

    virtual void detectMultiScale( InputArray image,
                           CV_OUT std::vector<Rect>& objects,
                           CV_OUT std::vector<int>& numDetections,
                           double scaleFactor,
                           int minNeighbors, int flags,
                           Size minSize, Size maxSize ) = 0;

    virtual void detectMultiScale( InputArray image,
                                   CV_OUT std::vector<Rect>& objects,
                                   CV_OUT std::vector<int>& rejectLevels,
                                   CV_OUT std::vector<double>& levelWeights,
                                   double scaleFactor,
                                   int minNeighbors, int flags,
                                   Size minSi