// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
#ifndef OPENCV_DNN_LAYER_DETAILS_HPP
#define OPENCV_DNN_LAYER_DETAILS_HPP

#include <opencv2/dnn/layer.hpp>

namespace cv {
namespace dnn {
CV__DNN_EXPERIMENTAL_NS_BEGIN

/** @brief Registers layer constructor in runtime.
*   @param type string, containing type name of the layer.
*   @param constuctorFunc pointer to the function of type LayerRegister::Constuctor, which creates the layer.
*   @details This macros must be placed inside the function code.
*/
#define CV_DNN_REGISTER_LAYER_FUNC(type, constuctorFunc) \
    cv::dnn::LayerFactory::registerLayer(#type, constuctorFunc);

/** @brief Registers layer class in runtime.
 *  @param type string, containing type name of the layer.
 *  @param class C++ class, derived from Layer.
 *  @details This macros must be placed inside the function code.
 */
#define CV_DNN_REGISTER_LAYER_CLASS(type, class) \
    cv::dnn::LayerFactory::registerLayer(#