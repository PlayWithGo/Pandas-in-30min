/***************************************************************************************************/
/*                                               Deep Learning Developing Kit                                                   */
/*								        		 	      ConvolutionalLayer Test   													     */
/*                                                   www.tianshicangxie.com                                                        */
/*                                      Copyright © 2015-2018 Celestial Tech Inc.                                          */
/***************************************************************************************************/

#define ConvolutionalLayerImgDebug

#ifdef ConvolutionalLayerImgDebug

#include "..\Algorithm\NeuralNetwork\NeuralLib.h"
//#include "..\Visualizer\Visualize.h"

int main(int argc, char ** argv)
{
	cv::Mat imgRGB = cv::imread("F:\\Software\\Top Peoject\\DeepLearningProject\\DeepLearningDevelopingKit\\DeepLearningDevelopingKit\\DeepLearningDevelopingKit\\data\\example\\OpenCV\\niconi.png");
	cv::Mat img = cv::imread("F:\\Software\\Top Peoject\\DeepLearningProject\\DeepLearningDevelopingKit\\DeepLearningDevelopingKit\\DeepLearningDevelopingKit\\data\\example\\OpenCV\\niconi.png", cv::IMREAD_GRAYSCALE);
	cv::imshow("Rem", imgRGB);
	cv::waitKey(500);

	cv::normalize(img, img, 0, 1, 