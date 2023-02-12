/***************************************************************************************************/
/*                                               Deep Learning Developing Kit                                                   */
/*								        		 	      SerializeLayer Test   													     */
/*                                                   www.tianshicangxie.com                                                        */
/*                                      Copyright © 2015-2018 Celestial Tech Inc.                                          */
/***************************************************************************************************/

// #define SerializeLayerDebug

#ifdef SerializeLayerDebug
#include "..\ConvolutionalNeuralNetwork\CNN_SerializeLayer.h"

int main(int argc, char ** argv)
{
	const size_t deserializedSizeM = 1;
	const size_t deserializedSizeN = 1;
	const size_t deserializedNum = 5;
	Neural::SerializeLayerInitor serialInitor;
	serialInitor.Seriali