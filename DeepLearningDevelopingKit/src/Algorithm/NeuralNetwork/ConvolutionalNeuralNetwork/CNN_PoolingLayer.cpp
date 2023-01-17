/***************************************************************************************************/
/*                                               Deep Learning Developing Kit                                                   */
/*								        		 			 Pooling Layer     	                                                          */
/*                                                   www.tianshicangxie.com                                                        */
/*                                      Copyright © 2015-2018 Celestial Tech Inc.                                          */
/***************************************************************************************************/
#include "CNN_PoolingLayer.h"

Neural::PoolingLayer::PoolingLayer(const PoolLayerInitor & _initor)
{
	this->_stride = _initor.Stride;
	this->_poolSize = _initor.PoolSize;
	this->_inputSize = _initor.InputSize;
	this->_poolingMethod = _initor.PoolingMethod;
	this->_paddingMethod = _initor.PaddingMethod;
	this->_paddingNum = _initor.PaddingNum;

	this->_outputSize.m = _inputSize.m /_stride;
	this->_outputSize.n = _inputSize.n / _stride;
	this->_paddingM = _inputSize.m - _stride * _outputSize.m;
	this->_paddingN = _inputSize.n - _stride * _outputSize.n;
}

void Neural::PoolingLayer::SetInput(const std::vector<Feature> & _input)
{
	this->_input = _input;
}

void Neural::PoolingLayer::SetDelta(const std::vector<Feature>& _delta)
{
	this->_delta = _delta;
}

Neural::Feature Neural::PoolingLayer::MaxPool(const Feature & _feature)
{
	Feature tempFeature(_outputSize.m, _outputSize.n,MathLib::MatrixType::Zero);
	size_t kernelOffsetM = 0;
	size_t kernelOffsetN = 0;
	for (size_t a = 0; a < _outputSize.m; a++)
	{
		for (size_t b = 0; b < _outputSize.n; b++)
		{
			tempFeature(a, b) = MaxPoolPart(_feature, kernelOffsetM, kernelOffsetN);
			kernelOffsetN += _stride;
		}
		kernelOffsetM += _stride;
		kernelOffsetN = 0;
	}
	return tempFeature;
}

Neural::ElemType Neural::PoolingLayer::MaxPoolPart(const Feature & _featu