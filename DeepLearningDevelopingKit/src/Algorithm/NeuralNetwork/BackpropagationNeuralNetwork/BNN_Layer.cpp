/***************************************************************************************************/
/*                                               Deep Learning Developing Kit                                                   */
/*								        		 	                Layer     	                                                               */
/*                                                   www.tianshicangxie.com                                                        */
/*                                      Copyright © 2015-2018 Celestial Tech Inc.                                          */
/***************************************************************************************************/

// Header files
#include "BNN_Layer.h"

/***************************************************************************************************/
// Class : Layer 
/// Base class of the layer class.
// Get the node number of the layers.

void Neural::Layer::SetLearnRate(const double _learnRate)
{
	this->learnRate = _learnRate;
}

size_t Neural::Layer::GetNodeNum(void)
{
	return m;
}


/***************************************************************************************************/
// Class : InputLayer

// Constructor
/// n is the input num of the layer.
/// m is the output num of the layer, which of course is the node num in this layer.
Neural::InputLayer::InputLayer(const size_t _n, const size_t _m)
{
	for (size_t i = 0; i < _m; i++)
	{
		InputNode tempNode = *new InputNode();
		this->_nodes.push_back(tempNode);
	}
	this->n = _n;
	this->m = _m;
}

// "<<" operator
/// Used for streaming in format.
std::ostream & Neural::operator<<(std::ostream & _outstream, InputLayer & _layer)
{
	_outstream << typeid(_layer).name() << std::endl;
	_outstream << "	Node Num : " << _layer.m << std::endl;
	return _outstream;
}

// Set the input of the layer.
/// Which means set 