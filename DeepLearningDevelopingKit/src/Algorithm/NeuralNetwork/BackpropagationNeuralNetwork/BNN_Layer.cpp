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
/// Which means set the nodes` tempInput.
void Neural::InputLayer::SetInput(const Vector<ElemType>& _vec)
{
	for (size_t i = 0; i < n; i++)
	{
		_nodes.at(i).tempInput = _vec(i);
	}
}

// Set the activation function of the layer.
void Neural::InputLayer::SetActivationFunction(const ActivationFunction _function)
{
	switch (_function)
	{
	case ActivationFunction::Sigmoid:
		this->activationFunction = Sigmoid;
		this->activationFunctionDerivative = SigmoidDerivative;
		break;
	case ActivationFunction::ReLU:
		this->activationFunction = ReLU;
		this->activationFunctionDerivative = ReLUDerivative;
		break;
	default:
		this->activationFunction = Sigmoid;
		this->activationFunctionDerivative = SigmoidDerivative;
		break;
	}
}

// Set the loss function of the layer.
void Neural::InputLayer::SetLossFunction(const LossFunction _function)
{
	switch (_function)
	{
	case LossFunction::MES:
		this->lossFunction = MES;
		this->lossFunctionDerivative = MESDerivative;
		break;
	default:
		this->lossFunction = MES;
		this->lossFunctionDerivative = MESDerivative;
		break;
	}
}

// Get the output of the layer.
/// Which means get the value of all nodes in Vector.
Vector<Neural::ElemType> Neural::InputLayer::GetOutput(void)
{
	Vector<ElemType> temp(m);
	for (size_t i = 0; i < m; i++)
	{
		temp(i) = _nodes.at(i).value;
	}
	return temp;
}

// ForwardPropagation Function
/// Calculate the value of each node.
void Neural::InputLayer::ForwardPropagation(void)
{
	for (size_t i = 0; i < m; i++)
	{
		_nodes.at(i).value = _nodes.at(i).tempInput;
	}
}

// BackwardPropagation Function
/// Calculate the gradient(delta) of each node.
Vector<Neural::ElemType> Neural::InputLayer::BackwardPropagation(const Vector<ElemType>& _vec)
{
	/// Calculate the partial derivative of loss to last layer value and return the expectation of last layer.
	return _vec;
}

// Update Function
/// Update the weight and bias of each node.
void Neural::InputLayer::Update(void)
{

}

// Sum up the delta of a batch.
void Neural::InputLayer::BatchDeltaSumUpdate(const size_t _batchSize)
{
}

// Clear the sumdelta of a batch.
void Neural::InputLayer::BatchDeltaSumClear(void)
{
}


/***********************