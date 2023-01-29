/***************************************************************************************************/
/*                                               Deep Learning Developing Kit                                                   */
/*								        		 	         Serialize Layer     	                                                          */
/*                                                   www.tianshicangxie.com                                                        */
/*                                      Copyright © 2015-2018 Celestial Tech Inc.                                          */
/***************************************************************************************************/
#include "CNN_SerializeLayer.h"

Neural::SerializeLayer::SerializeLayer(const SerializeLayerInitor _initor)
{
	this->_serializedSize = _initor.SerializeSize;
	this->_deserializedSize