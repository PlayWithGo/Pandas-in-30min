/***************************************************************************************************/
/*                                               Deep Learning Developing Kit                                                   */
/*								        		 	                 Plot   															          */
/*                                                   www.tianshicangxie.com                                                        */
/*                                      Copyright © 2015-2018 Celestial Tech Inc.                                          */
/***************************************************************************************************/

#include "..\..\MathLib\MathLib.h"
#include <string>
#include "..\OpenCV\OpenCV.hpp"

/***************************************************************************************************/
// Namespace : Visual
/// Provide tools for data visualization.
namespace Visual
{
	enum class Plot2DMode {
		R,
		G,
		B,
		RG,
		GB,
		RB,
		RGB
	};

	class Plot2D
	{
	public:
		template<class T>
		static void Plot2DMatrix(const MathLib::Matrix<T> & _mat, const std::string & _name = "Figure1", const Plot2DMode _mode = Plot2DMode::RB, unsigned _x = 100, unsigned _y = 100);

		template<class T>
		static void Plot2DMatrixVec(const std::vector<MathLib::Matrix<T>> & _mat, const std::string & _name = "Figure1", const Plot2DMode _mode = Plot2DMode::RB, unsigned _x = 100, unsigned _y = 100, bool _normalize = false );

		static void PerlinNoiseDemo(const size_t _m, const size_t _n);

		static cv::Mat Scale(const cv::Mat _mat, unsigned n);

		template<class T>
		static void PlotCNNTrainProcess(const std::vector<std::vector<MathLib::Matrix<T>>> _data, const std::string