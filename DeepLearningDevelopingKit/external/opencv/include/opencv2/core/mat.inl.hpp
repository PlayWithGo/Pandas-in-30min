
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
// Copyright (C) 2015, Itseez Inc., all rights reserved.
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

#ifndef OPENCV_CORE_MATRIX_OPERATIONS_HPP
#define OPENCV_CORE_MATRIX_OPERATIONS_HPP

#ifndef __cplusplus
#  error mat.inl.hpp header must be compiled as C++
#endif

#ifdef _MSC_VER
#pragma warning( push )
#pragma warning( disable: 4127 )
#endif

namespace cv
{
CV__DEBUG_NS_BEGIN


//! @cond IGNORED

//////////////////////// Input/Output Arrays ////////////////////////

inline void _InputArray::init(int _flags, const void* _obj)
{ flags = _flags; obj = (void*)_obj; }

inline void _InputArray::init(int _flags, const void* _obj, Size _sz)
{ flags = _flags; obj = (void*)_obj; sz = _sz; }

inline void* _InputArray::getObj() const { return obj; }
inline int _InputArray::getFlags() const { return flags; }
inline Size _InputArray::getSz() const { return sz; }

inline _InputArray::_InputArray() { init(NONE, 0); }
inline _InputArray::_InputArray(int _flags, void* _obj) { init(_flags, _obj); }
inline _InputArray::_InputArray(const Mat& m) { init(MAT+ACCESS_READ, &m); }
inline _InputArray::_InputArray(const std::vector<Mat>& vec) { init(STD_VECTOR_MAT+ACCESS_READ, &vec); }
inline _InputArray::_InputArray(const UMat& m) { init(UMAT+ACCESS_READ, &m); }
inline _InputArray::_InputArray(const std::vector<UMat>& vec) { init(STD_VECTOR_UMAT+ACCESS_READ, &vec); }

template<typename _Tp> inline
_InputArray::_InputArray(const std::vector<_Tp>& vec)
{ init(FIXED_TYPE + STD_VECTOR + traits::Type<_Tp>::value + ACCESS_READ, &vec); }

#ifdef CV_CXX_STD_ARRAY
template<typename _Tp, std::size_t _Nm> inline
_InputArray::_InputArray(const std::array<_Tp, _Nm>& arr)
{ init(FIXED_TYPE + FIXED_SIZE + STD_ARRAY + traits::Type<_Tp>::value + ACCESS_READ, arr.data(), Size(1, _Nm)); }

template<std::size_t _Nm> inline
_InputArray::_InputArray(const std::array<Mat, _Nm>& arr)
{ init(STD_ARRAY_MAT + ACCESS_READ, arr.data(), Size(1, _Nm)); }
#endif

inline
_InputArray::_InputArray(const std::vector<bool>& vec)
{ init(FIXED_TYPE + STD_BOOL_VECTOR + traits::Type<bool>::value + ACCESS_READ, &vec); }

template<typename _Tp> inline
_InputArray::_InputArray(const std::vector<std::vector<_Tp> >& vec)
{ init(FIXED_TYPE + STD_VECTOR_VECTOR + traits::Type<_Tp>::value + ACCESS_READ, &vec); }

inline
_InputArray::_InputArray(const std::vector<std::vector<bool> >&)
{ CV_Error(Error::StsUnsupportedFormat, "std::vector<std::vector<bool> > is not supported!\n"); }

template<typename _Tp> inline
_InputArray::_InputArray(const std::vector<Mat_<_Tp> >& vec)
{ init(FIXED_TYPE + STD_VECTOR_MAT + traits::Type<_Tp>::value + ACCESS_READ, &vec); }

template<typename _Tp, int m, int n> inline
_InputArray::_InputArray(const Matx<_Tp, m, n>& mtx)
{ init(FIXED_TYPE + FIXED_SIZE + MATX + traits::Type<_Tp>::value + ACCESS_READ, &mtx, Size(n, m)); }

template<typename _Tp> inline
_InputArray::_InputArray(const _Tp* vec, int n)
{ init(FIXED_TYPE + FIXED_SIZE + MATX + traits::Type<_Tp>::value + ACCESS_READ, vec, Size(n, 1)); }

template<typename _Tp> inline
_InputArray::_InputArray(const Mat_<_Tp>& m)
{ init(FIXED_TYPE + MAT + traits::Type<_Tp>::value + ACCESS_READ, &m); }

inline _InputArray::_InputArray(const double& val)
{ init(FIXED_TYPE + FIXED_SIZE + MATX + CV_64F + ACCESS_READ, &val, Size(1,1)); }

inline _InputArray::_InputArray(const MatExpr& expr)
{ init(FIXED_TYPE + FIXED_SIZE + EXPR + ACCESS_READ, &expr); }

inline _InputArray::_InputArray(const cuda::GpuMat& d_mat)
{ init(CUDA_GPU_MAT + ACCESS_READ, &d_mat); }

inline _InputArray::_InputArray(const std::vector<cuda::GpuMat>& d_mat)
{	init(STD_VECTOR_CUDA_GPU_MAT + ACCESS_READ, &d_mat);}

inline _InputArray::_InputArray(const ogl::Buffer& buf)
{ init(OPENGL_BUFFER + ACCESS_READ, &buf); }

inline _InputArray::_InputArray(const cuda::HostMem& cuda_mem)
{ init(CUDA_HOST_MEM + ACCESS_READ, &cuda_mem); }

inline _InputArray::~_InputArray() {}

inline Mat _InputArray::getMat(int i) const
{
    if( kind() == MAT && i < 0 )
        return *(const Mat*)obj;
    return getMat_(i);
}

inline bool _InputArray::isMat() const { return kind() == _InputArray::MAT; }
inline bool _InputArray::isUMat() const  { return kind() == _InputArray::UMAT; }
inline bool _InputArray::isMatVector() const { return kind() == _InputArray::STD_VECTOR_MAT; }
inline bool _InputArray::isUMatVector() const  { return kind() == _InputArray::STD_VECTOR_UMAT; }
inline bool _InputArray::isMatx() const { return kind() == _InputArray::MATX; }
inline bool _InputArray::isVector() const { return kind() == _InputArray::STD_VECTOR ||
                                                   kind() == _InputArray::STD_BOOL_VECTOR ||
                                                   kind() == _InputArray::STD_ARRAY; }
inline bool _InputArray::isGpuMatVector() const { return kind() == _InputArray::STD_VECTOR_CUDA_GPU_MAT; }

////////////////////////////////////////////////////////////////////////////////////////

inline _OutputArray::_OutputArray() { init(ACCESS_WRITE, 0); }
inline _OutputArray::_OutputArray(int _flags, void* _obj) { init(_flags|ACCESS_WRITE, _obj); }
inline _OutputArray::_OutputArray(Mat& m) { init(MAT+ACCESS_WRITE, &m); }
inline _OutputArray::_OutputArray(std::vector<Mat>& vec) { init(STD_VECTOR_MAT+ACCESS_WRITE, &vec); }
inline _OutputArray::_OutputArray(UMat& m) { init(UMAT+ACCESS_WRITE, &m); }
inline _OutputArray::_OutputArray(std::vector<UMat>& vec) { init(STD_VECTOR_UMAT+ACCESS_WRITE, &vec); }

template<typename _Tp> inline
_OutputArray::_OutputArray(std::vector<_Tp>& vec)
{ init(FIXED_TYPE + STD_VECTOR + traits::Type<_Tp>::value + ACCESS_WRITE, &vec); }

#ifdef CV_CXX_STD_ARRAY
template<typename _Tp, std::size_t _Nm> inline
_OutputArray::_OutputArray(std::array<_Tp, _Nm>& arr)
{ init(FIXED_TYPE + FIXED_SIZE + STD_ARRAY + traits::Type<_Tp>::value + ACCESS_WRITE, arr.data(), Size(1, _Nm)); }

template<std::size_t _Nm> inline
_OutputArray::_OutputArray(std::array<Mat, _Nm>& arr)
{ init(STD_ARRAY_MAT + ACCESS_WRITE, arr.data(), Size(1, _Nm)); }
#endif

inline
_OutputArray::_OutputArray(std::vector<bool>&)
{ CV_Error(Error::StsUnsupportedFormat, "std::vector<bool> cannot be an output array\n"); }

template<typename _Tp> inline
_OutputArray::_OutputArray(std::vector<std::vector<_Tp> >& vec)
{ init(FIXED_TYPE + STD_VECTOR_VECTOR + traits::Type<_Tp>::value + ACCESS_WRITE, &vec); }

inline
_OutputArray::_OutputArray(std::vector<std::vector<bool> >&)
{ CV_Error(Error::StsUnsupportedFormat, "std::vector<std::vector<bool> > cannot be an output array\n"); }

template<typename _Tp> inline
_OutputArray::_OutputArray(std::vector<Mat_<_Tp> >& vec)
{ init(FIXED_TYPE + STD_VECTOR_MAT + traits::Type<_Tp>::value + ACCESS_WRITE, &vec); }

template<typename _Tp> inline
_OutputArray::_OutputArray(Mat_<_Tp>& m)
{ init(FIXED_TYPE + MAT + traits::Type<_Tp>::value + ACCESS_WRITE, &m); }

template<typename _Tp, int m, int n> inline
_OutputArray::_OutputArray(Matx<_Tp, m, n>& mtx)
{ init(FIXED_TYPE + FIXED_SIZE + MATX + traits::Type<_Tp>::value + ACCESS_WRITE, &mtx, Size(n, m)); }

template<typename _Tp> inline
_OutputArray::_OutputArray(_Tp* vec, int n)
{ init(FIXED_TYPE + FIXED_SIZE + MATX + traits::Type<_Tp>::value + ACCESS_WRITE, vec, Size(n, 1)); }

template<typename _Tp> inline
_OutputArray::_OutputArray(const std::vector<_Tp>& vec)
{ init(FIXED_TYPE + FIXED_SIZE + STD_VECTOR + traits::Type<_Tp>::value + ACCESS_WRITE, &vec); }

#ifdef CV_CXX_STD_ARRAY
template<typename _Tp, std::size_t _Nm> inline
_OutputArray::_OutputArray(const std::array<_Tp, _Nm>& arr)
{ init(FIXED_TYPE + FIXED_SIZE + STD_ARRAY + traits::Type<_Tp>::value + ACCESS_WRITE, arr.data(), Size(1, _Nm)); }

template<std::size_t _Nm> inline
_OutputArray::_OutputArray(const std::array<Mat, _Nm>& arr)
{ init(FIXED_SIZE + STD_ARRAY_MAT + ACCESS_WRITE, arr.data(), Size(1, _Nm)); }
#endif

template<typename _Tp> inline
_OutputArray::_OutputArray(const std::vector<std::vector<_Tp> >& vec)
{ init(FIXED_TYPE + FIXED_SIZE + STD_VECTOR_VECTOR + traits::Type<_Tp>::value + ACCESS_WRITE, &vec); }

template<typename _Tp> inline
_OutputArray::_OutputArray(const std::vector<Mat_<_Tp> >& vec)
{ init(FIXED_TYPE + FIXED_SIZE + STD_VECTOR_MAT + traits::Type<_Tp>::value + ACCESS_WRITE, &vec); }

template<typename _Tp> inline
_OutputArray::_OutputArray(const Mat_<_Tp>& m)
{ init(FIXED_TYPE + FIXED_SIZE + MAT + traits::Type<_Tp>::value + ACCESS_WRITE, &m); }

template<typename _Tp, int m, int n> inline
_OutputArray::_OutputArray(const Matx<_Tp, m, n>& mtx)
{ init(FIXED_TYPE + FIXED_SIZE + MATX + traits::Type<_Tp>::value + ACCESS_WRITE, &mtx, Size(n, m)); }

template<typename _Tp> inline
_OutputArray::_OutputArray(const _Tp* vec, int n)
{ init(FIXED_TYPE + FIXED_SIZE + MATX + traits::Type<_Tp>::value + ACCESS_WRITE, vec, Size(n, 1)); }

inline _OutputArray::_OutputArray(cuda::GpuMat& d_mat)
{ init(CUDA_GPU_MAT + ACCESS_WRITE, &d_mat); }

inline _OutputArray::_OutputArray(std::vector<cuda::GpuMat>& d_mat)
{	init(STD_VECTOR_CUDA_GPU_MAT + ACCESS_WRITE, &d_mat);}

inline _OutputArray::_OutputArray(ogl::Buffer& buf)
{ init(OPENGL_BUFFER + ACCESS_WRITE, &buf); }

inline _OutputArray::_OutputArray(cuda::HostMem& cuda_mem)
{ init(CUDA_HOST_MEM + ACCESS_WRITE, &cuda_mem); }

inline _OutputArray::_OutputArray(const Mat& m)
{ init(FIXED_TYPE + FIXED_SIZE + MAT + ACCESS_WRITE, &m); }

inline _OutputArray::_OutputArray(const std::vector<Mat>& vec)
{ init(FIXED_SIZE + STD_VECTOR_MAT + ACCESS_WRITE, &vec); }

inline _OutputArray::_OutputArray(const UMat& m)
{ init(FIXED_TYPE + FIXED_SIZE + UMAT + ACCESS_WRITE, &m); }

inline _OutputArray::_OutputArray(const std::vector<UMat>& vec)
{ init(FIXED_SIZE + STD_VECTOR_UMAT + ACCESS_WRITE, &vec); }

inline _OutputArray::_OutputArray(const cuda::GpuMat& d_mat)
{ init(FIXED_TYPE + FIXED_SIZE + CUDA_GPU_MAT + ACCESS_WRITE, &d_mat); }


inline _OutputArray::_OutputArray(const ogl::Buffer& buf)
{ init(FIXED_TYPE + FIXED_SIZE + OPENGL_BUFFER + ACCESS_WRITE, &buf); }

inline _OutputArray::_OutputArray(const cuda::HostMem& cuda_mem)
{ init(FIXED_TYPE + FIXED_SIZE + CUDA_HOST_MEM + ACCESS_WRITE, &cuda_mem); }

///////////////////////////////////////////////////////////////////////////////////////////

inline _InputOutputArray::_InputOutputArray() { init(ACCESS_RW, 0); }
inline _InputOutputArray::_InputOutputArray(int _flags, void* _obj) { init(_flags|ACCESS_RW, _obj); }
inline _InputOutputArray::_InputOutputArray(Mat& m) { init(MAT+ACCESS_RW, &m); }
inline _InputOutputArray::_InputOutputArray(std::vector<Mat>& vec) { init(STD_VECTOR_MAT+ACCESS_RW, &vec); }
inline _InputOutputArray::_InputOutputArray(UMat& m) { init(UMAT+ACCESS_RW, &m); }
inline _InputOutputArray::_InputOutputArray(std::vector<UMat>& vec) { init(STD_VECTOR_UMAT+ACCESS_RW, &vec); }

template<typename _Tp> inline
_InputOutputArray::_InputOutputArray(std::vector<_Tp>& vec)
{ init(FIXED_TYPE + STD_VECTOR + traits::Type<_Tp>::value + ACCESS_RW, &vec); }

#ifdef CV_CXX_STD_ARRAY
template<typename _Tp, std::size_t _Nm> inline
_InputOutputArray::_InputOutputArray(std::array<_Tp, _Nm>& arr)
{ init(FIXED_TYPE + FIXED_SIZE + STD_ARRAY + traits::Type<_Tp>::value + ACCESS_RW, arr.data(), Size(1, _Nm)); }

template<std::size_t _Nm> inline
_InputOutputArray::_InputOutputArray(std::array<Mat, _Nm>& arr)
{ init(STD_ARRAY_MAT + ACCESS_RW, arr.data(), Size(1, _Nm)); }
#endif

inline _InputOutputArray::_InputOutputArray(std::vector<bool>&)
{ CV_Error(Error::StsUnsupportedFormat, "std::vector<bool> cannot be an input/output array\n"); }

template<typename _Tp> inline
_InputOutputArray::_InputOutputArray(std::vector<std::vector<_Tp> >& vec)
{ init(FIXED_TYPE + STD_VECTOR_VECTOR + traits::Type<_Tp>::value + ACCESS_RW, &vec); }

template<typename _Tp> inline
_InputOutputArray::_InputOutputArray(std::vector<Mat_<_Tp> >& vec)
{ init(FIXED_TYPE + STD_VECTOR_MAT + traits::Type<_Tp>::value + ACCESS_RW, &vec); }

template<typename _Tp> inline
_InputOutputArray::_InputOutputArray(Mat_<_Tp>& m)
{ init(FIXED_TYPE + MAT + traits::Type<_Tp>::value + ACCESS_RW, &m); }

template<typename _Tp, int m, int n> inline
_InputOutputArray::_InputOutputArray(Matx<_Tp, m, n>& mtx)
{ init(FIXED_TYPE + FIXED_SIZE + MATX + traits::Type<_Tp>::value + ACCESS_RW, &mtx, Size(n, m)); }

template<typename _Tp> inline
_InputOutputArray::_InputOutputArray(_Tp* vec, int n)
{ init(FIXED_TYPE + FIXED_SIZE + MATX + traits::Type<_Tp>::value + ACCESS_RW, vec, Size(n, 1)); }

template<typename _Tp> inline
_InputOutputArray::_InputOutputArray(const std::vector<_Tp>& vec)
{ init(FIXED_TYPE + FIXED_SIZE + STD_VECTOR + traits::Type<_Tp>::value + ACCESS_RW, &vec); }

#ifdef CV_CXX_STD_ARRAY
template<typename _Tp, std::size_t _Nm> inline
_InputOutputArray::_InputOutputArray(const std::array<_Tp, _Nm>& arr)
{ init(FIXED_TYPE + FIXED_SIZE + STD_ARRAY + traits::Type<_Tp>::value + ACCESS_RW, arr.data(), Size(1, _Nm)); }

template<std::size_t _Nm> inline
_InputOutputArray::_InputOutputArray(const std::array<Mat, _Nm>& arr)
{ init(FIXED_SIZE + STD_ARRAY_MAT + ACCESS_RW, arr.data(), Size(1, _Nm)); }
#endif

template<typename _Tp> inline
_InputOutputArray::_InputOutputArray(const std::vector<std::vector<_Tp> >& vec)
{ init(FIXED_TYPE + FIXED_SIZE + STD_VECTOR_VECTOR + traits::Type<_Tp>::value + ACCESS_RW, &vec); }

template<typename _Tp> inline
_InputOutputArray::_InputOutputArray(const std::vector<Mat_<_Tp> >& vec)
{ init(FIXED_TYPE + FIXED_SIZE + STD_VECTOR_MAT + traits::Type<_Tp>::value + ACCESS_RW, &vec); }

template<typename _Tp> inline
_InputOutputArray::_InputOutputArray(const Mat_<_Tp>& m)
{ init(FIXED_TYPE + FIXED_SIZE + MAT + traits::Type<_Tp>::value + ACCESS_RW, &m); }

template<typename _Tp, int m, int n> inline
_InputOutputArray::_InputOutputArray(const Matx<_Tp, m, n>& mtx)
{ init(FIXED_TYPE + FIXED_SIZE + MATX + traits::Type<_Tp>::value + ACCESS_RW, &mtx, Size(n, m)); }

template<typename _Tp> inline
_InputOutputArray::_InputOutputArray(const _Tp* vec, int n)
{ init(FIXED_TYPE + FIXED_SIZE + MATX + traits::Type<_Tp>::value + ACCESS_RW, vec, Size(n, 1)); }

inline _InputOutputArray::_InputOutputArray(cuda::GpuMat& d_mat)
{ init(CUDA_GPU_MAT + ACCESS_RW, &d_mat); }

inline _InputOutputArray::_InputOutputArray(ogl::Buffer& buf)
{ init(OPENGL_BUFFER + ACCESS_RW, &buf); }

inline _InputOutputArray::_InputOutputArray(cuda::HostMem& cuda_mem)
{ init(CUDA_HOST_MEM + ACCESS_RW, &cuda_mem); }

inline _InputOutputArray::_InputOutputArray(const Mat& m)
{ init(FIXED_TYPE + FIXED_SIZE + MAT + ACCESS_RW, &m); }

inline _InputOutputArray::_InputOutputArray(const std::vector<Mat>& vec)
{ init(FIXED_SIZE + STD_VECTOR_MAT + ACCESS_RW, &vec); }

inline _InputOutputArray::_InputOutputArray(const UMat& m)
{ init(FIXED_TYPE + FIXED_SIZE + UMAT + ACCESS_RW, &m); }

inline _InputOutputArray::_InputOutputArray(const std::vector<UMat>& vec)
{ init(FIXED_SIZE + STD_VECTOR_UMAT + ACCESS_RW, &vec); }

inline _InputOutputArray::_InputOutputArray(const cuda::GpuMat& d_mat)
{ init(FIXED_TYPE + FIXED_SIZE + CUDA_GPU_MAT + ACCESS_RW, &d_mat); }

inline _InputOutputArray::_InputOutputArray(const std::vector<cuda::GpuMat>& d_mat)
{ init(FIXED_TYPE + FIXED_SIZE + STD_VECTOR_CUDA_GPU_MAT + ACCESS_RW, &d_mat);}

template<> inline _InputOutputArray::_InputOutputArray(std::vector<cuda::GpuMat>& d_mat)
{ init(FIXED_TYPE + FIXED_SIZE + STD_VECTOR_CUDA_GPU_MAT + ACCESS_RW, &d_mat);}

inline _InputOutputArray::_InputOutputArray(const ogl::Buffer& buf)
{ init(FIXED_TYPE + FIXED_SIZE + OPENGL_BUFFER + ACCESS_RW, &buf); }

inline _InputOutputArray::_InputOutputArray(const cuda::HostMem& cuda_mem)
{ init(FIXED_TYPE + FIXED_SIZE + CUDA_HOST_MEM + ACCESS_RW, &cuda_mem); }

CV__DEBUG_NS_END

//////////////////////////////////////////// Mat //////////////////////////////////////////

inline
Mat::Mat()
    : flags(MAGIC_VAL), dims(0), rows(0), cols(0), data(0), datastart(0), dataend(0),
      datalimit(0), allocator(0), u(0), size(&rows), step(0)
{}

inline
Mat::Mat(int _rows, int _cols, int _type)
    : flags(MAGIC_VAL), dims(0), rows(0), cols(0), data(0), datastart(0), dataend(0),
      datalimit(0), allocator(0), u(0), size(&rows), step(0)
{
    create(_rows, _cols, _type);
}

inline
Mat::Mat(int _rows, int _cols, int _type, const Scalar& _s)
    : flags(MAGIC_VAL), dims(0), rows(0), cols(0), data(0), datastart(0), dataend(0),
      datalimit(0), allocator(0), u(0), size(&rows), step(0)
{
    create(_rows, _cols, _type);
    *this = _s;
}

inline
Mat::Mat(Size _sz, int _type)
    : flags(MAGIC_VAL), dims(0), rows(0), cols(0), data(0), datastart(0), dataend(0),
      datalimit(0), allocator(0), u(0), size(&rows), step(0)
{
    create( _sz.height, _sz.width, _type );
}

inline
Mat::Mat(Size _sz, int _type, const Scalar& _s)
    : flags(MAGIC_VAL), dims(0), rows(0), cols(0), data(0), datastart(0), dataend(0),
      datalimit(0), allocator(0), u(0), size(&rows), step(0)
{
    create(_sz.height, _sz.width, _type);
    *this = _s;
}

inline
Mat::Mat(int _dims, const int* _sz, int _type)
    : flags(MAGIC_VAL), dims(0), rows(0), cols(0), data(0), datastart(0), dataend(0),
      datalimit(0), allocator(0), u(0), size(&rows), step(0)
{
    create(_dims, _sz, _type);
}

inline
Mat::Mat(int _dims, const int* _sz, int _type, const Scalar& _s)
    : flags(MAGIC_VAL), dims(0), rows(0), cols(0), data(0), datastart(0), dataend(0),
      datalimit(0), allocator(0), u(0), size(&rows), step(0)
{
    create(_dims, _sz, _type);
    *this = _s;
}

inline
Mat::Mat(const std::vector<int>& _sz, int _type)
    : flags(MAGIC_VAL), dims(0), rows(0), cols(0), data(0), datastart(0), dataend(0),
      datalimit(0), allocator(0), u(0), size(&rows), step(0)
{
    create(_sz, _type);
}

inline
Mat::Mat(const std::vector<int>& _sz, int _type, const Scalar& _s)
    : flags(MAGIC_VAL), dims(0), rows(0), cols(0), data(0), datastart(0), dataend(0),
      datalimit(0), allocator(0), u(0), size(&rows), step(0)
{
    create(_sz, _type);
    *this = _s;
}

inline
Mat::Mat(const Mat& m)
    : flags(m.flags), dims(m.dims), rows(m.rows), cols(m.cols), data(m.data),
      datastart(m.datastart), dataend(m.dataend), datalimit(m.datalimit), allocator(m.allocator),
      u(m.u), size(&rows), step(0)
{
    if( u )
        CV_XADD(&u->refcount, 1);
    if( m.dims <= 2 )
    {
        step[0] = m.step[0]; step[1] = m.step[1];
    }
    else
    {
        dims = 0;
        copySize(m);
    }
}

inline
Mat::Mat(int _rows, int _cols, int _type, void* _data, size_t _step)
    : flags(MAGIC_VAL + (_type & TYPE_MASK)), dims(2), rows(_rows), cols(_cols),
      data((uchar*)_data), datastart((uchar*)_data), dataend(0), datalimit(0),
      allocator(0), u(0), size(&rows)
{
    CV_Assert(total() == 0 || data != NULL);

    size_t esz = CV_ELEM_SIZE(_type), esz1 = CV_ELEM_SIZE1(_type);
    size_t minstep = cols * esz;
    if( _step == AUTO_STEP )
    {
        _step = minstep;
        flags |= CONTINUOUS_FLAG;
    }
    else
    {
        CV_DbgAssert( _step >= minstep );

        if (_step % esz1 != 0)
        {
            CV_Error(Error::BadStep, "Step must be a multiple of esz1");
        }

        if (_step == minstep || rows == 1)
            flags |= CONTINUOUS_FLAG;
    }
    step[0] = _step;
    step[1] = esz;
    datalimit = datastart + _step * rows;
    dataend = datalimit - _step + minstep;
}

inline
Mat::Mat(Size _sz, int _type, void* _data, size_t _step)
    : flags(MAGIC_VAL + (_type & TYPE_MASK)), dims(2), rows(_sz.height), cols(_sz.width),
      data((uchar*)_data), datastart((uchar*)_data), dataend(0), datalimit(0),
      allocator(0), u(0), size(&rows)
{
    CV_Assert(total() == 0 || data != NULL);

    size_t esz = CV_ELEM_SIZE(_type), esz1 = CV_ELEM_SIZE1(_type);
    size_t minstep = cols*esz;
    if( _step == AUTO_STEP )
    {
        _step = minstep;
        flags |= CONTINUOUS_FLAG;
    }
    else
    {
        CV_DbgAssert( _step >= minstep );

        if (_step % esz1 != 0)
        {
            CV_Error(Error::BadStep, "Step must be a multiple of esz1");
        }

        if (_step == minstep || rows == 1)
            flags |= CONTINUOUS_FLAG;
    }
    step[0] = _step;
    step[1] = esz;
    datalimit = datastart + _step*rows;
    dataend = datalimit - _step + minstep;
}

template<typename _Tp> inline
Mat::Mat(const std::vector<_Tp>& vec, bool copyData)
    : flags(MAGIC_VAL | traits::Type<_Tp>::value | CV_MAT_CONT_FLAG), dims(2), rows((int)vec.size()),
      cols(1), data(0), datastart(0), dataend(0), datalimit(0), allocator(0), u(0), size(&rows), step(0)
{
    if(vec.empty())
        return;
    if( !copyData )
    {
        step[0] = step[1] = sizeof(_Tp);
        datastart = data = (uchar*)&vec[0];
        datalimit = dataend = datastart + rows * step[0];
    }
    else
        Mat((int)vec.size(), 1, traits::Type<_Tp>::value, (uchar*)&vec[0]).copyTo(*this);
}

#ifdef CV_CXX11
template<typename _Tp, typename> inline
Mat::Mat(const std::initializer_list<_Tp> list)
    : flags(MAGIC_VAL | traits::Type<_Tp>::value | CV_MAT_CONT_FLAG), dims(2), rows((int)list.size()),
      cols(1), data(0), datastart(0), dataend(0), datalimit(0), allocator(0), u(0), size(&rows), step(0)
{
    if(list.size() == 0)
        return;
    Mat((int)list.size(), 1, traits::Type<_Tp>::value, (uchar*)list.begin()).copyTo(*this);
}

template<typename _Tp> inline
Mat::Mat(const std::initializer_list<int> sizes, const std::initializer_list<_Tp> list)
    : Mat()
{
    size_t size_total = 1;
    int *sz = (int*)sizes.begin();
    for(auto s : sizes)
        size_total *= s;
    CV_Assert(list.size() != 0 || size_total == list.size());
    Mat((int)sizes.size(), sz, traits::Type<_Tp>::value, (uchar*)list.begin()).copyTo(*this);
}
#endif

#ifdef CV_CXX_STD_ARRAY
template<typename _Tp, std::size_t _Nm> inline
Mat::Mat(const std::array<_Tp, _Nm>& arr, bool copyData)
    : flags(MAGIC_VAL | traits::Type<_Tp>::value | CV_MAT_CONT_FLAG), dims(2), rows((int)arr.size()),
      cols(1), data(0), datastart(0), dataend(0), datalimit(0), allocator(0), u(0), size(&rows), step(0)
{
    if(arr.empty())
        return;
    if( !copyData )
    {
        step[0] = step[1] = sizeof(_Tp);
        datastart = data = (uchar*)arr.data();
        datalimit = dataend = datastart + rows * step[0];
    }
    else
        Mat((int)arr.size(), 1, traits::Type<_Tp>::value, (uchar*)arr.data()).copyTo(*this);
}
#endif

template<typename _Tp, int n> inline
Mat::Mat(const Vec<_Tp, n>& vec, bool copyData)
    : flags(MAGIC_VAL | traits::Type<_Tp>::value | CV_MAT_CONT_FLAG), dims(2), rows(n), cols(1), data(0),
      datastart(0), dataend(0), datalimit(0), allocator(0), u(0), size(&rows), step(0)
{
    if( !copyData )
    {
        step[0] = step[1] = sizeof(_Tp);
        datastart = data = (uchar*)vec.val;
        datalimit = dataend = datastart + rows * step[0];
    }
    else
        Mat(n, 1, traits::Type<_Tp>::value, (void*)vec.val).copyTo(*this);
}


template<typename _Tp, int m, int n> inline
Mat::Mat(const Matx<_Tp,m,n>& M, bool copyData)
    : flags(MAGIC_VAL | traits::Type<_Tp>::value | CV_MAT_CONT_FLAG), dims(2), rows(m), cols(n), data(0),
      datastart(0), dataend(0), datalimit(0), allocator(0), u(0), size(&rows), step(0)
{
    if( !copyData )
    {
        step[0] = cols * sizeof(_Tp);
        step[1] = sizeof(_Tp);
        datastart = data = (uchar*)M.val;
        datalimit = dataend = datastart + rows * step[0];
    }
    else
        Mat(m, n, traits::Type<_Tp>::value, (uchar*)M.val).copyTo(*this);
}

template<typename _Tp> inline
Mat::Mat(const Point_<_Tp>& pt, bool copyData)
    : flags(MAGIC_VAL | traits::Type<_Tp>::value | CV_MAT_CONT_FLAG), dims(2), rows(2), cols(1), data(0),
      datastart(0), dataend(0), datalimit(0), allocator(0), u(0), size(&rows), step(0)
{
    if( !copyData )
    {
        step[0] = step[1] = sizeof(_Tp);
        datastart = data = (uchar*)&pt.x;
        datalimit = dataend = datastart + rows * step[0];
    }
    else
    {
        create(2, 1, traits::Type<_Tp>::value);
        ((_Tp*)data)[0] = pt.x;
        ((_Tp*)data)[1] = pt.y;
    }
}

template<typename _Tp> inline
Mat::Mat(const Point3_<_Tp>& pt, bool copyData)
    : flags(MAGIC_VAL | traits::Type<_Tp>::value | CV_MAT_CONT_FLAG), dims(2), rows(3), cols(1), data(0),
      datastart(0), dataend(0), datalimit(0), allocator(0), u(0), size(&rows), step(0)
{
    if( !copyData )
    {
        step[0] = step[1] = sizeof(_Tp);
        datastart = data = (uchar*)&pt.x;
        datalimit = dataend = datastart + rows * step[0];
    }
    else
    {
        create(3, 1, traits::Type<_Tp>::value);
        ((_Tp*)data)[0] = pt.x;
        ((_Tp*)data)[1] = pt.y;
        ((_Tp*)data)[2] = pt.z;
    }
}

template<typename _Tp> inline
Mat::Mat(const MatCommaInitializer_<_Tp>& commaInitializer)
    : flags(MAGIC_VAL | traits::Type<_Tp>::value | CV_MAT_CONT_FLAG), dims(0), rows(0), cols(0), data(0),
      datastart(0), dataend(0), allocator(0), u(0), size(&rows)
{
    *this = commaInitializer.operator Mat_<_Tp>();
}

inline
Mat::~Mat()
{
    release();
    if( step.p != step.buf )
        fastFree(step.p);
}

inline
Mat& Mat::operator = (const Mat& m)
{
    if( this != &m )
    {
        if( m.u )
            CV_XADD(&m.u->refcount, 1);
        release();
        flags = m.flags;
        if( dims <= 2 && m.dims <= 2 )
        {
            dims = m.dims;
            rows = m.rows;
            cols = m.cols;
            step[0] = m.step[0];
            step[1] = m.step[1];
        }
        else
            copySize(m);
        data = m.data;
        datastart = m.datastart;
        dataend = m.dataend;
        datalimit = m.datalimit;
        allocator = m.allocator;
        u = m.u;
    }
    return *this;
}

inline
Mat Mat::row(int y) const
{
    return Mat(*this, Range(y, y + 1), Range::all());
}

inline
Mat Mat::col(int x) const
{
    return Mat(*this, Range::all(), Range(x, x + 1));
}

inline
Mat Mat::rowRange(int startrow, int endrow) const
{
    return Mat(*this, Range(startrow, endrow), Range::all());
}

inline
Mat Mat::rowRange(const Range& r) const
{
    return Mat(*this, r, Range::all());
}

inline
Mat Mat::colRange(int startcol, int endcol) const
{
    return Mat(*this, Range::all(), Range(startcol, endcol));
}

inline
Mat Mat::colRange(const Range& r) const
{
    return Mat(*this, Range::all(), r);
}

inline
Mat Mat::clone() const
{
    Mat m;
    copyTo(m);
    return m;
}

inline
void Mat::assignTo( Mat& m, int _type ) const
{
    if( _type < 0 )
        m = *this;
    else
        convertTo(m, _type);
}

inline
void Mat::create(int _rows, int _cols, int _type)
{
    _type &= TYPE_MASK;
    if( dims <= 2 && rows == _rows && cols == _cols && type() == _type && data )
        return;
    int sz[] = {_rows, _cols};
    create(2, sz, _type);
}

inline
void Mat::create(Size _sz, int _type)
{
    create(_sz.height, _sz.width, _type);
}

inline
void Mat::addref()
{
    if( u )
        CV_XADD(&u->refcount, 1);
}

inline
void Mat::release()
{
    if( u && CV_XADD(&u->refcount, -1) == 1 )
        deallocate();
    u = NULL;
    datastart = dataend = datalimit = data = 0;
    for(int i = 0; i < dims; i++)
        size.p[i] = 0;
#ifdef _DEBUG
    flags = MAGIC_VAL;
    dims = rows = cols = 0;
    if(step.p != step.buf)
    {
        fastFree(step.p);
        step.p = step.buf;
        size.p = &rows;
    }
#endif
}

inline
Mat Mat::operator()( Range _rowRange, Range _colRange ) const
{
    return Mat(*this, _rowRange, _colRange);
}

inline
Mat Mat::operator()( const Rect& roi ) const
{
    return Mat(*this, roi);
}

inline
Mat Mat::operator()(const Range* ranges) const
{
    return Mat(*this, ranges);
}

inline
Mat Mat::operator()(const std::vector<Range>& ranges) const
{
    return Mat(*this, ranges);
}

inline
bool Mat::isContinuous() const
{
    return (flags & CONTINUOUS_FLAG) != 0;
}

inline
bool Mat::isSubmatrix() const
{
    return (flags & SUBMATRIX_FLAG) != 0;
}

inline
size_t Mat::elemSize() const
{
    return dims > 0 ? step.p[dims - 1] : 0;
}

inline
size_t Mat::elemSize1() const
{
    return CV_ELEM_SIZE1(flags);
}

inline
int Mat::type() const
{
    return CV_MAT_TYPE(flags);
}

inline
int Mat::depth() const
{
    return CV_MAT_DEPTH(flags);
}

inline
int Mat::channels() const
{
    return CV_MAT_CN(flags);
}

inline
size_t Mat::step1(int i) const
{
    return step.p[i] / elemSize1();
}

inline
bool Mat::empty() const
{
    return data == 0 || total() == 0 || dims == 0;
}

inline
size_t Mat::total() const
{
    if( dims <= 2 )
        return (size_t)rows * cols;
    size_t p = 1;
    for( int i = 0; i < dims; i++ )
        p *= size[i];
    return p;
}

inline
size_t Mat::total(int startDim, int endDim) const
{
    CV_Assert( 0 <= startDim && startDim <= endDim);
    size_t p = 1;
    int endDim_ = endDim <= dims ? endDim : dims;
    for( int i = startDim; i < endDim_; i++ )
        p *= size[i];
    return p;
}

inline
uchar* Mat::ptr(int y)
{
    CV_DbgAssert( y == 0 || (data && dims >= 1 && (unsigned)y < (unsigned)size.p[0]) );
    return data + step.p[0] * y;
}

inline
const uchar* Mat::ptr(int y) const
{
    CV_DbgAssert( y == 0 || (data && dims >= 1 && (unsigned)y < (unsigned)size.p[0]) );
    return data + step.p[0] * y;
}

template<typename _Tp> inline
_Tp* Mat::ptr(int y)
{
    CV_DbgAssert( y == 0 || (data && dims >= 1 && (unsigned)y < (unsigned)size.p[0]) );
    return (_Tp*)(data + step.p[0] * y);
}

template<typename _Tp> inline
const _Tp* Mat::ptr(int y) const
{
    CV_DbgAssert( y == 0 || (data && dims >= 1 && data && (unsigned)y < (unsigned)size.p[0]) );
    return (const _Tp*)(data + step.p[0] * y);
}

inline
uchar* Mat::ptr(int i0, int i1)
{
    CV_DbgAssert(dims >= 2);
    CV_DbgAssert(data);
    CV_DbgAssert((unsigned)i0 < (unsigned)size.p[0]);
    CV_DbgAssert((unsigned)i1 < (unsigned)size.p[1]);
    return data + i0 * step.p[0] + i1 * step.p[1];
}

inline
const uchar* Mat::ptr(int i0, int i1) const
{
    CV_DbgAssert(dims >= 2);
    CV_DbgAssert(data);
    CV_DbgAssert((unsigned)i0 < (unsigned)size.p[0]);
    CV_DbgAssert((unsigned)i1 < (unsigned)size.p[1]);
    return data + i0 * step.p[0] + i1 * step.p[1];
}

template<typename _Tp> inline
_Tp* Mat::ptr(int i0, int i1)
{
    CV_DbgAssert(dims >= 2);
    CV_DbgAssert(data);
    CV_DbgAssert((unsigned)i0 < (unsigned)size.p[0]);
    CV_DbgAssert((unsigned)i1 < (unsigned)size.p[1]);
    return (_Tp*)(data + i0 * step.p[0] + i1 * step.p[1]);
}

template<typename _Tp> inline
const _Tp* Mat::ptr(int i0, int i1) const
{
    CV_DbgAssert(dims >= 2);
    CV_DbgAssert(data);
    CV_DbgAssert((unsigned)i0 < (unsigned)size.p[0]);
    CV_DbgAssert((unsigned)i1 < (unsigned)size.p[1]);
    return (const _Tp*)(data + i0 * step.p[0] + i1 * step.p[1]);
}

inline
uchar* Mat::ptr(int i0, int i1, int i2)
{
    CV_DbgAssert(dims >= 3);
    CV_DbgAssert(data);
    CV_DbgAssert((unsigned)i0 < (unsigned)size.p[0]);
    CV_DbgAssert((unsigned)i1 < (unsigned)size.p[1]);
    CV_DbgAssert((unsigned)i2 < (unsigned)size.p[2]);
    return data + i0 * step.p[0] + i1 * step.p[1] + i2 * step.p[2];
}

inline
const uchar* Mat::ptr(int i0, int i1, int i2) const
{
    CV_DbgAssert(dims >= 3);
    CV_DbgAssert(data);
    CV_DbgAssert((unsigned)i0 < (unsigned)size.p[0]);
    CV_DbgAssert((unsigned)i1 < (unsigned)size.p[1]);
    CV_DbgAssert((unsigned)i2 < (unsigned)size.p[2]);
    return data + i0 * step.p[0] + i1 * step.p[1] + i2 * step.p[2];
}

template<typename _Tp> inline
_Tp* Mat::ptr(int i0, int i1, int i2)
{
    CV_DbgAssert(dims >= 3);
    CV_DbgAssert(data);
    CV_DbgAssert((unsigned)i0 < (unsigned)size.p[0]);
    CV_DbgAssert((unsigned)i1 < (unsigned)size.p[1]);
    CV_DbgAssert((unsigned)i2 < (unsigned)size.p[2]);
    return (_Tp*)(data + i0 * step.p[0] + i1 * step.p[1] + i2 * step.p[2]);
}

template<typename _Tp> inline
const _Tp* Mat::ptr(int i0, int i1, int i2) const
{
    CV_DbgAssert(dims >= 3);
    CV_DbgAssert(data);
    CV_DbgAssert((unsigned)i0 < (unsigned)size.p[0]);
    CV_DbgAssert((unsigned)i1 < (unsigned)size.p[1]);
    CV_DbgAssert((unsigned)i2 < (unsigned)size.p[2]);
    return (const _Tp*)(data + i0 * step.p[0] + i1 * step.p[1] + i2 * step.p[2]);
}

inline
uchar* Mat::ptr(const int* idx)
{
    int i, d = dims;
    uchar* p = data;
    CV_DbgAssert( d >= 1 && p );
    for( i = 0; i < d; i++ )
    {
        CV_DbgAssert( (unsigned)idx[i] < (unsigned)size.p[i] );
        p += idx[i] * step.p[i];
    }
    return p;
}

inline
const uchar* Mat::ptr(const int* idx) const
{
    int i, d = dims;
    uchar* p = data;
    CV_DbgAssert( d >= 1 && p );
    for( i = 0; i < d; i++ )
    {
        CV_DbgAssert( (unsigned)idx[i] < (unsigned)size.p[i] );
        p += idx[i] * step.p[i];
    }
    return p;
}

template<typename _Tp> inline
_Tp* Mat::ptr(const int* idx)
{
    int i, d = dims;
    uchar* p = data;
    CV_DbgAssert( d >= 1 && p );
    for( i = 0; i < d; i++ )
    {
        CV_DbgAssert( (unsigned)idx[i] < (unsigned)size.p[i] );
        p += idx[i] * step.p[i];
    }
    return (_Tp*)p;
}

template<typename _Tp> inline
const _Tp* Mat::ptr(const int* idx) const
{
    int i, d = dims;
    uchar* p = data;
    CV_DbgAssert( d >= 1 && p );
    for( i = 0; i < d; i++ )
    {
        CV_DbgAssert( (unsigned)idx[i] < (unsigned)size.p[i] );
        p += idx[i] * step.p[i];
    }
    return (const _Tp*)p;
}

template<typename _Tp> inline
_Tp& Mat::at(int i0, int i1)
{
    CV_DbgAssert(dims <= 2);
    CV_DbgAssert(data);
    CV_DbgAssert((unsigned)i0 < (unsigned)size.p[0]);
    CV_DbgAssert((unsigned)(i1 * DataType<_Tp>::channels) < (unsigned)(size.p[1] * channels()));
    CV_DbgAssert(CV_ELEM_SIZE1(traits::Depth<_Tp>::value) == elemSize1());
    return ((_Tp*)(data + step.p[0] * i0))[i1];
}

template<typename _Tp> inline
const _Tp& Mat::at(int i0, int i1) const
{
    CV_DbgAssert(dims <= 2);
    CV_DbgAssert(data);
    CV_DbgAssert((unsigned)i0 < (unsigned)size.p[0]);
    CV_DbgAssert((unsigned)(i1 * DataType<_Tp>::channels) < (unsigned)(size.p[1] * channels()));
    CV_DbgAssert(CV_ELEM_SIZE1(traits::Depth<_Tp>::value) == elemSize1());
    return ((const _Tp*)(data + step.p[0] * i0))[i1];
}

template<typename _Tp> inline
_Tp& Mat::at(Point pt)
{
    CV_DbgAssert(dims <= 2);
    CV_DbgAssert(data);
    CV_DbgAssert((unsigned)pt.y < (unsigned)size.p[0]);
    CV_DbgAssert((unsigned)(pt.x * DataType<_Tp>::channels) < (unsigned)(size.p[1] * channels()));
    CV_DbgAssert(CV_ELEM_SIZE1(traits::Depth<_Tp>::value) == elemSize1());
    return ((_Tp*)(data + step.p[0] * pt.y))[pt.x];
}

template<typename _Tp> inline
const _Tp& Mat::at(Point pt) const
{
    CV_DbgAssert(dims <= 2);
    CV_DbgAssert(data);
    CV_DbgAssert((unsigned)pt.y < (unsigned)size.p[0]);
    CV_DbgAssert((unsigned)(pt.x * DataType<_Tp>::channels) < (unsigned)(size.p[1] * channels()));
    CV_DbgAssert(CV_ELEM_SIZE1(traits::Depth<_Tp>::value) == elemSize1());
    return ((const _Tp*)(data + step.p[0] * pt.y))[pt.x];
}

template<typename _Tp> inline
_Tp& Mat::at(int i0)
{
    CV_DbgAssert(dims <= 2);
    CV_DbgAssert(data);
    CV_DbgAssert((unsigned)i0 < (unsigned)(size.p[0] * size.p[1]));
    CV_DbgAssert(elemSize() == sizeof(_Tp));
    if( isContinuous() || size.p[0] == 1 )
        return ((_Tp*)data)[i0];
    if( size.p[1] == 1 )
        return *(_Tp*)(data + step.p[0] * i0);
    int i = i0 / cols, j = i0 - i * cols;
    return ((_Tp*)(data + step.p[0] * i))[j];
}

template<typename _Tp> inline
const _Tp& Mat::at(int i0) const
{
    CV_DbgAssert(dims <= 2);
    CV_DbgAssert(data);
    CV_DbgAssert((unsigned)i0 < (unsigned)(size.p[0] * size.p[1]));
    CV_DbgAssert(elemSize() == sizeof(_Tp));
    if( isContinuous() || size.p[0] == 1 )
        return ((const _Tp*)data)[i0];
    if( size.p[1] == 1 )
        return *(const _Tp*)(data + step.p[0] * i0);
    int i = i0 / cols, j = i0 - i * cols;
    return ((const _Tp*)(data + step.p[0] * i))[j];
}

template<typename _Tp> inline
_Tp& Mat::at(int i0, int i1, int i2)
{
    CV_DbgAssert( elemSize() == sizeof(_Tp) );
    return *(_Tp*)ptr(i0, i1, i2);
}

template<typename _Tp> inline
const _Tp& Mat::at(int i0, int i1, int i2) const
{
    CV_DbgAssert( elemSize() == sizeof(_Tp) );
    return *(const _Tp*)ptr(i0, i1, i2);
}

template<typename _Tp> inline
_Tp& Mat::at(const int* idx)
{
    CV_DbgAssert( elemSize() == sizeof(_Tp) );
    return *(_Tp*)ptr(idx);
}

template<typename _Tp> inline
const _Tp& Mat::at(const int* idx) const
{
    CV_DbgAssert( elemSize() == sizeof(_Tp) );
    return *(const _Tp*)ptr(idx);
}

template<typename _Tp, int n> inline
_Tp& Mat::at(const Vec<int, n>& idx)
{
    CV_DbgAssert( elemSize() == sizeof(_Tp) );
    return *(_Tp*)ptr(idx.val);
}

template<typename _Tp, int n> inline
const _Tp& Mat::at(const Vec<int, n>& idx) const
{
    CV_DbgAssert( elemSize() == sizeof(_Tp) );
    return *(const _Tp*)ptr(idx.val);
}

template<typename _Tp> inline
MatConstIterator_<_Tp> Mat::begin() const
{
    CV_DbgAssert( elemSize() == sizeof(_Tp) );
    return MatConstIterator_<_Tp>((const Mat_<_Tp>*)this);
}

template<typename _Tp> inline
MatConstIterator_<_Tp> Mat::end() const
{
    CV_DbgAssert( elemSize() == sizeof(_Tp) );
    MatConstIterator_<_Tp> it((const Mat_<_Tp>*)this);
    it += total();
    return it;
}

template<typename _Tp> inline
MatIterator_<_Tp> Mat::begin()
{
    CV_DbgAssert( elemSize() == sizeof(_Tp) );
    return MatIterator_<_Tp>((Mat_<_Tp>*)this);
}

template<typename _Tp> inline
MatIterator_<_Tp> Mat::end()
{
    CV_DbgAssert( elemSize() == sizeof(_Tp) );
    MatIterator_<_Tp> it((Mat_<_Tp>*)this);
    it += total();
    return it;
}

template<typename _Tp, typename Functor> inline
void Mat::forEach(const Functor& operation) {
    this->forEach_impl<_Tp>(operation);
}

template<typename _Tp, typename Functor> inline
void Mat::forEach(const Functor& operation) const {
    // call as not const
    (const_cast<Mat*>(this))->forEach<_Tp>(operation);
}

template<typename _Tp> inline
Mat::operator std::vector<_Tp>() const
{
    std::vector<_Tp> v;
    copyTo(v);
    return v;
}

#ifdef CV_CXX_STD_ARRAY
template<typename _Tp, std::size_t _Nm> inline
Mat::operator std::array<_Tp, _Nm>() const
{
    std::array<_Tp, _Nm> v;
    copyTo(v);
    return v;
}
#endif

template<typename _Tp, int n> inline
Mat::operator Vec<_Tp, n>() const
{
    CV_Assert( data && dims <= 2 && (rows == 1 || cols == 1) &&
               rows + cols - 1 == n && channels() == 1 );

    if( isContinuous() && type() == traits::Type<_Tp>::value )
        return Vec<_Tp, n>((_Tp*)data);
    Vec<_Tp, n> v;
    Mat tmp(rows, cols, traits::Type<_Tp>::value, v.val);
    convertTo(tmp, tmp.type());
    return v;
}

template<typename _Tp, int m, int n> inline
Mat::operator Matx<_Tp, m, n>() const
{
    CV_Assert( data && dims <= 2 && rows == m && cols == n && channels() == 1 );

    if( isContinuous() && type() == traits::Type<_Tp>::value )
        return Matx<_Tp, m, n>((_Tp*)data);
    Matx<_Tp, m, n> mtx;
    Mat tmp(rows, cols, traits::Type<_Tp>::value, mtx.val);
    convertTo(tmp, tmp.type());
    return mtx;
}

template<typename _Tp> inline
void Mat::push_back(const _Tp& elem)
{
    if( !data )
    {
        *this = Mat(1, 1, traits::Type<_Tp>::value, (void*)&elem).clone();
        return;
    }
    CV_Assert(traits::Type<_Tp>::value == type() && cols == 1
              /* && dims == 2 (cols == 1 implies dims == 2) */);
    const uchar* tmp = dataend + step[0];
    if( !isSubmatrix() && isContinuous() && tmp <= datalimit )
    {
        *(_Tp*)(data + (size.p[0]++) * step.p[0]) = elem;
        dataend = tmp;
    }
    else
        push_back_(&elem);
}

template<typename _Tp> inline
void Mat::push_back(const Mat_<_Tp>& m)
{
    push_back((const Mat&)m);
}

template<> inline
void Mat::push_back(const MatExpr& expr)
{