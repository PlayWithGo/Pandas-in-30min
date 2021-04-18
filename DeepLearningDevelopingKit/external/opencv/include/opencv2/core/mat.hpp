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

#ifndef OPENCV_CORE_MAT_HPP
#define OPENCV_CORE_MAT_HPP

#ifndef __cplusplus
#  error mat.hpp header must be compiled as C++
#endif

#include "opencv2/core/matx.hpp"
#include "opencv2/core/types.hpp"

#include "opencv2/core/bufferpool.hpp"

#ifdef CV_CXX11
#include <type_traits>
#endif

namespace cv
{

//! @addtogroup core_basic
//! @{

enum { ACCESS_READ=1<<24, ACCESS_WRITE=1<<25,
    ACCESS_RW=3<<24, ACCESS_MASK=ACCESS_RW, ACCESS_FAST=1<<26 };

CV__DEBUG_NS_BEGIN

class CV_EXPORTS _OutputArray;

//////////////////////// Input/Output Array Arguments /////////////////////////////////

/** @brief This is the proxy class for passing read-only input arrays into OpenCV functions.

It is defined as:
@code
    typedef const _InputArray& InputArray;
@endcode
where _InputArray is a class that can be constructed from `Mat`, `Mat_<T>`, `Matx<T, m, n>`,
`std::vector<T>`, `std::vector<std::vector<T> >`, `std::vector<Mat>`, `std::vector<Mat_<T> >`,
`UMat`, `std::vector<UMat>` or `double`. It can also be constructed from a matrix expression.

Since this is mostly implementation-level class, and its interface may change in future versions, we
do not describe it in details. There are a few key things, though, that should be kept in mind:

-   When you see in the reference manual or in OpenCV source code a function that takes
    InputArray, it means that you can actually pass `Mat`, `Matx`, `vector<T>` etc. (see above the
    complete list).
-   Optional input arguments: If some of the input arrays may be empty, pass cv::noArray() (or
    simply cv::Mat() as you probably did before).
-   The class is designed solely for passing parameters. That is, normally you *should not*
    declare class members, local and global variables of this type.
-   If you want to design your own function or a class method that can operate of arrays of
    multiple types, you can use InputArray (or OutputArray) for the respective parameters. Inside
    a function you should use _InputArray::getMat() method to construct a matrix header for the
    array (without copying data). _InputArray::kind() can be used to distinguish Mat from
    `vector<>` etc., but normally it is not needed.

Here is how you can use a function that takes InputArray :
@code
    std::vector<Point2f> vec;
    // points or a circle
    for( int i = 0; i < 30; i++ )
        vec.push_back(Point2f((float)(100 + 30*cos(i*CV_PI*2/5)),
                              (float)(100 - 30*sin(i*CV_PI*2/5))));
    cv::transform(vec, vec, cv::Matx23f(0.707, -0.707, 10, 0.707, 0.707, 20));
@endcode
That is, we form an STL vector containing points, and apply in-place affine transformation to the
vector using the 2x3 matrix created inline as `Matx<float, 2, 3>` instance.

Here is how such a function can be implemented (for simplicity, we implement a very specific case of
it, according to the assertion statement inside) :
@code
    void myAffineTransform(InputArray _src, OutputArray _dst, InputArray _m)
    {
        // get Mat headers for input arrays. This is O(1) operation,
        // unless _src and/or _m are matrix expressions.
        Mat src = _src.getMat(), m = _m.getMat();
        CV_Assert( src.type() == CV_32FC2 && m.type() == CV_32F && m.size() == Size(3, 2) );

        // [re]create the output array so that it has the proper size and type.
        // In case of Mat it calls Mat::create, in case of STL vector it calls vector::resize.
        _dst.create(src.size(), src.type());
        Mat dst = _dst.getMat();

        for( int i = 0; i < src.rows; i++ )
            for( int j = 0; j < src.cols; j++ )
            {
                Point2f pt = src.at<Point2f>(i, j);
                dst.at<Point2f>(i, j) = Point2f(m.at<float>(0, 0)*pt.x +
                                                m.at<float>(0, 1)*pt.y +
                                                m.at<float>(0, 2),
                                                m.at<float>(1, 0)*pt.x +
                                                m.at<float>(1, 1)*pt.y +
                                                m.at<float>(1, 2));
            }
    }
@endcode
There is another related type, InputArrayOfArrays, which is currently defined as a synonym for
InputArray:
@code
    typedef InputArray InputArrayOfArrays;
@endcode
It denotes function arguments that are either vectors of vectors or vectors of matrices. A separate
synonym is needed to generate Python/Java etc. wrappers properly. At the function implementation
level their use is similar, but _InputArray::getMat(idx) should be used to get header for the
idx-th component of the outer vector and _InputArray::size().area() should be used to find the
number of components (vectors/matrices) of the outer vector.
 */
class CV_EXPORTS _InputArray
{
public:
    enum {
        KIND_SHIFT = 16,
        FIXED_TYPE = 0x8000 << KIND_SHIFT,
        FIXED_SIZE = 0x4000 << KIND_SHIFT,
        KIND_MASK = 31 << KIND_SHIFT,

        NONE              = 0 << KIND_SHIFT,
        MAT               = 1 << KIND_SHIFT,
        MATX              = 2 << KIND_SHIFT,
        STD_VECTOR        = 3 << KIND_SHIFT,
        STD_VECTOR_VECTOR = 4 << KIND_SHIFT,
        STD_VECTOR_MAT    = 5 << KIND_SHIFT,
        EXPR              = 6 << KIND_SHIFT,
        OPENGL_BUFFER     = 7 << KIND_SHIFT,
        CUDA_HOST_MEM     = 8 << KIND_SHIFT,
        CUDA_GPU_MAT      = 9 << KIND_SHIFT,
        UMAT              =10 << KIND_SHIFT,
        STD_VECTOR_UMAT   =11 << KIND_SHIFT,
        STD_BOOL_VECTOR   =12 << KIND_SHIFT,
        STD_VECTOR_CUDA_GPU_MAT = 13 << KIND_SHIFT,
        STD_ARRAY         =14 << KIND_SHIFT,
        STD_ARRAY_MAT     =15 << KIND_SHIFT
    };

    _InputArray();
    _InputArray(int _flags, void* _obj);
    _InputArray(const Mat& m);
    _InputArray(const MatExpr& expr);
    _InputArray(const std::vector<Mat>& vec);
    template<typename _Tp> _InputArray(const Mat_<_Tp>& m);
    template<typename _Tp> _InputArray(const std::vector<_Tp>& vec);
    _InputArray(const std::vector<bool>& vec);
    template<typename _Tp> _InputArray(const std::vector<std::vector<_Tp> >& vec);
    _InputArray(const std::vector<std::vector<bool> >&);
    template<typename _Tp> _InputArray(const std::vector<Mat_<_Tp> >& vec);
    template<typename _Tp> _InputArray(const _Tp* vec, int n);
    template<typename _Tp, int m, int n> _InputArray(const Matx<_Tp, m, n>& matx);
    _InputArray(const double& val);
    _InputArray(const cuda::GpuMat& d_mat);
    _InputArray(const std::vector<cuda::GpuMat>& d_mat_array);
    _InputArray(const ogl::Buffer& buf);
    _InputArray(const cuda::HostMem& cuda_mem);
    template<typename _Tp> _InputArray(const cudev::GpuMat_<_Tp>& m);
    _InputArray(const UMat& um);
    _InputArray(const std::vector<UMat>& umv);

#ifdef CV_CXX_STD_ARRAY
    template<typename _Tp, std::size_t _Nm> _InputArray(const std::array<_Tp, _Nm>& arr);
    template<std::size_t _Nm> _InputArray(const std::array<Mat, _Nm>& arr);
#endif

    Mat getMat(int idx=-1) const;
    Mat getMat_(int idx=-1) const;
    UMat getUMat(int idx=-1) const;
    void getMatVector(std::vector<Mat>& mv) const;
    void getUMatVector(std::vector<UMat>& umv) const;
    void getGpuMatVector(std::vector<cuda::GpuMat>& gpumv) const;
    cuda::GpuMat getGpuMat() const;
    ogl::Buffer getOGlBuffer() const;

    int getFlags() const;
    void* getObj() const;
    Size getSz() const;

    int kind() const;
    int dims(int i=-1) const;
    int cols(int i=-1) const;
    int rows(int i=-1) const;
    Size size(int i=-1) const;
    int sizend(int* sz, int i=-1) const;
    bool sameSize(const _InputArray& arr) const;
    size_t total(int i=-1) const;
    int type(int i=-1) const;
    int depth(int i=-1) const;
    int channels(int i=-1) const;
    bool isContinuous(int i=-1) const;
    bool isSubmatrix(int i=-1) const;
    bool empty() const;
    void copyTo(const _OutputArray& arr) const;
    void copyTo(const _OutputArray& arr, const _InputArray & mask) const;
    size_t offset(int i=-1) const;
    size_t step(int i=-1) const;
    bool isMat() const;
    bool isUMat() const;
    bool isMatVector() const;
    bool isUMatVector() const;
    bool isMatx() const;
    bool isVector() const;
    bool isGpuMatVector() const;
    ~_InputArray();

protected:
    int flags;
    void* obj;
    Size sz;

    void init(int _flags, const void* _obj);
    void init(int _flags, const void* _obj, Size _sz);
};


/** @brief This type is very similar to InputArray except that it is used for input/output and output function
parameters.

Just like with InputArray, OpenCV users should not care about OutputArray, they just pass `Mat`,
`vector<T>` etc. to the functions. The same limitation as for `InputArray`: *Do not explicitly
create OutputArray instances* applies here too.

If you want to make your function polymorphic (i.e. accept different arrays as output parameters),
it is also not very difficult. Take the sample above as the reference. Note that
_OutputArray::create() needs to be called before _OutputArray::getMat(). This way you guarantee
that the output array is properly allocated.

Optional output parameters. If you do not need certain output array to be computed and returned to
you, pass cv::noArray(), just like you would in the case of optional input array. At the
implementation level, use _OutputArray::needed() to check if certain output array needs to be
computed or not.

There are several synonyms for OutputArray that are used to assist automatic Python/Java/... wrapper
generators:
@code
    typedef OutputArray OutputArrayOfArrays;
    typedef OutputArray InputOutputArray;
    typedef OutputArray InputOutputArrayOfArrays;
@endcode
 */
class CV_EXPORTS _OutputArray : public _InputArray
{
public:
    enum
    {
        DEPTH_MASK_8U = 1 << CV_8U,
        DEPTH_MASK_8S = 1 << CV_8S,
        DEPTH_MASK_16U = 1 << CV_16U,
        DEPTH_MASK_16S = 1 << CV_16S,
        DEPTH_MASK_32S = 1 << CV_32S,
        DEPTH_MASK_32F = 1 << CV_32F,
        DEPTH_MASK_64F = 1 << CV_64F,
        DEPTH_MASK_ALL = (DEPTH_MASK_64F<<1)-1,
        DEPTH_MASK_ALL_BUT_8S = DEPTH_MASK_ALL & ~DEPTH_MASK_8S,
        DEPTH_MASK_FLT = DEPTH_MASK_32F + DEPTH_MASK_64F
    };

    _OutputArray();
    _OutputArray(int _flags, void* _obj);
    _OutputArray(Mat& m);
    _OutputArray(std::vector<Mat>& vec);
    _OutputArray(cuda::GpuMat& d_mat);
    _OutputArray(std::vector<cuda::GpuMat>& d_mat);
    _OutputArray(ogl::Buffer& buf);
    _OutputArray(cuda::HostMem& cuda_mem);
    template<typename _Tp> _OutputArray(cudev::GpuMat_<_Tp>& m);
    template<typename _Tp> _OutputArray(std::vector<_Tp>& vec);
    _OutputArray(std::vector<bool>& vec);
    template<typename _Tp> _OutputArray(std::vector<std::vector<_Tp> >& vec);
    _OutputArray(std::vector<std::vector<bool> >&);
    template<typename _Tp> _OutputArray(std::vector<Mat_<_Tp> >& vec);
    template<typename _Tp> _OutputArray(Mat_<_Tp>& m);
    template<typename _Tp> _OutputArray(_Tp* vec, int n);
    template<typename _Tp, int m, int n> _OutputArray(Matx<_Tp, m, n>& matx);
    _OutputArray(UMat& m);
    _OutputArray(std::vector<UMat>& vec);

    _OutputArray(const Mat& m);
    _OutputArray(const std::vector<Mat>& vec);
    _OutputArray(const cuda::GpuMat& d_mat);
    _OutputArray(const std::vector<cuda::GpuMat>& d_mat);
    _OutputArray(const ogl::Buffer& buf);
    _OutputArray(const cuda::HostMem& cuda_mem);
    template<typename _Tp> _OutputArray(const cudev::GpuMat_<_Tp>& m);
    template<typename _Tp> _OutputArray(const std::vector<_Tp>& vec);
    template<typename _Tp> _OutputArray(const std::vector<std::vector<_Tp> >& vec);
    template<typename _Tp> _OutputArray(const std::vector<Mat_<_Tp> >& vec);
    template<typename _Tp> _OutputArray(const Mat_<_Tp>& m);
    template<typename _Tp> _OutputArray(const _Tp* vec, int n);
    template<typename _Tp, int m, int n> _OutputArray(const Matx<_Tp, m, n>& matx);
    _OutputArray(const UMat& m);
    _OutputArray(const std::vector<UMat>& vec);

#ifdef CV_CXX_STD_ARRAY
    template<typename _Tp, std::size_t _Nm> _OutputArray(std::array<_Tp, _Nm>& arr);
    template<typename _Tp, std::size_t _Nm> _OutputArray(const std::array<_Tp, _Nm>& arr);
    template<std::size_t _Nm> _OutputArray(std::array<Mat, _Nm>& arr);
    template<std::size_t _Nm> _OutputArray(const std::array<Mat, _Nm>& arr);
#endif

    bool fixedSize() const;
    bool fixedType() const;
    bool needed() const;
    Mat& getMatRef(int i=-1) const;
    UMat& getUMatRef(int i=-1) const;
    cuda::GpuMat& getGpuMatRef() const;
    std::vector<cuda::GpuMat>& getGpuMatVecRef() const;
    ogl::Buffer& getOGlBufferRef() const;
    cuda::HostMem& getHostMemRef() const;
    void create(Size sz, int type, int i=-1, bool allowTransposed=false, int fixedDepthMask=0) const;
    void create(int rows, int cols, int type, int i=-1, bool allowTransposed=false, int fixedDepthMask=0) const;
    void create(int dims, const int* size, int type, int i=-1, bool allowTransposed=false, int fixedDepthMask=0) const;
    void createSameSize(const _InputArray& arr, int mtype) const;
    void release() const;
    void clear() const;
    void setTo(const _InputArray& value, const _InputArray & mask = _InputArray()) const;

    void assign(const UMat& u) const;
    void assign(const Mat& m) const;

    void assign(const std::vector<UMat>& v) const;
    void assign(const std::vector<Mat>& v) const;
};


class CV_EXPORTS _InputOutputArray : public _OutputArray
{
public:
    _InputOutputArray();
    _InputOutputArray(int _flags, void* _obj);
    _InputOutputArray(Mat& m);
    _InputOutputArray(std::vector<Mat>& vec);
    _InputOutputArray(cuda::GpuMat& d_mat);
    _InputOutputArray(ogl::Buffer& buf);
    _InputOutputArray(cuda::HostMem& cuda_mem);
    template<typename _Tp> _InputOutputArray(cudev::GpuMat_<_Tp>& m);
    template<typename _Tp> _InputOutputArray(std::vector<_Tp>& vec);
    _InputOutputArray(std::vector<bool>& vec);
    template<typename _Tp> _InputOutputArray(std::vector<std::vector<_Tp> >& vec);
    template<typename _Tp> _InputOutputArray(std::vector<Mat_<_Tp> >& vec);
    template<typename _Tp> _InputOutputArray(Mat_<_Tp>& m);
    template<typename _Tp> _InputOutputArray(_Tp* vec, int n);
    template<typename _Tp, int m, int n> _InputOutputArray(Matx<_Tp, m, n>& matx);
    _InputOutputArray(UMat& m);
    _InputOutputArray(std::vector<UMat>& vec);

    _InputOutputArray(const Mat& m);
    _InputOutputArray(const std::vector<Mat>& vec);
    _InputOutputArray(const cuda::GpuMat& d_mat);
    _InputOutputArray(const std::vector<cuda::GpuMat>& d_mat);
    _InputOutputArray(const ogl::Buffer& buf);
    _InputOutputArray(const cuda::HostMem& cuda_mem);
    template<typename _Tp> _InputOutputArray(const cudev::GpuMat_<_Tp>& m);
    template<typename _Tp> _InputOutputArray(const std::vector<_Tp>& vec);
    template<typename _Tp> _InputOutputArray(const std::vector<std::vector<_Tp> >& vec);
    template<typename _Tp> _InputOutputArray(const std::vector<Mat_<_Tp> >& vec);
    template<typename _Tp> _InputOutputArray(const Mat_<_Tp>& m);
    template<typename _Tp> _InputOutputArray(const _Tp* vec, int n);
    template<typename _Tp, int m, int n> _InputOutputArray(const Matx<_Tp, m, n>& matx);
    _InputOutputArray(const UMat& m);
    _InputOutputArray(const std::vector<UMat>& vec);

#ifdef CV_CXX_STD_ARRAY
    template<typename _Tp, std::size_t _Nm> _InputOutputArray(std::array<_Tp, _Nm>& arr);
    template<typename _Tp, std::size_t _Nm> _InputOutputArray(const std::array<_Tp, _Nm>& arr);
    template<std::size_t _Nm> _InputOutputArray(std::array<Mat, _Nm>& arr);
    template<std::size_t _Nm> _InputOutputArray(const std::array<Mat, _Nm>& arr);
#endif

};

CV__DEBUG_NS_END

typedef const _InputArray& InputArray;
typedef InputArray InputArrayOfArrays;
typedef const _OutputArray& OutputArray;
typedef OutputArray OutputArrayOfArrays;
typedef const _InputOutputArray& InputOutputArray;
typedef InputOutputArray InputOutputArrayOfArrays;

CV_EXPORTS InputOutputArray noArray();

/////////////////////////////////// MatAllocator //////////////////////////////////////

//! Usage flags for allocator
enum UMatUsageFlags
{
    USAGE_DEFAULT = 0,

    // buffer allocation policy is platform and usage specific
    USAGE_ALLOCATE_HOST_MEMORY = 1 << 0,
    USAGE_ALLOCATE_DEVICE_MEMORY = 1 << 1,
    USAGE_ALLOCATE_SHARED_MEMORY = 1 << 2, // It is not equal to: USAGE_ALLOCATE_HOST_MEMORY | USAGE_ALLOCATE_DEVICE_MEMORY

    __UMAT_USAGE_FLAGS_32BIT = 0x7fffffff // Binary compatibility hint
};

struct CV_EXPORTS UMatData;

/** @brief  Custom array allocator
*/
class CV_EXPORTS MatAllocator
{
public:
    MatAllocator() {}
    virtual ~MatAllocator() {}

    // let's comment it off for now to detect and fix all the uses of allocator
    //virtual void allocate(int dims, const int* sizes, int type, int*& refcount,
    //                      uchar*& datastart, uchar*& data, size_t* step) = 0;
    //virtual void deallocate(int* refcount, uchar* datastart, uchar* data) = 0;
    virtual UMatData* allocate(int dims, const int* sizes, int type,
                               void* data, size_t* step, int flags, UMatUsageFlags usageFlags) const = 0;
    virtual bool allocate(UMatData* data, int accessflags, UMatUsageFlags usageFlags) const = 0;
    virtual void deallocate(UMatData* data) const = 0;
    virtual void map(UMatData* data, int accessflags) const;
    virtual void unmap(UMatData* data) const;
    virtual void download(UMatData* data, void* dst, int dims, const size_t sz[],
                          const size_t srcofs[], const size_t srcstep[],
                          const size_t dststep[]) const;
    virtual void upload(UMatData* data, const void* src, int dims, const size_t sz[],
                        const size_t dstofs[], const size_t dststep[],
                        const size_t srcstep[]) const;
    virtual void copy(UMatData* srcdata, UMatData* dstdata, int dims, const size_t sz[],
                      const size_t srcofs[], const size_t srcstep[],
                      const size_t dstofs[], const size_t dststep[], bool sync) const;

    // default implementation returns DummyBufferPoolController
    virtual BufferPoolController* getBufferPoolController(const char* id = NULL) const;
};


//////////////////////////////// MatCommaInitializer //////////////////////////////////

/** @brief  Comma-separated Matrix Initializer

 The class instances are usually not created explicitly.
 Instead, they are created on "matrix << firstValue" operator.

 The sample below initializes 2x2 rotation matrix:

 \code
 double angle = 30, a = cos(angle*CV_PI/180), b = sin(angle*CV_PI/180);
 Mat R = (Mat_<double>(2,2) << a, -b, b, a);
 \endcode
*/
template<typename _Tp> class MatCommaInitializer_
{
public:
    //! the constructor, created by "matrix << firstValue" operator, where matrix is cv::Mat
    MatCommaInitializer_(Mat_<_Tp>* _m);
    //! the operator that takes the next value and put it to the matrix
    template<typename T2> MatCommaInitializer_<_Tp>& operator , (T2 v);
    //! another form of conversion operator
    operator Mat_<_Tp>() const;
protected:
    MatIterator_<_Tp> it;
};


/////////////////////////////////////// Mat ///////////////////////////////////////////

// note that umatdata might be allocated together
// with the matrix data, not as a separate object.
// therefore, it does not have constructor or destructor;
// it should be explicitly initialized using init().
struct CV_EXPORTS UMatData
{
    enum { COPY_ON_MAP=1, HOST_COPY_OBSOLETE=2,
        DEVICE_COPY_OBSOLETE=4, TEMP_UMAT=8, TEMP_COPIED_UMAT=24,
        USER_ALLOCATED=32, DEVICE_MEM_MAPPED=64,
        ASYNC_CLEANUP=128
    };
    UMatData(const MatAllocator* allocator);
    ~UMatData();

    // provide atomic access to the structure
    void lock();
    void unlock();

    bool hostCopyObsolete() const;
    bool deviceCopyObsolete() const;
    bool deviceMemMapped() const;
    bool copyOnMap() const;
    bool tempUMat() const;
    bool tempCopiedUMat() const;
    void markHostCopyObsolete(bool flag);
    void markDeviceCopyObsolete(bool flag);
    void markDeviceMemMapped(bool flag);

    const MatAllocator* prevAllocator;
    const MatAllocator* currAllocator;
    int urefcount;
    int refcount;
    uchar* data;
    uchar* origdata;
    size_t size;

    int flags;
    void* handle;
    void* userdata;
    int allocatorFlags_;
    int mapcount;
    UMatData* originalUMatData;
};


struct CV_EXPORTS MatSize
{
    explicit MatSize(int* _p);
    Size operator()() const;
    const int& operator[](int i) const;
    int& operator[](int i);
    operator const int*() const;
    bool operator == (const MatSize& sz) const;
    bool operator != (const MatSize& sz) const;

    int* p;
};

struct CV_EXPORTS MatStep
{
    MatStep();
    explicit MatStep(size_t s);
    const size_t& operator[](int i) const;
    size_t& operator[](int i);
    operator size_t() const;
    MatStep& operator = (size_t s);

    size_t* p;
    size_t buf[2];
protected:
    MatStep& operator = (const MatStep&);
};

/** @example cout_mat.cpp
An example demonstrating the serial out capabilities of cv::Mat
*/

 /** @brief n-dimensional dense array class \anchor CVMat_Details

The class Mat represents an n-dimensional dense numerical single-channel or multi-channel array. It
can be used to store real or complex-valued vectors and matrices, grayscale or color images, voxel
volumes, vector fields, point clouds, tensors, histograms (though, very high-dimensional histograms
may be better stored in a SparseMat ). The data layout of the array `M` is defined by the array
`M.step[]`, so that the address of element \f$(i_0,...,i_{M.dims-1})\f$, where \f$0\leq i_k<M.size[k]\f$, is
computed as:
\f[addr(M_{i_0,...,i_{M.dims-1}}) = M.data + M.step[0]*i_0 + M.step[1]*i_1 + ... + M.step[M.dims-1]*i_{M.dims-1}\f]
In case of a 2-dimensional array, the above formula is reduced to:
\f[addr(M_{i,j}) = M.data + M.step[0]*i + M.step[1]*j\f]
Note that `M.step[i] >= M.step[i+1]` (in fact, `M.step[i] >= M.step[i+1]*M.size[i+1]` ). This means
that 2-dimensional matrices are stored row-by-row, 3-dimensional matrices are stored plane-by-plane,
and so on. M.step[M.dims-1] is minimal and always equal to the element size M.elemSize() .

So, the data layout in Mat is fully compatible with CvMat, IplImage, and CvMatND types from OpenCV
1.x. It is also compatible with the majority of dense array types from the standard toolkits and
SDKs, such as Numpy (ndarray), Win32 (independent device bitmaps), and others, that is, with any
array that uses *steps* (or *strides*) to compute the position of a pixel. Due to this
compatibility, it is possible to make a Mat header for user-allocated data and process it in-place
using OpenCV functions.

There are many different ways to create a Mat object. The most popular options are listed below:

- Use the create(nrows, ncols, type) method or the similar Mat(nrows, ncols, type[, fillValue])
constructor. A new array of the specified size and type is allocated. type has the same meaning as
in the cvCreateMat method. For example, CV_8UC1 means a 8-bit single-channel array, CV_32FC2
means a 2-channel (complex) floating-point array, and so on.
@code
    // make a 7x7 complex matrix filled with 1+3j.
    Mat M(7,7,CV_32FC2,Scalar(1,3));
    // and now turn M to a 100x60 15-channel 8-bit matrix.
    // The old content will be deallocated
    M.create(100,60,CV_8UC(15));
@endcode
As noted in the introduction to this chapter, create() allocates only a new array when the shape
or type of the current array are different from the specified ones.

- Create a multi-dimensional array:
@code
    // create a 100x100x100 8-bit array
    int sz[] = {100, 100, 100};
    Mat bigCube(3, sz, CV_8U, Scalar::all(0));
@endcode
It passes the number of dimensions =1 to the Mat constructor but the created array will be
2-dimensional with the number of columns set to 1. So, Mat::dims is always \>= 2 (can also be 0
when the array is empty).

- Use a copy constructor or assignment operator where there can be an array or expression on the
right side (see below). As noted in the introduction, the array assignment is an O(1) operation
because it only copies the header and increases the reference counter. The Mat::clone() method can
be used to get a full (deep) copy of the array when you need it.

- Construct a header for a part of another array. It can be a single row, single column, several
rows, several columns, rectangular region in the array (called a *minor* in algebra) or a
diagonal. Such operations are also O(1) because the new header references the same data. You can
actually modify a part of the array using this feature, for example:
@code
    // add the 5-th row, multiplied by 3 to the 3rd row
    M.row(3) = M.row(3) + M.row(5)*3;
    // now copy the 7-th column to the 1-st column
    // M.col(1) = M.col(7); // this will not work
    Mat M1 = M.col(1);
    M.col(7).copyTo(M1);
    // create a new 320x240 image
    Mat img(Size(320,240),CV_8UC3);
    // select a ROI
    Mat roi(img, Rect(10,10,100,100));
    // fill the ROI with (0,255,0) (which is green in RGB space);
    // the original 320x240 image will be modified
    roi = Scalar(0,255,0);
@endcode
Due to the additional datastart and dataend members, it is possible to compute a relative
sub-array position in the main *container* array using locateROI():
@code
    Mat A = Mat::eye(10, 10, CV_32S);
    // extracts A columns, 1 (inclusive) to 3 (exclusive).
    Mat B = A(Range::all(), Range(1, 3));
    // extracts B rows, 5 (inclusive) to 9 (exclusive).
    // that is, C \~ A(Range(5, 9), Range(1, 3))
    Mat C = B(Range(5, 9), Range::all());
    Size size; Point ofs;
    C.locateROI(size, ofs);
    // size will be (width=10,height=10) and the ofs will be (x=1, y=5)
@endcode
As in case of whole matrices, if you need a deep copy, use the `clone()` method of the extracted
sub-matrices.

- Make a header for user-allocated data. It can be useful to do the following:
    -# Process "foreign" data using OpenCV (for example, when you implement a DirectShow\* filter or
    a processing module for gstreamer, and so on). For example:
    @code
        void process_video_frame(const unsigned char* pixels,
                                 int width, int height, int step)
        {
            Mat img(height, width, CV_8UC3, pixels, step);
            GaussianBlur(img, img, Size(7,7), 1.5, 1.5);
        }
    @endcode
    -# Quickly initialize small matrices and/or get a super-fast element access.
    @code
        double m[3][3] = {{a, b, c}, {d, e, f}, {g, h, i}};
        Mat M = Mat(3, 3, CV_64F, m).inv();
    @endcode
    .
    Partial yet very common cases of this *user-allocated data* case are conversions from CvMat and
    IplImage to Mat. For this purpose, there is function cv::cvarrToMat taking pointers to CvMat or
    IplImage and the optional flag indicating whether to copy the data or not.
    @snippet samples/cpp/image.cpp iplimage

- Use MATLAB-style array initializers, zeros(), ones(), eye(), for example:
@code
    // create a double-precision identity matrix and add it to M.
    M += Mat::eye(M.rows, M.cols, CV_64F);
@endcode

- Use a comma-separated initializer:
@code
    // create a 3x3 double-precision identity matrix
    Mat M = (Mat_<double>(3,3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
@endcode
With this approach, you first call a constructor of the Mat class with the proper parameters, and
then you just put `<< operator` followed by comma-separated values that can be constants,
variables, expressions, and so on. Also, note the extra parentheses required to avoid compilation
errors.

Once the array is created, it is automatically managed via a reference-counting mechanism. If the
array header is built on top of user-allocated data, you should handle the data by yourself. The
array data is deallocated when no one points to it. If you want to release the data pointed by a
array header before the array destructor is called, use Mat::release().

The next important thing to learn about the array class is element access. This manual already
described how to compute an address of each array element. Normally, you are not required to use the
formula directly in the code. If you know the array element type (which can be retrieved using the
method Mat::type() ), you can access the element \f$M_{ij}\f$ of a 2-dimensional array as:
@code
    M.at<double>(i,j) += 1.f;
@endcode
assuming that `M` is a double-precision floating-point array. There are several variants of the method
at for a different number of dimensions.

If you need to process a whole row of a 2D array, the most efficient way is to get the pointer to
the row first, and then just use the plain C operator [] :
@code
    // compute sum of positive matrix elements
    // (assuming that M is a double-precision matrix)
    double sum=0;
    for(int i = 0; i < M.rows; i++)
    {
        const double* Mi = M.ptr<double>(i);
        for(int j = 0; j < M.cols; j++)
            sum += std::max(Mi[j], 0.);
    }
@endcode
Some operations, like the one above, do not actually depend on the array shape. They just process
elements of an array one by one (or elements from multiple arrays that have the same coordinates,
for example, array addition). Such operations are called *element-wise*. It makes sense to check
whether all the input/output arrays are continuous, namely, have no gaps at the end of each row. If
yes, process them as a long single row:
@code
    // compute the sum of positive matrix elements, optimized variant
    double sum=0;
    int cols = M.cols, rows = M.rows;
    if(M.isContinuous())
    {
        cols *= rows;
        rows = 1;
    }
    for(int i = 0; i < rows; i++)
    {
        const double* Mi = M.ptr<double>(i);
        for(int j = 0; j < cols; j++)
            sum += std::max(Mi[j], 0.);
    }
@endcode
In case of the continuous matrix, the outer loop body is executed just once. So, the overhead is
smaller, which is especially noticeable in case of small matrices.

Finally, there are STL-style iterators that are smart enough to skip gaps between successive rows:
@code
    // compute sum of positive matrix elements, iterator-based variant
    double sum=0;
    MatConstIterator_<double> it = M.begin<double>(), it_end = M.end<double>();
    for(; it != it_end; ++it)
        sum += std::max(*it, 0.);
@endcode
The matrix iterators are random-access iterators, so they can be passed to any STL algorithm,
including std::sort().

@note Matrix Expressions and arithmetic see MatExpr
*/
class CV_EXPORTS Mat
{
public:
    /**
    These are various constructors that form a matrix. As noted in the AutomaticAllocation, often
    the default constructor is enough, and the proper matrix will be allocated by an OpenCV function.
    The constructed matrix can further be assigned to another matrix or matrix expression or can be
    allocated with Mat::create . In the former case, the old content is de-referenced.
     */
    Mat();

    /** @overload
    @param rows Number of rows in a 2D array.
    @param cols Number of columns in a 2D array.
    @param type Array type. Use CV_8UC1, ..., CV_64FC4 to create 1-4 channel matrices, or
    CV_8UC(n), ..., CV_64FC(n) to create multi-channel (up to CV_CN_MAX channels) matrices.
    */
    Mat(int rows, int cols, int type);

    /** @overload
    @param size 2D array size: Size(cols, rows) . In the Size() constructor, the number of rows and the
    number of columns go in the reverse order.
    @param type Array type. Use CV_8UC1, ..., CV_64FC4 to create 1-4 channel matrices, or
    CV_8UC(n), ..., CV_64FC(n) to create multi-channel (up to CV_CN_MAX channels) matrices.
      */
    Mat(Size size, int type);

    /** @overload
    @param rows Number of rows in a 2D array.
    @param cols Number of columns in a 2D array.
    @param type Array type. Use CV_8UC1, ..., CV_64FC4 to create 1-4 channel matrices, or
    CV_8UC(n), ..., CV_64FC(n) to create multi-channel (up to CV_CN_MAX channels) matrices.
    @param s An optional value to initialize each matrix element with. To set all the matrix elements to
    the particular value after the construction, use the assignment operator
    Mat::operator=(const Scalar& value) .
    */
    Mat(int rows, int cols, int type, const Scalar& s);

    /** @overload
    @param size 2D array size: Size(cols, rows) . In the Size() constructor, the number of rows and the
    number of columns go in the reverse order.
    @param type Array type. Use CV_8UC1, ..., CV_64FC4 to create 1-4 channel matrices, or
    CV_8UC(n), ..., CV_64FC(n) to create multi-channel (up to CV_CN_MAX channels) matrices.
    @param s An optional value to initialize each matrix element with. To set all the matrix elements to
    the particular value after the construction, use the assignment operator
    Mat::operator=(const Scalar& value) .
      */
    Mat(Size size, int type, const Scalar& s);

    /** @overload
    @param ndims Array dimensionality.
    @param sizes Array of integers specifying an n-dimensional array shape.
    @param type Array type. Use CV_8UC1, ..., CV_64FC4 to create 1-4 channel matrices, or
    CV_8UC(n), ..., CV_64FC(n) to create multi-channel (up to CV_CN_MAX channels) matrices.
    */
    Mat(int ndims, const int* sizes, int type);

    /** @overload
    @param sizes Array of integers specifying an n-dimensional array shape.
    @param type Array type. Use CV_8UC1, ..., CV_64FC4 to create 1-4 channel matrices, or
    CV_8UC(n), ..., CV_64FC(n) to create multi-channel (up to CV_CN_MAX channels) matrices.
    */
    Mat(const std::vector<int>& sizes, int type);

    /** @overload
    @param ndims Array dimensionality.
    @param sizes Array of integers specifying an n-dimensional array shape.
    @param type Array type. Use CV_8UC1, ..., CV_64FC4 to create 1-4 channel matrices, or
    CV_8UC(n), ..., CV_64FC(n) to create multi-channel (up to CV_CN_MAX channels) matrices.
    @param s An optional value to initialize each matrix element with. To set all the matrix elements to
    the particular value after the construction, use the assignment operator
    Mat::operator=(const Scalar& value) .
    */
    Mat(int ndims, const int* sizes, int type, const Scalar& s);

    /** @overload
    @param sizes Array of integers specifying an n-dimensional array shape.
    @param type Array type. Use CV_8UC1, ..., CV_64FC4 to create 1-4 channel matrices, or
    CV_8UC(n), ..., CV_64FC(n) to create multi-channel (up to CV_CN_MAX channels) matrices.
    @param s An optional value to initialize each matrix element with. To set all the matrix elements to
    the particular value after the construction, use the assignment operator
    Mat::operator=(const Scalar& value) .
    */
    Mat(const std::vector<int>& sizes, int type, const Scalar& s);


    /** @overload
    @param m Array that (as a whole or partly) is assigned to the constructed matrix. No data is copied
    by these constructors. Instead, the header pointing