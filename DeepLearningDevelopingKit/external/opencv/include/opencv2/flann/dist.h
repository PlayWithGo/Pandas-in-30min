/***********************************************************************
 * Software License Agreement (BSD License)
 *
 * Copyright 2008-2009  Marius Muja (mariusm@cs.ubc.ca). All rights reserved.
 * Copyright 2008-2009  David G. Lowe (lowe@cs.ubc.ca). All rights reserved.
 *
 * THE BSD LICENSE
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *************************************************************************/

#ifndef OPENCV_FLANN_DIST_H_
#define OPENCV_FLANN_DIST_H_

#include <cmath>
#include <cstdlib>
#include <string.h>
#ifdef _MSC_VER
typedef unsigned __int32 uint32_t;
typedef unsigned __int64 uint64_t;
#else
#include <stdint.h>
#endif

#include "defines.h"

#if defined _WIN32 && defined(_M_ARM)
# include <Intrin.h>
#endif

#ifdef __ARM_NEON__
# include "arm_neon.h"
#endif

namespace cvflann
{

template<typename T>
inline T abs(T x) { return (x<0) ? -x : x; }

template<>
inline int abs<int>(int x) { return ::abs(x); }

template<>
inline float abs<float>(float x) { return fabsf(x); }

template<>
inline double abs<double>(double x) { return fabs(x); }

template<typename T>
struct Accumulator { typedef T Type; };
template<>
struct Accumulator<unsigned char>  { typedef float Type; };
template<>
struct Accumulator<unsigned short> { typedef float Type; };
template<>
struct Accumulator<unsigned int> { typedef float Type; };
template<>
struct Accumulator<char>   { typedef float Type; };
template<>
struct Accumulator<short>  { typedef float Type; };
template<>
struct Accumulator<int> { typedef float Type; };

#undef True
#undef False

class True
{
};

class False
{
};


/**
 * Squared Euclidean distance functor.
 *
 * This is the simpler, unrolled version. This is preferable for
 * very low dimensionality data (eg 3D points)
 */
template<class T>
struct L2_Simple
{
    typedef True is_kdtree_distance;
    typedef True is_vector_space_distance;

    typedef T ElementType;
    typedef typename Accumulator<T>::Type ResultType;

    template <typename Iterator1, typename Iterator2>
    ResultType operator()(Iterator1 a, Iterator2 b, size_t size, ResultType /*worst_dist*/ = -1) const
    {
        ResultType result = ResultType();
        ResultType diff;
        for(size_t i = 0; i < size; ++i ) {
            diff = *a++ - *b++;
            result += diff*diff;
        }
        return result;
    }

    template <typename U, typename V>
    inline ResultType accum_dist(const U& a, const V& b, int) const
    {
        return (a-b)*(a-b);
    }
};



/**
 * Squared Euclidean distance functor, optimized version
 */
template<class T>
struct L2
{
    typedef True is_kdtree_distance;
    typedef True is_vector_space_distance;

    typedef T ElementType;
    typedef typename Accumulator<T>::Type ResultType;

    /**
     *  Compute the squared Euclidean distance between two vectors.
     *
     *	This is highly optimised, with loop unrolling, as it is one
     *	of the most expensive inner loops.
     *
     *	The computation of squared root at the end is omitted for
     *	efficiency.
     */
    template <typename Iterator1, typename Iterator2>
    ResultType operator()(Iterator1 a, Iterator2 b, size_t size, ResultType worst_dist = -1) const
    {
        ResultType result = ResultType();
        ResultType diff0, diff1, diff2, diff3;
        Iterator1 last = a + size;
        Iterator1 lastgroup = last - 3;

        /* Process 4 items with each loop for efficiency. */
        while (a < lastgroup) {
            diff0 = (ResultType)(a[0] - b[0]);
            diff1 = (ResultType)(a[1] - b[1]);
            diff2 = (ResultType)(a[2] - b[2]);
            diff3 = (ResultType)(a[3] - b[3]);
            result += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
            a += 4;
            b += 4;

            if ((worst_dist>0)&&(result>worst_dist)) {
                return result;
            }
        }
        /* Process last 0-3 pixels.  Not needed for standard vector lengths. */
        while (a < last) {
            diff0 = (ResultType)(*a++ - *b++);
            result += diff0 * diff0;
        }
        return result;
    }

    /**
     *	Partial euclidean distance, using just one dimension. This is used by the
     *	kd-tree when computing partial distances while traversing the tree.
     *
     *	Squared root is omitted for efficiency.
     */
    template <typename U, typename V>
    inline ResultType accum_dist(const U& a, const V& b, int) const
    {
        return (a-b)*(a-b);
    }
};


/*
 * Manhattan distance functor, optimized version
 */
template<class T>
struct L1
{
    typedef True is_kdtree_distance;
    typedef True is_vector_space_distance;

    typedef T ElementType;
    typedef typename Accumulator<T>::Type ResultType;

    /**
     *  Compute the Manhattan (L_1) distance between two vectors.
     *
     *	This is highly optimised, with loop unrolling, as it is one
     *	of the most expensive inner loops.
     */
    template <typename Iterator1, typename Iterator2>
    ResultType operator()(Iterator1 a, Iterator2 b, size_t size, ResultType worst_dist = -1) const
    {
        ResultType result = ResultType();
        ResultType diff0, diff1, diff2, diff3;
        Iterator1 last = a + size;
        Iterator1 lastgroup = last - 3;

        /* Process 4 items with each loop for efficiency. */
        while (a < lastgroup) {
            diff0 = (ResultType)abs(a[0] - b[0]);
            diff1 = (ResultType)abs(a[1] - b[1]);
            diff2 = (ResultType)abs(a[2] - b[2]);
            diff3 = (ResultType)abs(a[3] - b[3]);
            result += diff0 + diff1 + diff2 + diff3;
            a += 4;
            b += 4;

            if ((worst_dist>0)&&(result>worst_dist)) {
                return result;
            }
        }
        /* Process last 0-3 pixels.  Not needed for standard vector lengths. */
        while (a < last) {
            diff0 = (ResultType)abs(*a++ - *b++);
            result += diff0;
        }
        return result;
    }

    /**
     * Partial distance, used by the kd-tree.
     */
    template <typename U, typename V>
    inline ResultType accum_dist(const U& a, const V& b, int) const
    {
        return abs(a-b);
    }
};



template<class T>
struct MinkowskiDistance
{
    typedef True is_kdtree_distance;
    typedef True is_vector_space_distance;

    typedef T ElementType;
    typedef typename Accumulator<T>::Type ResultType;

    int order;

    MinkowskiDistance(int order_) : order(order_) {}

    /**
     *  Compute the Minkowsky (L_p) distance between two vectors.
     *
     *	This is highly optimised, with loop unrolling, as it is one
     *	of the most expensive inner loops.
     *
     *	The computation of squared root at the end is omitted for
     *	efficiency.
     */
    template <typename Iterator1, typename Iterator2>
    ResultType operator()(Iterator1 a, Iterator2 b, size_t size, ResultType worst_dist = -1) const
    {
        ResultType result = ResultType();
        ResultType diff0, diff1, diff2, diff3;
        Iterator1 last = a + size;
        Iterator1 lastgroup = last - 3;

        /* Process 4 items with each loop for efficiency. */
        while (a < lastgroup) {
            diff0 = (ResultType)abs(a[0] - b[0]);
            diff1 = (ResultType)abs(a[1] - b[1]);
            diff2 = (ResultType)abs(a[2] - b[2]);
            diff3 = (ResultType)abs(a[3] - b[3]);
            result += pow(diff0,order) + pow(diff1,order) + pow(diff2,order) + pow(diff3,order);
            a += 4;
            b += 4;

            if ((worst_dist>0)&&(result>worst_dist)) {
                return result;
            }
        }
        /* Process last 0-3 pixels.  Not needed for standard vector lengths. */
        while (a < last) {
            diff0 = (ResultType)abs(*a++ - *b++);
            result += pow(diff0,order);
        }
        return result;
    }

    /**
     * Partial distance, used by the kd-tree.
     */
    template <typename U, typename V>
    inline ResultType accum_dist(const U& a, const V& b, int) const
    {
        return pow(static_cast<ResultType>(abs(a-b)),order);
    }
};



template<class T>
struct MaxDistance
{
    typedef False is_kdtree_distance;
    typedef True is_vector_space_distance;

    typedef T ElementType;
    typedef typename Accumulator<T>::Type ResultType;

    /**
     *  Compute the max distance (L_infinity) between two vectors.
     *
     *  This distance is not a valid kdtree distance, it's not dimensionwise additive.
     */
    template <typename Iterator1, typename Iterator2>
    ResultType operator()(Iterator1 a, Iterator2 b, size_t size, ResultType worst_dist = -1) const
    {
        ResultType result = ResultType();
        ResultType diff0, diff1, diff2, diff3;
        Iterator1 last = a + size;
        Iterator1 lastgroup = last - 3;

        /* Process 4 items with each loop for efficiency. */
        while (a < lastgroup) {
            diff0 = abs(a[0] - b[0]);
            diff1 = abs(a[1] - b[1]);
            diff2 = abs(a[2] - b[2]);
            diff3 = abs(a[3] - b[3]);
            if (diff0>result) {result = diff0; }
            if (diff1>result) {result = diff1; }
            if (diff2>result) {result = diff2; }
            if (diff3>result) {result = diff3; }
            a += 4;
            b += 4;

            if ((worst_dist>0)&&(result>worst_dist)) {
                return result;
            }
        }
        /* Process last 0-3 pixels.  Not needed for standard vector lengths. */
        while (a < last) {
            diff0 = abs(*a++ - *b++);
            result = (diff0>result) ? diff0 : result;
        }
        return result;
    }

    /* This distance functor is not dimension-wise additive, which
     * makes it an invalid kd-tree distance, not implementing the accum_dist method */

};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Hamming distance functor - counts the bit differences between two strings - useful for the Brief descriptor
 * bit count of A exclusive XOR'ed with B
 */
struct HammingLUT
{
    typedef False is_kdtree_distance;
    typedef False is_vector_space_distance;

    typedef unsigned char ElementType;
    typedef int ResultType;

    /** this will count the bits in a ^ b
     */
    ResultType operator()(const unsigned char* a, const unsigned char* b, size_t size) const
    {
        static const uchar popCountTable[] =
        {
            0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
            1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
            1, 2, 2, 3, 2, 3, 3, 4, 2,