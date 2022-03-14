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

/***********************************************************************
 * Author: Vincent Rabaud
 *************************************************************************/

#ifndef OPENCV_FLANN_DYNAMIC_BITSET_H_
#define OPENCV_FLANN_DYNAMIC_BITSET_H_

#ifndef FLANN_USE_BOOST
#  define FLANN_USE_BOOST 0
#endif
//#define FLANN_USE_BOOST 1
#if FLANN_USE_BOOST
#include <boost/dynamic_bitset.hpp>
typedef boost::dynamic_bitset<> DynamicBitset;
#else

#include <limits.h>

#include "dist.h"

namespace cvflann {

/** Class re-implementing the boost version of it
 * This helps not depending on boost, it also does not do the bound checks
 * and has a way to reset a block for speed
 */
class DynamicBitset
{
public:
    /** default constructor
     */
    DynamicBitset() : size_(0)
    {
    }

    /** only constructor we use in our code
     * @param sz the size of the bitset (in bits)
     */
    DynamicBitset(size_t sz)
    {
        resize(sz);
        reset();
    }

    /** Sets all the bits to 0
     */
    void clear()
    {
        std::fill(bitset_.begin(), bitset_.end(), 0);
    }

    /** @brief checks if the bitset is empty
     * @return true if the bitset is empty
     */
    bool empty() const
    {
        return bitset_.empty();
    }

    /** set all the bits to 0
     */
    void reset()
    {
        std::fill(bitset_.begin(), bitset_.end(), 0);
    }

    /** @brief set one bit to 0
     * @param index
     */
    void reset(size_t index)
    {
        bitset_[index / cell_bit_size_] &= ~(size_t(1) << (index % cell_bit_size_));
    }

    /** @brief se