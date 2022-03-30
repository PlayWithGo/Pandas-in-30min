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

#ifndef OPENCV_FLANN_KDTREE_INDEX_H_
#define OPENCV_FLANN_KDTREE_INDEX_H_

#include <algorithm>
#include <map>
#include <cassert>
#include <cstring>

#include "general.h"
#include "nn_index.h"
#include "dynamic_bitset.h"
#include "matrix.h"
#include "result_set.h"
#include "heap.h"
#include "allocator.h"
#include "random.h"
#include "saving.h"


namespace cvflann
{

struct KDTreeIndexParams : public IndexParams
{
    KDTreeIndexParams(int trees = 4)
    {
        (*this)["algorithm"] = FLANN_INDEX_KDTREE;
        (*this)["trees"] = trees;
    }
};


/**
 * Randomized kd-tree index
 *
 * Contains the k-d trees and other information for indexing a set of points
 * for nearest-neighbor matching.
 */
template <typename Distance>
class KDTreeIndex : public NNIndex<Distance>
{
public:
    typedef typename Distance::ElementType ElementType;
    typedef typename Distance::ResultType DistanceType;


    /**
     * KDTree constructor
     *
     * Params:
     *          inputData = dataset with the input features
     *          params = parameters passed to the kdtree algorithm
     */
    KDTreeIndex(const Matrix<ElementType>& inputData, const IndexParams& params = KDTreeIndexParams(),
                Distance d = Distance() ) :
        dataset_(inputData), index_params_(params), distance_(d)
    {
        size_ = dataset_.rows;
        veclen_ = dataset_.cols;

        trees_ = get_param(index_params_,"trees",4);
        tree_roots_ = new NodePtr[trees_];

        // Create a permutable array of indices to the input vectors.
        vind_.resize(size_);
        for (size_t i = 0; i < size_; ++i) {
            vind_[i] = int(i);
        }

        mean_ = new DistanceType[veclen_];
        var_ = new DistanceType[veclen_];
    }


    KDTreeIndex(const KDTreeIndex&);
    KDTreeIndex& operator=(const KDTreeIndex&);

    /**
     * Standard destructor
     */
    ~KDTreeIndex()
    {
        if (tree_roots_!=NULL) {
            delete[] tree_roots_;
        }
        delete[] mean_;
        delete[] var_;
    }

    /**
     * Builds the index
     */
    void buildIndex()
    {
        /* Construct the randomized trees. */
        for (int i = 0; i < trees_; i++) {
            /* Randomize the order of vectors to allow for unbiased sampling. */
#ifndef OPENCV_FLANN_USE_STD_RAND
            cv::randShuffle(vind_);
#else
            std::random_shuffle(vind_.begin(), vind_.end());
#endif

            tree_roots_[i] = divideTree(&vind_[0], int(size_) );
        }
    }


    flann_algorithm_t getType() const
    {
        return FLANN_INDEX_KDTREE;
    }


    void saveIndex(FILE* stream)
    {
        save_value(stream, trees_);
        for (int i=0; i<trees_; ++i) {
            save_tree(stream, tree_roots_[i]);
        }
    }



    void loadIndex(FILE* stream)
    {
        load_value(stream, trees_);
        if (tree_roots_!=NULL) {
            delete[] tree_roots_;
        }
        tree_roots_ = new NodePtr[trees_];
        for (int i=0; i<trees_; ++i) {
            load_tree(stream,tree_roots_[i]);
        }

        index_params_["algorithm"] = getType();
        index_params_["trees"] = tree_roots_;
    }

    /**
     *  Returns size of index.
     */
    size_t size() const
    {
        return size_;
    }

    /**
     * Returns the length of an index feature.
     */
    size_t veclen() const
    {
        return veclen_;
    }

    /**
     * Computes the inde memory usage
     * Returns: memory used by the index
     */
    int usedMemory() const
    {
        return int(pool_.usedMemory+pool_.wastedMemory+dataset_.rows*sizeof(int));  // pool memory and vind array memory
    }

    /**
     * Find set of nearest neighbors to vec. Their indices are stored inside
     * the result object.
     *
     * Params:
     *     result 