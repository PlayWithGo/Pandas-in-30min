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
     *     result = the result object in which the indices of the nearest-neighbors are stored
     *     vec = the vector for which to search the nearest neighbors
     *     maxCheck = the maximum number of restarts (in a best-bin-first manner)
     */
    void findNeighbors(ResultSet<DistanceType>& result, const ElementType* vec, const SearchParams& searchParams)
    {
        int maxChecks = get_param(searchParams,"checks", 32);
        float epsError = 1+get_param(searchParams,"eps",0.0f);

        if (maxChecks==FLANN_CHECKS_UNLIMITED) {
            getExactNeighbors(result, vec, epsError);
        }
        else {
            getNeighbors(result, vec, maxChecks, epsError);
        }
    }

    IndexParams getParameters() const
    {
        return index_params_;
    }

private:


    /*--------------------- Internal Data Structures --------------------------*/
    struct Node
    {
        /**
         * Dimension used for subdivision.
         */
        int divfeat;
        /**
         * The values used for subdivision.
         */
        DistanceType divval;
        /**
         * The child nodes.
         */
        Node* child1, * child2;
    };
    typedef Node* NodePtr;
    typedef BranchStruct<NodePtr, DistanceType> BranchSt;
    typedef BranchSt* Branch;



    void save_tree(FILE* stream, NodePtr tree)
    {
        save_value(stream, *tree);
        if (tree->child1!=NULL) {
            save_tree(stream, tree->child1);
        }
        if (tree->child2!=NULL) {
            save_tree(stream, tree->child2);
        }
    }


    void load_tree(FILE* stream, NodePtr& tree)
    {
        tree = pool_.allocate<Node>();
        load_value(stream, *tree);
        if (tree->child1!=NULL) {
            load_tree(stream, tree->child1);
        }
        if (tree->child2!=NULL) {
            load_tree(stream, tree->child2);
        }
    }


    /**
     * Create a tree node that subdivides the list of vecs from vind[first]
     * to vind[last].  The routine is called recursively on each sublist.
     * Place a pointer to this new tree node in the location pTree.
     *
     * Params: pTree = the new node to create
     *                  first = index of the first vector
     *                  last = index of the last vector
     */
    NodePtr divideTree(int* ind, int count)
    {
        NodePtr node = pool_.allocate<Node>(); // allocate memory

        /* If too few exemplars remain, then make this a leaf node. */
        if ( count == 1) {
            node->child1 = node->child2 = NULL;    /* Mark as leaf node. */
            node->divfeat = *ind;    /* Store index of this vec. */
        }
        else {
            int idx;
            int cutfeat;
            DistanceType cutval;
            meanSplit(ind, count, idx, cutfeat, cutval);

            node->divfeat = cutfeat;
            node->divval = cutval;
            node->child1 = divideTree(ind, idx);
            node->child2 = divideTree(ind+idx, count-idx);
        }

        return node;
    }


    /**
     * Choose which feature to use in order to subdivide this set of vectors.
     * Make a random choice among those with the highest variance, and use
     * its variance as the threshold value.
     */
    void meanSplit(int* ind, int count, int& index, int& cutfeat, DistanceType& cutval)
    {
        memset(mean_,0,veclen_*sizeof(DistanceType));
        memset(var_,0,veclen_*sizeof(DistanceType));

        /* Compute mean values.  Only the first SAMPLE_MEAN values need to be
            sampled to get a good estimate.
         */
        int cnt = std::min((int)SAMPLE_MEAN+1, count);
        for (int j = 0; j < cnt; ++j) {
            ElementType* v = dataset_[ind[j]];
            for (size_t k=0; k<veclen_; ++k) {
                mean_[k] += v[k];
            }
        }
        for (size_t k=0; k<veclen_; ++k) {
            mean_[k] /= cnt;
        }

        /* Compute variances (no need to divide by count). */
        for (int j = 0; j < cnt; ++j) {
            ElementType* v = dataset_[ind[j]];
            for (size_t k=0; k<veclen_; ++k) {
        