/***********************************************************************
 * Software License Agreement (BSD License)
 *
 * Copyright 2008-2011  Marius Muja (mariusm@cs.ubc.ca). All rights reserved.
 * Copyright 2008-2011  David G. Lowe (lowe@cs.ubc.ca). All rights reserved.
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

#ifndef OPENCV_FLANN_HIERARCHICAL_CLUSTERING_INDEX_H_
#define OPENCV_FLANN_HIERARCHICAL_CLUSTERING_INDEX_H_

#include <algorithm>
#include <map>
#include <cassert>
#include <limits>
#include <cmath>

#include "general.h"
#include "nn_index.h"
#include "dist.h"
#include "matrix.h"
#include "result_set.h"
#include "heap.h"
#include "allocator.h"
#include "random.h"
#include "saving.h"


namespace cvflann
{

struct HierarchicalClusteringIndexParams : public IndexParams
{
    HierarchicalClusteringIndexParams(int branching = 32,
                                      flann_centers_init_t centers_init = FLANN_CENTERS_RANDOM,
                                      int trees = 4, int leaf_size = 100)
    {
        (*this)["algorithm"] = FLANN_INDEX_HIERARCHICAL;
        // The branching factor used in the hierarchical clustering
        (*this)["branching"] = branching;
        // Algorithm used for picking the initial cluster centers
        (*this)["centers_init"] = centers_init;
        // number of parallel trees to build
        (*this)["trees"] = trees;
        // maximum leaf size
        (*this)["leaf_size"] = leaf_size;
    }
};


/**
 * Hierarchical index
 *
 * Contains a tree constructed through a hierarchical clustering
 * and other information for indexing a set of points for nearest-neighbour matching.
 */
template <typename Distance>
class HierarchicalClusteringIndex : public NNIndex<Distance>
{
public:
    typedef typename Distance::ElementType ElementType;
    typedef typename Distance::ResultType DistanceType;

private:


    typedef void (HierarchicalClusteringIndex::* centersAlgFunction)(int, int*, int, int*, int&);

    /**
     * The function used for choosing the cluster centers.
     */
    centersAlgFunction chooseCenters;



    /**
     * Chooses the initial centers in the k-means clustering in a random manner.
     *
     * Params:
     *     k = number of centers
     *     vecs = the dataset of points
     *     indices = indices in the dataset
     *     indices_length = length of indices vector
     *
     */
    void chooseCentersRandom(int k, int* dsindices, int indices_length, int* centers, int& centers_length)
    {
        UniqueRandom r(indices_length);

        int index;
        for (index=0; index<k; ++index) {
            bool duplicate = true;
            int rnd;
            while (duplicate) {
                duplicate = false;
                rnd = r.next();
                if (rnd<0) {
                    centers_length = index;
                    return;
                }

                centers[index] = dsindices[rnd];

                for (int j=0; j<index; ++j) {
                    DistanceType sq = distance(dataset[centers[index]], dataset[centers[j]], dataset.cols);
                    if (sq<1e-16) {
                        duplicate = true;
                    }
                }
            }
        }

        centers_length = index;
    }


    /**
     * Chooses the initial centers in the k-means using Gonzales' algorithm
     * so that the centers are spaced apart from each other.
     *
     * Params:
     *     k = number of centers
     *     vecs = the dataset of points
     *     indices = indices in the dataset
     * Returns:
     */
    void chooseCentersGonzales(int k, int* dsindices, int indices_length, int* centers, int& centers_length)
    {
        int n = indices_length;

        int rnd = rand_int(n);
        assert(rnd >=0 && rnd < n);

        centers[0] = dsindices[rnd];

        int index;
        for (index=1; index<k; ++index) {

            int best_index = -1;
            DistanceType best_val = 0;
            for (int j=0; j<n; ++j) {
                DistanceType dist = distance(dataset[centers[0]],dataset[dsindices[j]],dataset.cols);
                for (int i=1; i<index; ++i) {
                    DistanceType tmp_dist = distance(dataset[centers[i]],dataset[dsindices[j]],dataset.cols);
                    if (tmp_dist<dist) {
                        dist = tmp_dist;
                    }
                }
                if (dist>best_val) {
                    best_val = dist;
                    best_index = j;
                }
            }
            if (best_index!=-1) {
                centers[index] = dsindices[best_index];
            }
            else {
                break;
            }
        }
        centers_length = index;
    }


    /**
     * Chooses the initial centers in the k-means using the algorithm
     * proposed in the KMeans++ paper:
     * Arthur, David; Vassilvitskii, Sergei - k-means++: The Advantages of Careful Seeding
     *
     * Implementation of this function was converted from the one provided in Arthur's code.
     *
     * Params:
     *     k = number of centers
     *     vecs = the dataset of points
     *     indices = indices in the dataset
     * Returns:
     */
    void chooseCentersKMeanspp(int k, int* dsindices, int indices_length, int* centers, int& centers_length)
    {
        int n = indices_length;

        double currentPot = 0;
        DistanceType* closestDistSq = new DistanceType[n];

        // Choose one random center and set the closestDistSq values
        int index = rand_int(n);
        assert(index >=0 && index < n);
        centers[0] = dsindices[index];

        // Computing distance^2 will have the advantage of even higher probability further to pick new centers
        // far from previous centers (and this complies to "k-means++: the advantages of careful seeding" article)
        for (int i = 0; i < n; i++) {
            closestDistSq[i] = distance(dataset[dsindices[i]], dataset[dsindices[index]], dataset.cols);
            closestDistSq[i] = ensureSquareDistance<Distance>( closestDistSq[i] );
            currentPot += closestDistSq[i];
        }


        const int numLocalTries = 1;

        // Choose each center
        int centerCount;
        for (centerCount = 1; centerCount < k; centerCount++) {

            // Repeat several trials
            double bestNewPot = -1;
            int bestNewIndex = 0;
            for (int localTrial = 0; localTrial < numLocalTries; localTrial++) {

                // Choose our center - have to be slightly careful to return a valid answer even accounting
                // for possible rounding errors
                double randVal = rand_double(currentPot);
                for (index = 0; index < n-1; index++) {
                    if (randVal <= closestDistSq[index]) break;
                    else randVal -= closestDistSq[index];
                }

                // Compute the new potential
                double newPot = 0;
                for (int i = 0; i < n; i++) {
                    DistanceType dist = distance(dataset[dsindices[i]], dataset[dsindices[index]], dataset.cols);
                    newPot += std::min( ensureSquareDistance<Distance>(dist), closestDistSq[i] );
                }

                // Store the best result
                if ((bestNewPot < 0)||(newPot < bestNewPot)) {
                    bestNewPot = newPot;
                    bestNewIndex = index;
                }
            }

            // Add the appropriate center
            centers[centerCount] = dsindices[bestNewIndex];
            currentPot = bestNewPot;
            for (int i = 0; i < n; i++) {
                DistanceType dist = distance(dataset[dsindices[i]], dataset[dsindices[bestNewIndex]], dataset.cols);
                closestDistSq[i] = std::min( ensureSquareDistance<Distance>(dist), closestDistSq[i] );
            }
        }

        centers_length = centerCount;

        delete[] closestDistSq;
    }


    /**
     * Chooses the initial centers in a way inspired by Gonzales (by Pierre-Emmanuel Viel):
     * select the first point of the list as a candidate, then parse the points list. If another
     * point is further than current candidate from the other centers, test if it is a good center
     * of a local aggregation. If it is, replace current candidate by this point. And so on...
     *
     * Used with KMeansIndex that computes centers coordinates by averaging positions of clusters points,
     * this doesn't make a real difference with previous methods. But used with HierarchicalClusteringIndex
     * class that pick centers among existing points instead of computing the barycenters, there is a real
     * improvement.
     *
     * Params:
     *     k = number of centers
     *     vecs = the dataset of points
     *     indices = indices in the dataset
     * Returns:
     */
    void GroupWiseCenterChooser(int k, int* dsindices, int indices_length, int* centers, int& centers_length)
    {
        const float kSpeedUpFactor = 1.3f;

        int n = indices_length;

        DistanceType* closestDistSq = new DistanceType[n];

        // Choose one random center and set the closestDistSq values
        int index = rand_int(n);
        assert(index >=0 && index < n);
        centers[0] = dsindices[index];

        for (int i = 0; i < n; i++) {
            closestDistSq[i] = distance(dataset[dsindices[i]], dataset[dsindices[index]], dataset.cols);
        }


        // Choose each center
        int centerCount;
        for (centerCount = 1; centerCount < k; centerCount++) {

            // Repeat several trials
            double bestNewPot = -1;
            int bestNewIndex = 0;
            DistanceType furthest = 0;
            for (index = 0; index < n; index++) {

                // We will test only the potential of the points further than current candidate
                if( closestDistSq[index] > kSpeedUpFactor * (float)furthest ) {

                    // Compute the new potential
                    double newPot = 0;
                    for (int i = 0; i < n; i++) {
                        newPot += std::min( distance(dataset[dsindices[i]], dataset[dsindices[index]], dataset.cols)
                                            , closestDistSq[i] );
                    }

                    // Store the best result
                    if ((bestNewPot < 0)||(newPot <= bestNewPot)) {
                        bestNewPot = newPot;
                        bestNewIndex = index;
                        furthest = closestDistSq[index];
                    }
                }
            }

            // Add the appropriate center
            centers[centerCount] = dsindices[bestNewIndex];
            for (int i = 0; i < n; i++) {
                closestDistSq[i] = std::min( distance(dataset[dsindices[i]], dataset[dsindices[bestNewIndex]], dataset.cols)
                                             , closestDistSq[i] );
            }
        }

        centers_length = centerCount;

        delete[] closestDistSq;
    }


public:


    /**
     * Index constructor
     *
     * Params:
     *          inputData = dataset with the input features
     *          params = parameters passed to the hierarchical k-means algorithm
     */
    HierarchicalClusteringIndex(const Matrix<ElementType>& inputData, const IndexParams& index_params = HierarchicalClusteringIndexParams(),
                                Distance d = Distance())
        : dataset(inputData), params(index_params), root(NULL), indices(NULL), distance(d)
    {
        memoryCounter = 0;

        size_ = dataset.rows;
        veclen_ = dataset.cols;

        branching_ = get_param(params,"branching",32);
        centers_init_ = get_param(params,"centers_init", FLANN_CENTERS_RANDOM);
        trees_ = get_param(params,"trees",4);
        leaf_size_ = get_param(params,"leaf_size",100);

        if (centers_init_==FLANN_CENTERS_RANDOM) {
            chooseCenters = &HierarchicalClusteringIndex::chooseCentersRandom;
        }
        else if (centers_init_==FLANN_CENTERS_GONZALES) {
            chooseCenters = &HierarchicalClusteringIndex::chooseCentersGonzales;
        }
        else if (centers_init_==FLANN_CENTERS_KMEANSPP) {
            chooseCenters = &HierarchicalClusteringIndex::chooseCentersKMeanspp;
        }
        else if (centers_init_==FLANN_CENTERS_GROUPWISE) {
            chooseCenters = &HierarchicalClusteringIndex::GroupWiseCenterChooser;
        }
        else {
            throw FLANNException("Unknown algorithm for choosing initial centers.");
        }

        trees_ = get_param(params,"trees",4);
        root = new NodePtr[trees_];
        indices = new int*[trees_];

        for (int i=0; i<trees_; ++i) {
            root[i] = NULL;
            indices[i] = NULL;
        }
    }

    HierarchicalClusteringIndex(const HierarchicalClusteringIndex&);
    HierarchicalClusteringIndex& operator=(const HierarchicalClusteringIndex&);

    /**
     * Index destructor.
     *
     * Release the memory used by the index.
     */
    virtual ~HierarchicalClusteringIndex()
    {
        free_elements();

        if (root!=NULL) {
            delete[] root;
        }

        if (indices!=NULL) {
            delete[] indices;
        }
    }


    /**
     * Release the inner elements of indices[]
     */
    void free_elements()
    {
        if (indices!=NULL) {
            for(int i=0; i<trees_; ++i) {
                if (indices[i]!=NULL) {
                    delete[] indices[i];
                    indices[i] = NULL;
                }
            }
        }
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
        return pool.usedMemory+pool.wastedMemory+memoryCounter;
    }

    /**
     * Builds the index
     */
    void buildIndex()
    {
        if (branching_<2) {
            throw FLANNException("Branching factor must be at least 2");
        }

        free_elements();

        for (int i=0; i<trees_; ++i) {
            indices[i] = new int[size_];
            for (size_t j=0; j<size_; ++j) {
                indices[i][j] = (int)j;
            }
            root[i] = pool.allocate<Node>();
            computeClustering(root[i], indices[i], (int)size_, branching_,0);
        }
    }


    flann_algorithm_t getType() const
    {
        return FLANN_INDEX_HIERARCHICAL;
    }


    void saveIndex(FILE* stream)
    {
        save_value(stream, branching_);
        save_value(stream, trees_);
        save_value(stream, centers_init_);
        save_value(stream, leaf_size_);
        save_value(stream, memoryCounter);
        for (int i=0; i<trees_; ++i) {
            save_value(stream, *indices[i], size_);
            save_tree(stream, root[i], i);
        }

    }


    void loadIndex(FILE* stream)
    {
        free_elements();

        if (root!=NULL) {
            delete[] root;
        }

        if (indices!=NULL) {
            delete[] indices;
        }

        load_value(stream, branching_);
        load_value(stream, trees_);
        load_value(stream, centers_init_);
        load_value(stream, leaf_size_);
        load_value(stream, memoryCounter);

        indices = new int*[trees_];
        root = new NodePtr[trees_];
        for (int i=0; i<trees_; ++i) {
            indices[i] = new int[size_];
            load_value(stream, *indices[i], size_);
            load_tree(stream, root[i], i);
        }

        params