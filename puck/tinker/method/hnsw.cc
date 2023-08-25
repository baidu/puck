/**
 * Non-metric Space Library
 *
 * Main developers: Bilegsaikhan Naidan, Leonid Boytsov, Yury Malkov, Ben Frederickson, David Novak
 *
 * For the complete list of contributors and further details see:
 * https://github.com/nmslib/nmslib
 *
 * Copyright (c) 2013-2018
 *
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 */
/*
*
* A Hierarchical Navigable Small World (HNSW) approach.
*
* The main publication is (available on arxiv: http://arxiv.org/abs/1603.09320):
* "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs" by Yu. A. Malkov, D. A. Yashunin
* This code was contributed by Yu. A. Malkov. It also was used in tests from the paper.
*
*
*/

#include <cmath>
#include <iostream>
#include <memory>

#include "puck/tinker/portable_prefetch.h"
#include "puck/tinker/portable_simd.h"
//#include "knnquery.h"
#include "puck/tinker/method/hnsw.h"
#include "puck/tinker/method/hnsw_distfunc_opt_impl_inline.h"
//#include "ported_boost_progress.h"
//#include "rangequery.h"
#include "puck/tinker/space.h"
#include "puck/tinker/space/space_lp.h"
//#include "space/space_scalar.h"
#include "puck/tinker/thread_pool.h"
#include "puck/tinker/utils.h"
#include <stdlib.h>
#include <unistd.h>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <typeinfo>
#include <vector>

//#include "sort_arr_bi.h"
#define MERGE_BUFFER_ALGO_SWITCH_THRESHOLD 100

#define USE_BITSET_FOR_INDEXING 1
#define EXTEND_USE_EXTENDED_NEIGHB_AT_CONSTR (0) // 0 is faster build, 1 is faster search on clustered data

// For debug purposes we also implemented saving an index to a text file
#define USE_TEXT_REGULAR_INDEX (false)

#define TOTAL_QTY       "TOTAL_QTY"
#define MAX_LEVEL       "MAX_LEVEL"
#define ENTER_POINT_ID  "ENTER_POINT_ID"
#define FIELD_M         "M"
#define FIELD_MAX_M     "MAX_M"
#define FIELD_MAX_M0    "MAX_M0"
#define CURR_LEVEL      "CURR_LEVEL"

#define EXTRA_MEM_PAD_SIZE 64

namespace similarity {

int                                                 defaultRandomSeed = 0;
thread_local std::unique_ptr<RandomGeneratorType>   randomGen;


size_t Object::data_length_ = 0;

float
NegativeDotProduct(const float* pVect1, const float* pVect2, size_t& qty, float* __restrict TmpRes) {
    return -ScalarProduct(pVect1, pVect2, qty, TmpRes);
}

/*
 * Important note: This function is applicable only when both vectors are normalized!
 */
float
NormCosine(const float* pVect1, const float* pVect2, size_t& qty, float* __restrict TmpRes) {
    return std::max(0.0f, 1 - std::max(float(-1), std::min(float(1), ScalarProduct(pVect1, pVect2, qty,
                                       TmpRes))));
}

EfficientDistFunc getDistFunc(DistFuncType funcType) {
    switch (funcType) {
    case DIST_TYPE_L2SQR16EXT :
        return L2Sqr16Ext;

    case DIST_TYPE_L2SQREXT   :
        return L2SqrExt;

    case DIST_TYPE_NORMCOSINE :
        return NormCosine;

    case DIST_TYPE_UNKNOWN :
        return nullptr;
        //case kNegativeDotProduct : return NegativeDotProduct;
        //case kL1Norm : return L1NormWrapper;
        //case kLInfNorm : return LInfNormWrapper;
    }

    return nullptr;
}



// This is the counter to keep the size of neighborhood information (for one node)
// TODO Can this one overflow? I really doubt
typedef uint32_t SIZEMASS_TYPE;

using namespace std;

template <typename dist_t>
Hnsw<dist_t>::Hnsw(const Space<dist_t>& space, const ObjectVector& data)
    : Index<dist_t>(data)
    , space_(space)
    //, PrintProgress_(PrintProgress)
    , visitedlistpool(nullptr)
    , enterpoint_(nullptr)
    , data_level0_memory_(nullptr)
      //, linkLists_(nullptr)
    , fstdistfunc_(nullptr) {
        objectDataOffset_ = ID_SIZE + LABEL_SIZE + DATALENGTH_SIZE;
}

void
checkList1(vector<HnswNode*> list) {
    int ok = 1;

    for (size_t i = 0; i < list.size(); i++) {
        for (size_t j = 0; j < list[i]->allFriends_[0].size(); j++) {
            for (size_t k = j + 1; k < list[i]->allFriends_[0].size(); k++) {
                if (list[i]->allFriends_[0][j] == list[i]->allFriends_[0][k]) {
                    cout << "\nDuplicate links\n\n\n\n\n!!!!!";
                    ok = 0;
                }
            }

            if (list[i]->allFriends_[0][j] == list[i]) {
                cout << "\nLink to the same element\n\n\n\n\n!!!!!";
                ok = 0;
            }
        }
    }

    if (ok) {
        cout << "\nOK\n";
    } else {
        cout << "\nNOT OK!!!\n";
    }

    return;
}

void
getDegreeDistr(string filename, vector<HnswNode*> list) {
    ofstream out(filename);
    size_t maxdegree = 0;

    for (HnswNode* node : list) {
        if (node->allFriends_[0].size() > maxdegree) {
            maxdegree = node->allFriends_[0].size();
        }
    }

    vector<int> distrin = vector<int>(1000);
    vector<int> distrout = vector<int>(1000);
    vector<int> inconnections = vector<int>(list.size());
    vector<int> outconnections = vector<int>(list.size());

    for (size_t i = 0; i < list.size(); i++) {
        for (HnswNode* node : list[i]->allFriends_[0]) {
            outconnections[list[i]->getId()]++;
            inconnections[node->getId()]++;
        }
    }

    for (size_t i = 0; i < list.size(); i++) {
        distrin[inconnections[i]]++;
        distrout[outconnections[i]]++;
    }

    for (size_t i = 0; i < distrin.size(); i++) {
        out << i << "\t" << distrin[i] << "\t" << distrout[i] << "\n";
    }

    out.close();
    return;
}

template <typename dist_t>
void
Hnsw<dist_t>::CreateIndex(const AnyParams& IndexParams) {
    AnyParamManager pmgr(IndexParams);

    pmgr.GetParamOptional("M", M_, 16);

    // Let's use a generic algorithm by default!
    pmgr.GetParamOptional(
        "searchMethod", searchMethod_,
        0); // this is just to prevent terminating the program when searchMethod is specified
    searchMethod_ = 0;

    indexThreadQty_ = 64;//std::thread::hardware_concurrency();

    pmgr.GetParamOptional("indexThreadQty", indexThreadQty_, indexThreadQty_);
    // indexThreadQty_ = 1;
    pmgr.GetParamOptional("efConstruction", efConstruction_, 200);
    pmgr.GetParamOptional("maxM", maxM_, M_);
    pmgr.GetParamOptional("maxM0", maxM0_, M_ * 2);
    pmgr.GetParamOptional("mult", mult_, 1 / log(1.0 * M_));
    pmgr.GetParamOptional("delaunay_type", delaunay_type_, 2);
    int post_;
    pmgr.GetParamOptional("post", post_, 0);
    int skip_optimized_index = 0;
    pmgr.GetParamOptional("skip_optimized_index", skip_optimized_index, 0);

    LOG(INFO) << "M                   = " << M_;
    LOG(INFO) << "indexThreadQty      = " << indexThreadQty_;
    LOG(INFO) << "efConstruction      = " << efConstruction_;
    LOG(INFO) << "maxM			          = " << maxM_;
    LOG(INFO) << "maxM0			          = " << maxM0_;

    LOG(INFO) << "mult                = " << mult_;
    LOG(INFO) << "skip_optimized_index= " << skip_optimized_index;
    LOG(INFO) << "delaunay_type       = " << delaunay_type_;

    SetQueryTimeParams(getEmptyParams());
    LOG(INFO) << "this->data_.empty() = " << this->data_.empty();

    if (this->data_.empty()) {
        ////pmgr.CheckUnused();
        return;
    }

    ElList_.resize(this->data_.size());
    // One entry should be added before all the threads are started, or else add() will not work properly
    HnswNode* first = new HnswNode(this->data_[0], 0 /* id == 0 */);
    first->init(getRandomLevel(mult_), maxM_, maxM0_);
    maxlevel_ = first->level;
    enterpoint_ = first;
    ElList_[0] = first;
    visitedlistpool = new VisitedListPool(indexThreadQty_, this->data_.size());

    ParallelFor(1, this->data_.size(), indexThreadQty_, [&](int id, int threadId) {
        HnswNode* node = new HnswNode(this->data_[id], id);
        if (id % 1000000 == 0){
            LOG(INFO) << "adding "<<id <<" / "<<this->data_.size() <<" in threadId = "<<threadId;
        }
        add(&space_, node);
        {
            unique_lock<mutex> lock(ElListGuard_);
            ElList_[id] = node;

        }
    });

    data_level0_memory_ = NULL;
    //linkLists_ = NULL;

    enterpointId_ = enterpoint_->getId();

    if (skip_optimized_index) {
        LOG(INFO) << "searchMethod			  = " << searchMethod_;
        //pmgr.CheckUnused();
        return;
    }

    int friendsSectionSize = (maxM0_ + 1) * sizeof(int);

    // Checking for maximum size of the datasection:
    size_t dataSectionSize = 1;

    for (size_t i = 0; i < ElList_.size(); i++) {
        if (ElList_[i]->getData()->bufferlength() > dataSectionSize) {
            dataSectionSize = ElList_[i]->getData()->bufferlength();
            vectorlength_ = ElList_[i]->getData()->datalength();
        }
    }

    // Selecting custom made functions
    dist_func_type_ = DIST_TYPE_UNKNOWN;

    // Although we removed double, let's keep this check here
    CHECK(sizeof(dist_t) == 4);


    const SpaceLp<dist_t>* pLpSpace = dynamic_cast<const SpaceLp<dist_t>*>(&space_);

    fstdistfunc_ = nullptr;
    iscosine_ = false;
    searchMethod_ = 3; // The same for all "optimized" indices

    if (pLpSpace != nullptr) {
        if (pLpSpace->getP() == 2) {
            LOG(INFO) << "\nThe space is Euclidean";
            //vectorlength_ = ((dataSectionSize - 16) >> 2);
            LOG(INFO) << "Vector length=" << vectorlength_;

            if (vectorlength_ % 16 == 0) {
                LOG(INFO) << "Thus using an optimised function for base 16";
                dist_func_type_ = DIST_TYPE_L2SQR16EXT;
            } else {
                LOG(INFO) << "Thus using function with any base";
                dist_func_type_ = DIST_TYPE_L2SQREXT;
            }
        }

        //else if (pLpSpace->getP() == 1) {
        //    dist_func_type_ = kL1Norm;
        //}
        //else if (pLpSpace->getP() == -1) {
        //    dist_func_type_ = kLInfNorm;
        //}
    }

    /*
    else if (dynamic_cast<const SpaceCosineSimilarity<dist_t>*>(&space_) != nullptr) {
        LOG(INFO) << "\nThe vector space is " << space_.StrDesc();
        vectorlength_ = ((dataSectionSize - 16) >> 2);
        LOG(INFO) << "Vector length=" << vectorlength_;
        dist_func_type_ = NORMCOSINE;
    } else if (dynamic_cast<const SpaceNegativeScalarProduct<dist_t>*>(&space_) != nullptr) {
        LOG(INFO) << "\nThe space is " << SPACE_NEGATIVE_SCALAR;
        vectorlength_ = ((dataSectionSize - 16) >> 2);
        LOG(INFO) << "Vector length=" << vectorlength_;
        dist_func_type_ = kNegativeDotProduct;
    }*/
    LOG(INFO) << "dist_func_type: " << dist_func_type_;
    fstdistfunc_ = getDistFunc(dist_func_type_);
    iscosine_ = (dist_func_type_ == DIST_TYPE_NORMCOSINE);

    if (fstdistfunc_ == nullptr) {
        //LOG(INFO) << "No appropriate custom distance function for " << space_.StrDesc();
        searchMethod_ = 0;
        LOG(INFO) << "searchMethod			  = " << searchMethod_;
        //pmgr.CheckUnused();
        return; // No optimized index
    }

    CHECK(dist_func_type_ != DIST_TYPE_UNKNOWN);

    //pmgr.CheckUnused();
    LOG(INFO) << "searchMethod			  = " << searchMethod_;
    //第0层每个样本需要的存储空间
    memoryPerObject_ = dataSectionSize + friendsSectionSize + 1; // 非0层的offset
    //uint64_t total_level = 0;

    //for (size_t i = 0; i < ElList_.size(); i++) {
    //    total_level += ElList_[i]->level;
    //}

    //uint64_t memory_per_node_higher_level = sizeof(int) * (1 + maxM_);
    //uint64_t higher_level_size = total_level * memory_per_node_higher_level;

    //size_t total_memory_allocated = (memoryPerObject_ * ElList_.size());
    // we allocate a few extra bytes to prevent prefetch from accessing out of range memory
    //data_level0_memory_ = (char*)malloc((memoryPerObject_ * ElList_.size()) + EXTRA_MEM_PAD_SIZE);
    //CHECK(data_level0_memory_);

    offsetLevel0_ = dataSectionSize;
    offsetData_ = 0;

    //memset(data_level0_memory_, 1, memoryPerObject_ * ElList_.size());
    //LOG(INFO) << "Making optimized index";

    //data_rearranged_.resize(ElList_.size());
    //for (long i = 0; i < ElList_.size(); i++) {
    //    ElList_[i]->copyDataAndLevel0LinksToOptIndex(
    //        data_level0_memory_ + (size_t)i * memoryPerObject_, offsetLevel0_, offsetData_);
        //data_rearranged_[i] = new Object(data_level0_memory_ + (i)*memoryPerObject_ + offsetData_);
    //};

    LOG(INFO) << "Finished making optimized index";

    LOG(INFO) << "Maximum level = " << enterpoint_->level;

    //LOG(INFO) << "Total memory allocated for optimized index+data: " << (total_memory_allocated >> 20) << " Mb";
}

template <typename dist_t>
void
Hnsw<dist_t>::SetQueryTimeParams(const AnyParams& QueryTimeParams) {
    AnyParamManager pmgr(QueryTimeParams);

    if (pmgr.hasParam("ef") && pmgr.hasParam("efSearch")) {
        throw runtime_error("The user shouldn't specify parameters ef and efSearch at the same time (they are synonyms)");
    }

    // ef and efSearch are going to be parameter-synonyms with the default value 20
    pmgr.GetParamOptional("ef", ef_, 20);
    pmgr.GetParamOptional("efSearch", ef_, ef_);

    int tmp;
    pmgr.GetParamOptional(
        "searchMethod", tmp, 0); // this is just to prevent terminating the program when searchMethod is specified

    string tmps;
    pmgr.GetParamOptional("algoType", tmps, "hybrid");
    ToLower(tmps);

    //if (tmps == "v1merge") {
    //    searchAlgoType_ = kV1Merge;
    //} else if (tmps == "old") {
    //    searchAlgoType_ = kOld;
    //} else if (tmps == "hybrid") {
    //    searchAlgoType_ = kHybrid;
    //} else {
    //    throw runtime_error("algoType should be one of the following: old, v1merge");
    //}

    //pmgr.CheckUnused();
    LOG(INFO) << "Set HNSW query-time parameters:";
    LOG(INFO) << "ef(Search)         =" << ef_;
    //LOG(INFO) << "algoType           =" << searchAlgoType_;
}

template <typename dist_t>
const std::string
Hnsw<dist_t>::StrDesc() const {
    return METH_HNSW;
}

template <typename dist_t> Hnsw<dist_t>::~Hnsw() {
    delete visitedlistpool;

    if (data_level0_memory_) {
        free(data_level0_memory_);
    }

    for (size_t i = 0; i < ElList_.size(); i++) {
        delete ElList_[i];
    }

    //for (const Object *p : data_rearranged_)
    //    delete p;
}

template <typename dist_t>
void
Hnsw<dist_t>::add(const Space<dist_t>* space, HnswNode* NewElement) {
    int curlevel = getRandomLevel(mult_);
    unique_lock<mutex>* lock = nullptr;

    if (curlevel > maxlevel_) {
        lock = new unique_lock<mutex>(MaxLevelGuard_);
    }

    NewElement->init(curlevel, maxM_, maxM0_);
    //LOG(INFO)<<NewElement->getId()<<"\t"<<NewElement->level;
    int maxlevelcopy = maxlevel_;
    HnswNode* ep = enterpoint_;

    if (curlevel < maxlevelcopy) {
        const Object* currObj = ep->getData();

        dist_t d = space->IndexTimeDistance(NewElement->getData(), currObj);
        dist_t curdist = d;
        HnswNode* curNode = ep;

        for (int level = maxlevelcopy; level > curlevel; level--) {
            bool changed = true;

            while (changed) {
                changed = false;
                unique_lock<mutex> lock(curNode->accessGuard_);
                const vector<HnswNode*>& neighbor = curNode->getAllFriends(level);
                int size = neighbor.size();

                for (int i = 0; i < size; i++) {
                    HnswNode* node = neighbor[i];
                    PREFETCH((char*)(node)->getData(), _MM_HINT_T0);
                }

                for (int i = 0; i < size; i++) {
                    currObj = (neighbor[i])->getData();
                    d = space->IndexTimeDistance(NewElement->getData(), currObj);

                    if (d < curdist) {
                        curdist = d;
                        curNode = neighbor[i];
                        changed = true;
                    }
                }
            }
        }

        ep = curNode;
    }

    for (int level = min(curlevel, maxlevelcopy); level >= 0; level--) {
        priority_queue<HnswNodeDistCloser<dist_t>> resultSet;
        kSearchElementsWithAttemptsLevel(space, NewElement->getData(), efConstruction_, resultSet, ep, level);

        switch (delaunay_type_) {
        case 0:
            while (resultSet.size() > M_) {
                resultSet.pop();
            }

            break;

        case 1:
            NewElement->getNeighborsByHeuristic1(resultSet, M_, space);
            break;

        case 2:
            NewElement->getNeighborsByHeuristic2(resultSet, M_, space, level);
            break;

        case 3:
            NewElement->getNeighborsByHeuristic3(resultSet, M_, space, level);
            break;
        }

        while (!resultSet.empty()) {
            ep = resultSet.top().getMSWNodeHier(); // memorizing the closest
            link(resultSet.top().getMSWNodeHier(), NewElement, level, space, delaunay_type_);
            resultSet.pop();
        }
    }

    if (curlevel > enterpoint_->level) {
        enterpoint_ = NewElement;
        maxlevel_ = curlevel;
    }

    if (lock != nullptr) {
        delete lock;
    }
}

template <typename dist_t>
void
Hnsw<dist_t>::kSearchElementsWithAttemptsLevel(const Space<dist_t>* space, const Object* queryObj,
        size_t efConstruction,
        priority_queue<HnswNodeDistCloser<dist_t>>& resultSet, HnswNode* ep,
        int level) const {
#if EXTEND_USE_EXTENDED_NEIGHB_AT_CONSTR != 0
    priority_queue<HnswNodeDistCloser<dist_t>> fullResultSet;
#endif

#if USE_BITSET_FOR_INDEXING
    VisitedList* vl = visitedlistpool->getFreeVisitedList();
    vl_type* mass = vl->mass;
    vl_type curV = vl->curV;
#else
    unordered_set<HnswNode*> visited;
#endif
    HnswNode* provider = ep;
    priority_queue<HnswNodeDistFarther<dist_t>> candidateSet;
    dist_t d = space->IndexTimeDistance(queryObj, provider->getData());
    HnswNodeDistFarther<dist_t> ev(d, provider);

    candidateSet.push(ev);
    resultSet.emplace(d, provider);

#if EXTEND_USE_EXTENDED_NEIGHB_AT_CONSTR != 0
    fullResultSet.emplace(d, provider);
#endif

#if USE_BITSET_FOR_INDEXING
    size_t nodeId = provider->getId();
    mass[nodeId] = curV;
#else
    visited.insert(provider);
#endif

    while (!candidateSet.empty()) {
        const HnswNodeDistFarther<dist_t>& currEv = candidateSet.top();
        dist_t lowerBound = resultSet.top().getDistance();

        /*
        * Check if we reached a local minimum.
        */
        if (currEv.getDistance() > lowerBound) {
            break;
        }

        HnswNode* currNode = currEv.getMSWNodeHier();

        /*
        * This lock protects currNode from being modified
        * while we are accessing elements of currNode.
        */
        unique_lock<mutex> lock(currNode->accessGuard_);
        const vector<HnswNode*>& neighbor = currNode->getAllFriends(level);

        // Can't access curEv anymore! The reference would become invalid
        candidateSet.pop();

        // calculate distance to each neighbor
        for (auto iter = neighbor.begin(); iter != neighbor.end(); ++iter) {
            PREFETCH((char*)(*iter)->getData(), _MM_HINT_T0);
        }

        for (auto iter = neighbor.begin(); iter != neighbor.end(); ++iter) {
#if USE_BITSET_FOR_INDEXING
            size_t nodeId = (*iter)->getId();

            if (mass[nodeId] != curV) {
                mass[nodeId] = curV;
#else

            if (visited.find((*iter)) == visited.end()) {
                visited.insert(*iter);
#endif
                d = space->IndexTimeDistance(queryObj, (*iter)->getData());
                HnswNodeDistFarther<dist_t> evE1(d, *iter);

#if EXTEND_USE_EXTENDED_NEIGHB_AT_CONSTR != 0
                fullResultSet.emplace(d, *iter);
#endif

                if (resultSet.size() < efConstruction || resultSet.top().getDistance() > d) {
                    resultSet.emplace(d, *iter);
                    candidateSet.push(evE1);

                    if (resultSet.size() > efConstruction) {
                        resultSet.pop();
                    }
                }
            }
        }
    }

#if EXTEND_USE_EXTENDED_NEIGHB_AT_CONSTR != 0
    resultSet.swap(fullResultSet);
#endif

#if USE_BITSET_FOR_INDEXING
    visitedlistpool->releaseVisitedList(vl);
#endif
}

template <typename dist_t>
void
Hnsw<dist_t>::SaveIndex(const string& location) {
    std::ofstream output(location,
                         std::ios::binary /* text files can be opened in binary mode as well */);
    //CHECK_MSG(output, "Cannot open file '" + location + "' for writing");
    output.exceptions(ios::badbit | ios::failbit);

    unsigned int optimIndexFlag = data_level0_memory_ != nullptr;

    writeBinaryPOD(output, optimIndexFlag);

    SaveOptimizedIndex(output);

    output.close();
}

template <typename dist_t>
void
Hnsw<dist_t>::SaveOptimizedIndex(std::ostream& output) {
    totalElementsStored_ = ElList_.size();

    writeBinaryPOD(output, totalElementsStored_);
    writeBinaryPOD(output, memoryPerObject_);
    writeBinaryPOD(output, offsetLevel0_);
    writeBinaryPOD(output, offsetData_);
    writeBinaryPOD(output, maxlevel_);
    writeBinaryPOD(output, enterpointId_);
    writeBinaryPOD(output, maxM_);
    writeBinaryPOD(output, maxM0_);
    writeBinaryPOD(output, dist_func_type_);
    writeBinaryPOD(output, searchMethod_);


    LOG(INFO) << "totalElementsStored_                   = " << totalElementsStored_;
    LOG(INFO) << "memoryPerObject_      = " << memoryPerObject_;
    LOG(INFO) << "offsetData_      = " << offsetData_;
    LOG(INFO) << "maxlevel_			          = " << maxlevel_;
    LOG(INFO) << "enterpointId_			          = " << enterpointId_;

    LOG(INFO) << "maxM_                = " << maxM_;
    LOG(INFO) << "maxM0_= " << maxM0_;
    LOG(INFO) << "dist_func_type_       = " << dist_func_type_;
    LOG(INFO) << "searchMethod_       = " << searchMethod_;

    size_t data_plus_links0_size = memoryPerObject_ * totalElementsStored_;
    LOG(INFO) << "writing " << data_plus_links0_size << " bytes";
    //output.write(data_level0_memory_, data_plus_links0_size);
    char* data_level0_memory_ = (char*)malloc((memoryPerObject_) + EXTRA_MEM_PAD_SIZE);
    CHECK(data_level0_memory_);
   
    for (size_t i = 0; i < ElList_.size(); i++) {
        ElList_[i]->copyDataAndLevel0LinksToOptIndex(
            data_level0_memory_, offsetLevel0_, offsetData_);
        output.write(data_level0_memory_, memoryPerObject_);
    };
    
    
    return;

}

template <typename dist_t>
void
Hnsw<dist_t>::LoadIndex(const string& location) {
    LOG(INFO) << "Loading index from " << location;
    std::ifstream input(location,
                        std::ios::binary); /* text files can be opened in binary mode as well */
    //CHECK_MSG(input, "Cannot open file '" + location + "' for reading");

    input.exceptions(ios::badbit | ios::failbit);
    unsigned int optimIndexFlag = 0;

    readBinaryPOD(input, optimIndexFlag);
    LoadOptimizedIndex(input);
    input.close();

    LOG(INFO) << "Finished loading index";
    visitedlistpool = new VisitedListPool(1, totalElementsStored_);


}

template <typename dist_t>
void
Hnsw<dist_t>::LoadOptimizedIndex(std::istream& input) {
    LOG(INFO) << "Loading optimized index.";

    readBinaryPOD(input, totalElementsStored_);
    readBinaryPOD(input, memoryPerObject_);
    readBinaryPOD(input, offsetLevel0_);
    readBinaryPOD(input, offsetData_);
    readBinaryPOD(input, maxlevel_);
    readBinaryPOD(input, enterpointId_);
    readBinaryPOD(input, maxM_);
    readBinaryPOD(input, maxM0_);
    readBinaryPOD(input, dist_func_type_);
    readBinaryPOD(input, searchMethod_);

    LOG(INFO) << "totalElementsStored_                   = " << totalElementsStored_;
    LOG(INFO) << "memoryPerObject_      = " << memoryPerObject_;
    LOG(INFO) << "offsetData_      = " << offsetData_;
    LOG(INFO) << "maxlevel_			          = " << maxlevel_;
    LOG(INFO) << "enterpointId_			          = " << enterpointId_;

    LOG(INFO) << "maxM_                = " << maxM_;
    LOG(INFO) << "maxM0_= " << maxM0_;
    LOG(INFO) << "dist_func_type_       = " << dist_func_type_;
    LOG(INFO) << "searchMethod_       = " << searchMethod_;



    LOG(INFO) << "searchMethod: " << searchMethod_;
    LOG(INFO) << "dist_func_type: " << dist_func_type_;
    fstdistfunc_ = getDistFunc(dist_func_type_);
    iscosine_ = (dist_func_type_ == DIST_TYPE_NORMCOSINE);
    //CHECK_MSG(fstdistfunc_ != nullptr, "Unknown distance function code: " + std::to_string(dist_func_type_));

    //        LOG(INFO) << input.tellg();
    LOG(INFO) << "Total: " << totalElementsStored_ << ", Memory per object: " << memoryPerObject_;
    size_t data_plus_links0_size = memoryPerObject_ * totalElementsStored_;
    // we allocate a few extra bytes to prevent prefetch from accessing out of range memory

    //data_level0_memory_ = (char *)malloc(data_plus_links0_size + EXTRA_MEM_PAD_SIZE);
    int32_t pagesize = getpagesize();
    size_t true_data_plus_links0_size = data_plus_links0_size + (pagesize - data_plus_links0_size % pagesize);
    void* memb = nullptr;
    int err = posix_memalign(&memb, pagesize, true_data_plus_links0_size);

    if (err != 0) {
        std::runtime_error("alloc_aligned_mem_failed errno=" + errno);
        return;
    }

    data_level0_memory_ = reinterpret_cast<char*>(memb);
    CHECK(data_level0_memory_);
    input.read(data_level0_memory_, data_plus_links0_size);
    return;
}


template class Hnsw<float>;
template class Hnsw<int>;
}
