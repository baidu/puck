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

#pragma once

#include <string>
#include <limits>
#include <map>
#include <stdexcept>

#include <string.h>
//#include "global.h"
#include "puck/tinker/object.h"
#include "puck/tinker/utils.h"
#include "puck/tinker/space.h"
//#include "space_vector.h"
#include "puck/tinker/method/hnsw_distfunc_opt_impl_inline.h"

//#define SPACE_L     "lp"
//#define SPACE_LINF  "linf"
//#define SPACE_L1    "l1"
#define SPACE_L2    "l2"

namespace similarity {

template <typename dist_t>
class SpaceLpDist {
public:
    explicit SpaceLpDist(dist_t pf) : p_(static_cast<int>(pf)), pf_(pf), custom_(false) {
        if (fabs(dist_t(p_) - pf_) < numeric_limits<dist_t>::min()) {
            custom_ = p_ == -1 || p_ == 1 || p_ == 2;
        }
    }

    dist_t operator()(const dist_t* x, const dist_t* y, size_t length) const {
        CHECK(p_ == 2);
        TMP_RES_ARRAY(TmpRes)
        return L2SqrExt(x, y, length, TmpRes);
    }
    dist_t getP() const {
        return pf_;
    }
    bool getCustom() const {
        return custom_;
    }
private:
    int     p_;
    dist_t  pf_;
    bool    custom_; // Do we use a custom implementation for l=0,1,2?
};

template <typename dist_t>
class SpaceLp : public Space<dist_t> {
public:
    explicit SpaceLp(dist_t p) : distObj_(p) {}
    virtual ~SpaceLp() {}
    dist_t getP() const {
        return distObj_.getP();
    }

protected:
    virtual dist_t HiddenDistance(const Object* obj1, const Object* obj2) const {
        CHECK(obj1->datalength() > 0);
        CHECK(obj1->datalength() == obj2->datalength());
        const dist_t* x = reinterpret_cast<const dist_t*>(obj1->data());
        const dist_t* y = reinterpret_cast<const dist_t*>(obj2->data());
        const size_t length = obj1->datalength() / sizeof(dist_t);
        return distObj_(x, y, length);
    }
private:
    SpaceLpDist<dist_t> distObj_;
    //DISABLE_COPY_AND_ASSIGN(SpaceLp);
};

}  // namespace similarity

