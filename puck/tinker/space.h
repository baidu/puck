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
#include <map>
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <memory>
#include <limits>
#include <vector>

#include <string.h>
//#include "global.h"
#include "puck/tinker/object.h"
#include "puck/tinker/utils.h"
#include <glog/logging.h>
//#include "permutation_type.h"

#define LABEL_PREFIX "label:"

#define DIST_TYPE_INT      "int"
#define DIST_TYPE_FLOAT    "float"

namespace similarity {

using std::map;
using std::string;
using std::ifstream;
using std::ofstream;
using std::pair;
using std::make_pair;
using std::runtime_error;
using std::hash;
template <typename dist_t> class Space;

template <typename dist_t>
class Space {
public:
    explicit Space() {}
    virtual ~Space() {}
    // This function is public and it is not supposed to be used in the query-mode
    dist_t IndexTimeDistance(const Object* obj1, const Object* obj2) const {

        return HiddenDistance(obj1, obj2);
    }
protected:
    /*
     * This function is private, but it will be accessible by the friend class Query
     * IndexTimeDistance access can be disable/enabled only by function friends
     */
    virtual dist_t HiddenDistance(const Object* obj1, const Object* obj2) const = 0;
private:

    //DISABLE_COPY_AND_ASSIGN(Space);
};

}  // namespace similarity

