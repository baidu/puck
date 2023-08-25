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

#include <stdio.h>
#include <string>
#include <vector>

#include "puck/tinker/params.h"
#include "puck/tinker/object.h"

namespace similarity {


/*
 * Abstract class for all index structures
 */
template <typename dist_t>
class Index {
public:
    Index(const ObjectVector& data) : data_(data) {}

    // Create an index using given parameters
    virtual void CreateIndex(const AnyParams& indexParams) = 0;
    // SaveIndex is not necessarily implemented
    virtual void SaveIndex(const std::string& location) {
        (void)location;
        throw std::runtime_error("SaveIndex is not implemented for method: " + StrDesc());
    }
    // LoadIndex is not necessarily implemented
    virtual void LoadIndex(const std::string& location) {
        (void)location;
        throw std::runtime_error("LoadIndex is not implemented for method: " + StrDesc());
    }
    virtual ~Index() {}
    virtual const std::string StrDesc() const = 0;
    // Set query-time parameters
    virtual void SetQueryTimeParams(const AnyParams& params) = 0;
    virtual size_t GetSize() const {
        return data_.size();
    }
protected:
    const ObjectVector& data_;

private:

};

}  // namespace similarity
