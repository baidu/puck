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

#include <cstring>
#include <cctype>
#include <string>
#include <sstream>
#include <vector>
#include <list>
#include <utility>
#include <limits>
#include <algorithm>
#include <cstdint>

//#include "global.h"
#include "puck/tinker/idtype.h"
#include <glog/logging.h>

namespace similarity {

using std::string;
using std::stringstream;
using std::numeric_limits;

/*
 * Structure of object: | 4-byte id | 4-byte label | 8-byte datasize | data ........ |
 * We need data to be aligned on 8-byte boundaries.
 *
 * TODO 1) this all apparenlty hinges on the assumption that malloc() gives addresses
 *      that are 8-bye aligned. So, this is related to issue #9
 *      2) even though GCC doesn't complain, using a single char buffer may break aliasing rules
 *
 * See also http://searchivarius.org/blog/what-you-must-know-about-alignment-21st-century
 */
class Object {
public:
    /*
     * Memory ownership handling of the Object class is not good, but we keep it this way for historical/compatibility reasons
     * Currrently, when the second constructor is called, new memory is always allocated.
     * However, this constructor is nornally called to created an object that reuses someone else's memory.
     * In NMSLIB 1.8.1 we make it possible to take ownership of the memory provided.
     */
    explicit Object(char* buffer, bool memory_allocated = false) : buffer_(buffer),
        memory_allocated_(memory_allocated) {}

    Object(IdType id, LabelType label, size_t datalength, const void* data) {
        buffer_ = new char[ID_SIZE + LABEL_SIZE + DATALENGTH_SIZE + datalength];
        CHECK(buffer_ != NULL);
        memory_allocated_ = true;
        char* ptr = buffer_;
        memcpy(ptr, &id, ID_SIZE);
        ptr += ID_SIZE;
        memcpy(ptr, &label, LABEL_SIZE);
        ptr += LABEL_SIZE;
        memcpy(ptr, &datalength, DATALENGTH_SIZE);
        ptr += DATALENGTH_SIZE;
        data_length_ = datalength;

        if (data != NULL) {
            memcpy(ptr, data, datalength);
        } else {
            memset(ptr, 0, datalength);
        }
    }

    ~Object() {
        if (memory_allocated_) {
            delete[] buffer_;
        }
    }

    inline IdType    id()         const {
        return *(reinterpret_cast<IdType*>(buffer_));
    }
    inline LabelType label()      const {
        return *(reinterpret_cast<LabelType*>(buffer_ + ID_SIZE));
    }
    inline size_t datalength()    const {
        //return *(reinterpret_cast<size_t*>(buffer_ + LABEL_SIZE + ID_SIZE));
        return data_length_;
    }
    inline const char* data() const {
        return buffer_ + ID_SIZE + LABEL_SIZE + DATALENGTH_SIZE;
    }
    inline char* data()             {
        return buffer_ + ID_SIZE + LABEL_SIZE + DATALENGTH_SIZE;
    }

    inline const char* buffer()  const {
        return buffer_;
    }
    inline size_t bufferlength() const {
        return ID_SIZE + LABEL_SIZE + DATALENGTH_SIZE + datalength();
    }
private:
    char* buffer_;
    bool  memory_allocated_;
public:
    static size_t data_length_;
    // disable copy and assign
    //DISABLE_COPY_AND_ASSIGN(Object);
};

/*
 * Clearing memory: we will use some smart pointer here (somewhen).
 *                  can't use standard shared_ptr, b/c they have
 *                  performance issues.
 * see, e.g.: http://nerds-central.blogspot.com/2012/03/sharedptr-performance-issues-and.html
 */
typedef std::vector<const Object*> ObjectVector;

}   // namespace similarity

