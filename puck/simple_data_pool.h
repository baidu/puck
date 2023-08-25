// Copyright (c) 2018  authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Authors: Ge,Jun (gejun@baidu.com)

#pragma once

#include <pthread.h>
#include <mutex>
#include <atomic>
#include <string.h>

namespace puck {

class DataFactory {
public:
    virtual ~DataFactory() {}

    //Implement this method to create a piece of data.
    // Notice that this method is const.
    // Returns the data, NULL on error.
    virtual void* CreateData() const = 0;

    // Implement this method to destroy a piece of data that was created
    // by Create().
    // Notice that this method is const.
    virtual void DestroyData(void*) const = 0;
};

// As the name says, this is a simple unbounded dynamic-size pool for
// reusing void* data. We're assuming that data consumes considerable
// memory and should be reused as much as possible, thus unlike the
// multi-threaded allocator caching objects thread-locally, we just
// put everything in a global list to maximize sharing. It's currently
// used by Server to reuse session-local data.
class SimpleDataPool {
public:
    struct Stat {
        unsigned nfree;
        unsigned ncreated;
    };

    explicit SimpleDataPool(const DataFactory* factory);
    ~SimpleDataPool();
    void Reset(const DataFactory* factory);
    void Reserve(unsigned n);
    void* Borrow();
    void Return(void*);
    Stat stat() const;

private:
    std::mutex _mutex;
    unsigned _capacity;
    unsigned _size;
    std::atomic<unsigned> _ncreated;
    void** _pool;
    const DataFactory* _factory;
};

inline SimpleDataPool::SimpleDataPool(const DataFactory* factory)
    : _capacity(0)
    , _size(0)
    , _ncreated(0)
    , _pool(nullptr)
    , _factory(factory) {
}

inline SimpleDataPool::~SimpleDataPool() {
    Reset(nullptr);
}

inline void SimpleDataPool::Reset(const DataFactory* factory) {
    unsigned saved_size = 0;
    void** saved_pool = nullptr;
    const DataFactory* saved_factory = nullptr;
    {

        std::unique_lock<std::mutex> mu(_mutex);
        saved_size = _size;
        saved_pool = _pool;
        saved_factory = _factory;
        _capacity = 0;
        _size = 0;
        _ncreated.store(0, std::memory_order_relaxed);
        _pool = nullptr;
        _factory = factory;
    }

    if (saved_pool) {
        if (saved_factory) {
            for (unsigned i = 0; i < saved_size; ++i) {
                saved_factory->DestroyData(saved_pool[i]);
            }
        }

        free(saved_pool);
    }
}

inline void SimpleDataPool::Reserve(unsigned n) {
    if (_capacity >= n) {
        return;
    }

    std::unique_lock<std::mutex> mu(_mutex);

    if (_capacity >= n) {
        return;
    }

    // Resize.
    const unsigned new_cap = std::max(_capacity * 3 / 2, n);
    void** new_pool = (void**)malloc(new_cap * sizeof(void*));

    if (nullptr == new_pool) {
        return;
    }

    if (_pool) {
        memcpy(new_pool, _pool, _capacity * sizeof(void*));
        free(_pool);
    }

    unsigned i = _capacity;
    _capacity = new_cap;
    _pool = new_pool;

    for (; i < n; ++i) {
        void* data = _factory->CreateData();

        if (data == nullptr) {
            break;
        }

        _ncreated.fetch_add(1,  std::memory_order_relaxed);
        _pool[_size++] = data;
    }
}

inline void* SimpleDataPool::Borrow() {
    if (_size) {
        std::unique_lock<std::mutex> mu(_mutex);

        if (_size) {
            return _pool[--_size];
        }
    }

    void* data = _factory->CreateData();

    if (data) {
        _ncreated.fetch_add(1,  std::memory_order_relaxed);
    }

    return data;
}

inline void SimpleDataPool::Return(void* data) {
    if (data == nullptr) {
        return;
    }

    std::unique_lock<std::mutex> mu(_mutex);

    if (_capacity == _size) {
        const unsigned new_cap = (_capacity == 0 ? 128 : (_capacity * 3 / 2));
        void** new_pool = (void**)malloc(new_cap * sizeof(void*));

        if (nullptr == new_pool) {
            mu.unlock();
            return _factory->DestroyData(data);
        }

        if (_pool) {
            memcpy(new_pool, _pool, _capacity * sizeof(void*));
            free(_pool);
        }

        _capacity = new_cap;
        _pool = new_pool;
    }

    _pool[_size++] = data;
}

inline SimpleDataPool::Stat SimpleDataPool::stat() const {
    Stat s = { _size, _ncreated.load(std::memory_order_relaxed) };
    return s;
}

}  // namespace puck

