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

#include <atomic>
#include <thread>
#include <queue>
#include <mutex>

namespace similarity {

/*
 * replacement for the openmp '#pragma omp parallel for' directive
 * only handles a subset of functionality (no reductions etc)
 * Process ids from start (inclusive) to end (EXCLUSIVE)
 */
template <class Function>
inline void ParallelFor(size_t start, size_t end, size_t numThreads, Function fn) {
    if (numThreads <= 0) {
        numThreads = std::thread::hardware_concurrency();
    }

    if (numThreads == 1) {
        for (size_t id = start; id < end; id++) {
            fn(id, 0);
        }
    } else {
        std::vector<std::thread>  threads;
        //std::atomic<size_t>       current(0);

        // keep track of exceptions in threads
        // https://stackoverflow.com/a/32428427/1713196
        std::exception_ptr lastException = nullptr;
        std::mutex         lastExceptMutex;
        int avg_cnt = std::ceil(1.0 * (end - start) / numThreads);

        for (size_t threadId = 0; threadId < numThreads; ++threadId) {
            threads.push_back(std::thread([&, threadId] {
                //size_t thread_id = current.fetch_add(1);
                size_t thread_id = threadId;
                int id = thread_id * avg_cnt + start;
                int cur_end = std::min((thread_id + 1) * avg_cnt + start, end);
                LOG(INFO) << "threadId = " << threadId << " [" << id << ", " << cur_end << "]";

                while (true) {
                    if ((id >= cur_end)) {
                        break;
                    }

                    try {
                        fn(id, threadId);
                        ++id;
                    } catch (...) {
                        std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
                        lastException = std::current_exception();
                        /*
                         * This will work even when current is the largest value that
                         * size_t can fit, because fetch_add returns the previous value
                         * before the increment (what will result in overflow
                         * and produce 0 instead of current + 1).
                         */
                        break;
                    }
                }
            }));
        }

        for (auto& thread : threads) {
            thread.join();
        }

        if (lastException) {
            std::rethrow_exception(lastException);
        }
    }


}
};
