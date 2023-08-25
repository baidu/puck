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

#include <cctype>
#include <cstddef>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <cstring>
#include <cctype>
#include <map>
#include <typeinfo>
#include <random>
#include <climits>
#include <stdexcept>
#include <memory>

#include "puck/tinker/idtype.h"

// compiler_warning.h
#define STRINGISE_IMPL(x) #x
#define STRINGISE(x) STRINGISE_IMPL(x)

/*
 * This solution for generating
 * cross-platform warnings
 * is taken from http://stackoverflow.com/a/1911632/2120401
 * Use: #pragma message WARN("My message")
 * Use: //#pragma message INFO("My message")
 *
 * Note: We may need other other definitions for other compilers,
 *       but so far it worked for MSVS, GCC, CLang, and Intel.
 */

#ifdef _MSC_VER
#define PATH_SEPARATOR "\\"
#else
#define PATH_SEPARATOR "/"
#endif

#define FIELD_DELIMITER ':'

#define NMSLIB_SIZE_T_MAX (std::numeric_limits<size_t>::max())

namespace similarity {

typedef std::mt19937 RandomGeneratorType;

/*
 * 1. Random number generation is thread safe when respective
 *    objects are not shared among threads. So, we will keep one
 *    random number generator per thread.
 * 2. There is a default seed to initialize all random generators.
 * 3. However, sometimes we may want to reset the random number generator
 *    within a working thread (i.e., this would be only a thread-specific change).
 *    In particular, this is needed to improve reproducibility of integration tests.
 */
extern int defaultRandomSeed;

inline RandomGeneratorType& getThreadLocalRandomGenerator() {
    static thread_local RandomGeneratorType  randomGen(defaultRandomSeed);

    return randomGen;
}

// random 32-bit integer number
inline int32_t RandomInt() {
    /*
     * Random number generation is thread safe when respective
     * objects are not shared among threads. So, we will keep one
     * random number generator per thread.
    */
    // thread_local is static by default, but let's keep it static for clarity
    static thread_local std::uniform_int_distribution<int32_t> distr(0, std::numeric_limits<int32_t>::max());

    return distr(getThreadLocalRandomGenerator());
}

template <class T>
// random real number from 0 (inclusive) to 1 (exclusive)
inline T RandomReal() {
    /*
     * Random number generation is thread safe when respective
     * objects are not shared among threads. So, we will keep one
     * random number generator per thread.
    */
    // thread_local is static by default, but let's keep it static for clarity
    static thread_local std::uniform_real_distribution<T> distr(0, 1);

    return distr(getThreadLocalRandomGenerator());
}
/*
 * This function will only work for strings without spaces and commas
 * TODO(@leo) replace, perhaps, it with a more generic version.
 * In particular, we want to be able to escape both spaces and commas.
 */
template <typename ElemType>
inline bool SplitStr(const std::string& str_, std::vector<ElemType>& res, const char SplitChar) {
    res.clear();

    if (str_.empty()) {
        return true;
    }

    std::string str = str_;

    for (auto it = str.begin(); it != str.end(); ++it) {
        if (*it == SplitChar) {
            *it = ' ';
        }
    }

    std::stringstream inp(str);

    while (!inp.eof()) {
        ElemType token;

        if (!(inp >> token)) {
            return false;
        }

        res.push_back(token);
    }

    return true;
}

template <typename T>
void writeBinaryPOD(std::ostream& out, const T& podRef) {
    out.write((char*)&podRef, sizeof(T));
}

template <typename T>
static void readBinaryPOD(std::istream& in, T& podRef) {
    in.read((char*)&podRef, sizeof(T));
}

template <typename T>
static void readBinaryPOD(const void* in, T& podRef) {
    std::memcpy((char*)&podRef, in, sizeof(T));
}

template <typename T>
void writeBinaryPOD(void* out, const T& podRef) {
    std::memcpy(out, (char*)&podRef, sizeof(T));
}

/**/

inline void ToLower(std::string& s) {
    for (size_t i = 0; i < s.size(); ++i) {
        s[i] = std::tolower(s[i]);
    }
}

}  // namespace similarity

