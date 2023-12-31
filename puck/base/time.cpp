// Copyright (c) 2010 baidu-rpc authors.
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

// Author: Ge,Jun (gejun@baidu.com)
// Date: Fri Aug 29 15:01:15 CST 2014

#include <unistd.h>                          // close
#include <sys/types.h>                       // open
#include <sys/stat.h>                        // ^
#include <fcntl.h>                           // ^
#include <sstream>
#include <string>
#include <limits>
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <string.h>                          // memmem

#include "puck/base/time.h"
//#include "base/file_util.h"
//#include "base/strings/string_number_conversions.h"
//#include "base/strings/string_util.h"
namespace puck {
namespace base {

const char* last_changed_revision() {
#if defined(BASE_REVISION)
    return BASE_REVISION;
#else
    return "undefined";
#endif
}

int64_t monotonic_time_ns() {
    // MONOTONIC_RAW is slower than MONOTONIC in linux 2.6.32, trying to
    // use the RAW version does not make sense anymore.
    // NOTE: Not inline to keep ABI-compatible with previous versions.
    timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return now.tv_sec * 1000000000L + now.tv_nsec;
}

namespace detail {

// read_cpu_frequency() is modified from source code of glibc.
int64_t read_cpu_frequency(bool* invariant_tsc) {
    /* We read the information from the /proc filesystem.  It contains at
       least one line like
       cpu MHz         : 497.840237
       or also
       cpu MHz         : 497.841
       We search for this line and convert the number in an integer.  */

    const int fd = open("/proc/cpuinfo", O_RDONLY);
    if (fd < 0) {
        return 0;
    }

    int64_t result = 0;
    char buf[4096];  // should be enough
    const ssize_t n = read(fd, buf, sizeof(buf));
    if (n > 0) {
        char *mhz = static_cast<char*>(memmem(buf, n, "cpu MHz", 7));

        if (mhz != NULL) {
            char *endp = buf + n;
            int seen_decpoint = 0;
            int ndigits = 0;

            /* Search for the beginning of the string.  */
            while (mhz < endp && (*mhz < '0' || *mhz > '9') && *mhz != '\n') {
                ++mhz;
            }
            while (mhz < endp && *mhz != '\n') {
                if (*mhz >= '0' && *mhz <= '9') {
                    result *= 10;
                    result += *mhz - '0';
                    if (seen_decpoint)
                        ++ndigits;
                } else if (*mhz == '.') {
                    seen_decpoint = 1;
                }
                ++mhz;
            }

            /* Compensate for missing digits at the end.  */
            while (ndigits++ < 6) {
                result *= 10;
            }
        }

        if (invariant_tsc) {
            char* flags_pos = static_cast<char*>(memmem(buf, n, "flags", 5));
            *invariant_tsc = 
                (flags_pos &&
                 memmem(flags_pos, buf + n - flags_pos, "constant_tsc", 12) &&
                 memmem(flags_pos, buf + n - flags_pos, "nonstop_tsc", 11));
        }
    }
    close (fd);
    return result;
}
bool stringToUINT64(const std::string& s, int64_t& ret)
{
    std::stringstream a;
    a << s;
    a >> ret;
    return true;
}

static bool read_int64_from_file(const char* filename, int64_t* value) {
    std::string contents;
    //if (!ReadFileToString(FilePath(filename), &contents)) {
    //    return false;
    //}
    int fd = -1;
    fd = open(filename, O_RDONLY);
    if (fd == -1){
        return false;
    }  

    char buf[1 << 16];
    size_t len;
    size_t size = 0;
    bool read_status = true;
    // Many files supplied in |path| have incorrect size (proc files etc).
    // Hence, the file is read sequentially as opposed to a one-shot read.
    while ((len = read(fd, buf, 1)) > 0) {
      contents.append(buf, len);
      size += len;
    }
    close(fd);
    return !contents.empty() && stringToUINT64(contents, *value);
}

// Return value must be >= 0
int64_t read_invariant_cpu_frequency() {
    bool invariant_tsc = false;
    int64_t cpuinfo_freq = read_cpu_frequency(&invariant_tsc);
    if (!invariant_tsc || cpuinfo_freq < 0) {
        return 0;
    }
    const char* tsc_freq_khz_file = "/sys/devices/system/cpu/cpu0/cpufreq/tsc_freq_khz";
    int64_t freq = 0;
    if (read_int64_from_file(tsc_freq_khz_file, &freq) && freq > 0) {
        return freq * 1000;
    }
    return cpuinfo_freq;
}

int64_t invariant_cpu_freq = -1;
}  // namespace detail

}  // namespace base
}  // namespace puck
