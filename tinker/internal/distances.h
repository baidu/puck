#pragma once

#include <stdint.h>
#include <x86intrin.h>

namespace puck {

char *TMP_ARRAY();

float distanceL2(const float* a, const float* b, size_t n);

}
