#pragma once

#include <memory>
#include <limits>

namespace puck {

enum class QuantType {
    UNDEF = -1,
    FLAT = 0,
    LVQ
};

constexpr bool need_recode(const QuantType quant_type){
    return quant_type != QuantType::FLAT;
}

class QuantizerParams{
public:
    // 通用参数
    //// simd accumulation/reduction buffer
    char *buffer{nullptr};
    //// decoded vector buffer
    char *decoded_buffer{nullptr};

    // LVQ 参数
    //// 解码等级，<code_level>级残差解码
    int code_level{std::numeric_limits<int>::max()};
};

class Quantizer{
public:
    Quantizer() {}
    virtual ~Quantizer() {};

    virtual int train() = 0;

    virtual int encode(char *des, const float *src, QuantizerParams &params) = 0;
    virtual int decode(float* des, const char *src, QuantizerParams &params) = 0;
    virtual float sdc_distance(const char *c_u8, const char *v_u8, QuantizerParams &params) noexcept = 0;
    virtual float adc_distance(const char *c_u8, const float *v_fp32, QuantizerParams &params) noexcept = 0;

    virtual int save(const std::string &filename) = 0;
    virtual int load(const std::string &filename) = 0;

    virtual QuantType quant_type() const = 0;
    virtual bool need_recode() const = 0;
    virtual bool support_rerank() const = 0;

    size_t dim() { return _feature_dim; };
    size_t data_length() { return _data_length; };

protected:
    size_t _feature_dim;
    size_t _data_length;
};

} // namespace puck
