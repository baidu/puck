#pragma once

#include <fstream>
#include <memory>
#include <vector>
#include <string>
#include <cassert>
#include "quantizer/quantizer.h"

namespace puck {

template <typename T, typename U>
constexpr auto DIV_UP(T x, U y) -> decltype(x + y) { return (x + y - 1) / y; }

class QuantizerLVQ : public Quantizer{
public:
    // 2 level LVQ量化
    QuantizerLVQ() : _whether_norm(false) {}
    QuantizerLVQ(size_t feature_dim, size_t bitx, size_t bity,
                 bool whether_norm) : _whether_norm(whether_norm) {
        LOG(INFO) << "initializing Quantizer LVQ" << bitx << "x" << bity;
        _feature_dim = feature_dim;
        _bitx = bitx, _bity = bity;
        _support_rerank = (bity > 0);
        _data_length = DIV_UP(_feature_dim * bitx, 8) + // 1st level LVQ code (for main distance computation)
                       DIV_UP(_feature_dim * bity, 8) + // 2nd level LVQ residual code (for rerank distance computation)
                       2 * sizeof(float);               // codebook: minval and maxval
    }
    ~QuantizerLVQ(){};
    /*
     * @brief 计算LVQ码本，主要是_mean向量，全量训练
     * @@return (int) : 正常返回0，错误返回值<0
     **/
    virtual int train() override;
    // LVQ code: dim * bit_x + dim * bit_y + 2 * fp32
    virtual int encode(char *des, const float *src, QuantizerParams &params) override;
    /*
     * @brief 解码函数，将量化码解压回原始浮点数，level控制解压等级，等级越高，解压精度越高，计算量越大
     * @@return (int) : 正常返回0，错误返回值<0
     **/
    virtual int decode(float *des, const char *src, QuantizerParams &params) override;

    virtual float sdc_distance(const char *c_u8, const char *v_u8, QuantizerParams &params) noexcept override;
    virtual float adc_distance(const char *c_u8, const float *v_fp32, QuantizerParams &params) noexcept override;

    virtual int save(const std::string &filename) override;
    virtual int load(const std::string &filename) override;

    virtual puck::QuantType quant_type() const override  { return puck::QuantType::LVQ; }
    virtual bool need_recode() const override { return true; }
    virtual bool support_rerank() const override { return _support_rerank; }
    void set_whether_norm(bool whether_norm) { _whether_norm = whether_norm; }

private:
    size_t _bitx, _bity; // only support 0, 4, 8 
    std::vector<float> _mean;
    bool _support_rerank;
    bool _whether_norm;
    
    int encode_impl(unsigned char *des, float *src, size_t bit, float low, float up);
    int decode_impl(float *des, const unsigned char *src, size_t bit, float low, float up);
};

} // namespace puck
