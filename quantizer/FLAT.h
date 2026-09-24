#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <glog/logging.h>

#include "quantizer/quantizer.h"

namespace puck {

class QuantizerFLAT : public Quantizer{
public:
    QuantizerFLAT(){}
    QuantizerFLAT(size_t feature_dim){
        LOG(INFO) << "initializing Quantizer FLAT";
        _feature_dim = feature_dim;
        _data_length = _feature_dim * sizeof(float);
    }
    ~QuantizerFLAT(){}

    virtual puck::QuantType quant_type() const override { return puck::QuantType::FLAT; }

    virtual int train() override { return 0; }

    virtual int encode(char *des, const float *src, QuantizerParams &params) override {
        memcpy(des, src, _data_length);
        return 0;
    }

    virtual int decode(float *des, const char *src, QuantizerParams &params) override {
        memcpy(des, src, _data_length);
        return 0;
    }

    virtual bool need_recode() const override { return false; }

    virtual bool support_rerank() const override { return false; }

    virtual int save(const std::string &filename) override;
    virtual int load(const std::string &filename) override;
    virtual float sdc_distance(const char *c_u8, const char *v_u8, QuantizerParams &params) noexcept override;
    virtual float adc_distance(const char *c_u8, const float *v_fp32, QuantizerParams &params) noexcept override;


};

} // namespace puck
