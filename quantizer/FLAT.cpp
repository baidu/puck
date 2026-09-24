#include "tinker/internal/distances.h"
#include "quantizer/FLAT.h"

namespace puck {

int QuantizerFLAT::save(const std::string &filename){
    std::ofstream out(filename);
    if(!out.is_open()){
        LOG(ERROR) << "open file error, filename = " << filename;
        return -1;
    }
    size_t dim = _feature_dim;
    puck::QuantType quant_t = quant_type();
    out.write(reinterpret_cast<char*>(&quant_t), sizeof(puck::QuantType));
    out.write(reinterpret_cast<char*>(&dim), sizeof(size_t));

    if(!out.good()){
        LOG(ERROR) << "save FLAT error!";
        return -1;
    }

    LOG(INFO) << "save FLAT quantizer, filename = " << filename;

    return 0;
}

int QuantizerFLAT::load(const std::string &filename){
    std::ifstream in(filename);
    if(!in.is_open()){
        LOG(ERROR) << "open file error, filename = " << filename;
        return -1;
    }
    puck::QuantType quant_t;
    in.read(reinterpret_cast<char*>(&quant_t), sizeof(puck::QuantType));
    if(quant_t != quant_type()){
        LOG(ERROR) << "quant type not match! read typeID = " << static_cast<int>(quant_t) 
                    << ", expect typeID = " << static_cast<int>(quant_type());
        return -1;
    }
    LOG(INFO) << "loading Quantizer FLAT";
    in.read(reinterpret_cast<char*>(&_feature_dim), sizeof(size_t));
    _data_length = _feature_dim * sizeof(float);
    return 0;
}

float QuantizerFLAT::sdc_distance(const char *c_u8, const char *v_u8, QuantizerParams &params) noexcept {
    const float *c = reinterpret_cast<const float*>(c_u8);
    const float *v = reinterpret_cast<const float*>(v_u8);
    return puck::distanceL2(c, v, _feature_dim);
}

float QuantizerFLAT::adc_distance(const char *c_u8, const float *v_fp32, QuantizerParams &params) noexcept {
    const float *c = reinterpret_cast<const float*>(c_u8);
    return puck::distanceL2(c, v_fp32, _feature_dim);
}

} // namespace puck
