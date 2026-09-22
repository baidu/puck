#pragma once

#include <functional>
#include "tinker/internal/utils.h"
#include "quantizer/quantizer.h"
#include "quantizer/FLAT.h"
#include "quantizer/LVQ.h"

namespace puck {

inline bool is_digits(std::string ss){
    for(auto c : ss){
        if(!std::isdigit(c)) return false;
    }
    return true;
}

inline int make_quantizer_dynamic(const std::string &type_str,
                                  std::unique_ptr<Quantizer> &quant,
                                  size_t feature_dim,
                                  bool whether_norm){
    if(type_str.find("FLAT") != std::string::npos){
        quant.reset(new QuantizerFLAT(feature_dim));
    }
    else if(size_t loc = type_str.find("LVQ"); loc != std::string::npos){
        std::vector<std::string> params = utils::ssplit(type_str.substr(loc + 3), 'x');
        if(params.size() < 2 || !is_digits(params[0]) || !is_digits(params[1])){
            LOG(ERROR) << "Invalid LVQ type: " << type_str << ". LVQ8x8 / LVQ8x4 / LVQ8x0 / LVQ4x8 / LVQ4x4 / LVQ4x0 supported";
            return -1;
        }
        int bita = std::stoi(params[0]);
        int bitb = std::stoi(params[1]);
        if(bita != 4 && bita != 8){
            LOG(ERROR) << "Unsupported LVQ first level quant bit for " << bita << ", only 4 and 8 are supported";
            return -1;
        }
        if(bitb != 0 && bitb != 4 && bitb != 8){
            LOG(ERROR) << "Unsupported LVQ second level quant bit for " << bitb << ", only 0, 4 and 8 are supported";
            return -1;
        }
        quant.reset(new QuantizerLVQ(feature_dim,
                                     std::stoi(params[0]),
                                     std::stoi(params[1]), whether_norm));
    }
    else{
        LOG(ERROR) << "Unknown quantizer type: " << type_str;
        return -1;
    }
    return 0;
}

inline std::function<float(const char*, const float*, QuantizerParams&)> 
get_quant_adc_distance_func(std::unique_ptr<Quantizer> &quant){
    std::function<float(const char*, const float*, QuantizerParams&)> func;
    switch(quant->quant_type()){
        case QuantType::FLAT:
            func = std::bind(
                        &QuantizerFLAT::adc_distance, 
                        static_cast<QuantizerFLAT*>(quant.get()), 
                        std::placeholders::_1, 
                        std::placeholders::_2, 
                        std::placeholders::_3);
            break;
        case QuantType::LVQ:
            func = std::bind(
                        &QuantizerLVQ::adc_distance, 
                        static_cast<QuantizerLVQ*>(quant.get()), 
                        std::placeholders::_1, 
                        std::placeholders::_2, 
                        std::placeholders::_3);
            break;
        default:
            LOG(ERROR) << "Unknown quantizer typeID: " << static_cast<int>(quant->quant_type());
            func = nullptr;
    }
    return func;
}

inline std::function<float(const char*, const char*, QuantizerParams&)> 
get_quant_sdc_distance_func(std::unique_ptr<Quantizer> &quant){
    std::function<float(const char*, const char*, QuantizerParams&)> func;
    switch(quant->quant_type()){
        case QuantType::FLAT:
            func = std::bind(
                        &QuantizerFLAT::sdc_distance, 
                        static_cast<QuantizerFLAT*>(quant.get()), 
                        std::placeholders::_1, 
                        std::placeholders::_2, 
                        std::placeholders::_3);
            break;
        case QuantType::LVQ:
            func = std::bind(
                        &QuantizerLVQ::sdc_distance, 
                        static_cast<QuantizerLVQ*>(quant.get()), 
                        std::placeholders::_1, 
                        std::placeholders::_2, 
                        std::placeholders::_3);
            break;
        default:
            LOG(ERROR) << "Unknown quantizer typeID: " << static_cast<int>(quant->quant_type());
            func = nullptr;
    }
    return func;
}

inline int load_quantizer_dynamic(const std::string &filename,
                                  std::unique_ptr<Quantizer> &quant,
                                  bool whether_norm){
    std::ifstream in(filename);
    if(!in.is_open()){
        LOG(ERROR) << "Cannot open file: " << filename;
        return -1;
    }
    LOG(INFO) << "loading quantizer from " << filename;
    QuantType quant_t;
    in.read(reinterpret_cast<char*>(&quant_t), sizeof(QuantType));
    // init
    switch(quant_t){
        case QuantType::FLAT:
            quant.reset(new QuantizerFLAT());
            break;
        case QuantType::LVQ:
            quant.reset(new QuantizerLVQ());
            static_cast<QuantizerLVQ*>(quant.get())->set_whether_norm(whether_norm);
            break;
        default:
            LOG(ERROR) << "Unknown quantizer typeID: " << static_cast<int>(quant_t);
            return -1;
    }
    in.close();
    // load
    return quant->load(filename);
}

} // namespace puck
