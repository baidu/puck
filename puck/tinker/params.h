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
#ifndef PARAMS_H
#define PARAMS_H

#include <string>
#include <vector>
#include <limits>
#include <map>
#include <set>
#include <memory>
#include <sstream>
#include <algorithm>
#include <stdexcept>

#include <glog/logging.h>
#include "puck/tinker/utils.h"

namespace similarity {

//using std::string;
//using std::vector;
//using std::multimap;
//using std::set;
//using std::stringstream;
//using std::shared_ptr;
//using std::unique_ptr;

#define FAKE_MAX_LEAVES_TO_VISIT std::numeric_limits<int>::max()

class AnyParams {
public:
    /*
     * Each element of the description array is in the form:
     * <param name>=<param value>
     */
    AnyParams(const std::vector<std::string>& Desc) : ParamNames(0), ParamValues(0) {
        std::set<std::string> seen;

        for (unsigned i = 0; i < Desc.size(); ++i) {
            std::vector<std::string>  OneParamPair;

            if (!SplitStr(Desc[i], OneParamPair, '=') ||
                    OneParamPair.size() != 2) {
                std::stringstream err;
                err << "Wrong format of an argument: '" << Desc[i] << "' should be in the format: <Name>=<Value>";
                LOG(ERROR)  << err.str();
                throw std::runtime_error(err.str());
            }

            const std::string& Name = OneParamPair[0];
            const std::string& sVal = OneParamPair[1];

            if (seen.count(Name)) {
                std::string err = "Duplicate parameter: " + Name;
                LOG(ERROR)  << err;
                throw std::runtime_error(err);
            }

            seen.insert(Name);

            ParamNames.push_back(Name);
            ParamValues.push_back(sVal);
        }
    }

    AnyParams() {}

    std::vector<std::string>  ParamNames;
    std::vector<std::string>  ParamValues;

};

const inline AnyParams& getEmptyParams() {
    static AnyParams empty;
    return empty;
}

class AnyParamManager {
public:
    AnyParamManager(const AnyParams& params) : params(params), seen() {
        if (params.ParamNames.size() != params.ParamValues.size()) {
            std::string err = "Bug: different # of parameters and values";
            LOG(ERROR) << err;
            throw std::runtime_error(err);
        }
    }

    template <typename ParamType, typename DefaultType>
    void GetParamOptional(const std::string&  Name, ParamType& Value, const DefaultType& DefaultValue) {
        Value = DefaultValue;
        GetParam<ParamType>(Name, Value, false);
    }
    bool hasParam(const std::string& name) {
        for (const std::string& s : params.ParamNames)
            if (s == name) {
                return true;
            }

        return false;
    };

private:
    const AnyParams&  params;
    std::set<std::string>       seen;

    template <typename ParamType>
    void GetParam(const std::string&  Name, ParamType& Value, bool bRequired) {
        bool bFound = false;

        /*
         * This loop is reasonably efficient, unless
         * we have thousands of parameters (which realistically won't happen)
         */
        for (size_t i = 0; i < params.ParamNames.size(); ++i)
            if (Name == params.ParamNames[i]) {
                bFound = true;
                ConvertStrToValue<ParamType>(params.ParamValues[i], Value);
            }

        if (bFound) {
            seen.insert(Name);
        }

        if (!bFound) {
            if (bRequired) {
                std::stringstream err;
                err <<  "Mandatory parameter: '" << Name << "' is missing!";
                LOG(ERROR) << err.str();
                throw std::runtime_error(err.str());
            }
        }
    }

    template <typename ParamType>
    void ConvertStrToValue(const std::string& s, ParamType& Value);
};


template <typename ParamType>
inline void AnyParamManager::ConvertStrToValue(const std::string& s, ParamType& Value) {
    std::stringstream str(s);

    if (!(str >> Value) || !str.eof()) {
        std::stringstream err;
        err << "Failed to convert value '" << s << "' from type: " << typeid(Value).name();
        LOG(ERROR) << err.str();
        throw std::runtime_error(err.str());
    }
}

template <>
inline void AnyParamManager::ConvertStrToValue<std::string>(const std::string& str, std::string& Value) {
    Value = str;
}

};

#endif
