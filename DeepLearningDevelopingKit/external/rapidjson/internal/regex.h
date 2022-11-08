// Tencent is pleased to support the open source community by making RapidJSON available.
// 
// Copyright (C) 2015 THL A29 Limited, a Tencent company, and Milo Yip. All rights reserved.
//
// Licensed under the MIT License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// http://opensource.org/licenses/MIT
//
// Unless required by applicable law or agreed to in writing, software distributed 
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR 
// CONDITIONS OF ANY KIND, either express or implied. See the License for the 
// specific language governing permissions and limitations under the License.

#ifndef RAPIDJSON_INTERNAL_REGEX_H_
#define RAPIDJSON_INTERNAL_REGEX_H_

#include "../allocators.h"
#include "../stream.h"
#include "stack.h"

#ifdef __clang__
RAPIDJSON_DIAG_PUSH
RAPIDJSON_DIAG_OFF(padded)
RAPIDJSON_DIAG_OFF(switch-enum)
RAPIDJSON_DIAG_OFF(implicit-fallthrough)
#elif defined(_MSC_VER)
RAPIDJSON_DIAG_PUSH
RAPIDJSON_DIAG_OFF(4512) // assignment operator could not be generated
#endif

#ifdef __GNUC__
RAPIDJSON_DIAG_PUSH
RAPIDJSON_DIAG_OFF(effc++)
#if __GNUC__ >= 7
RAPIDJSON_DIAG_OFF(implicit-fallthrough)
#endif
#endif

#ifndef RAPIDJSON_REGEX_VERBOSE
#define RAPIDJSON_REGEX_VERBOSE 0
#endif

RAPIDJSON_NAMESPACE_BEGIN
namespace internal {

///////////////////////////////////////////////////////////////////////////////
// DecodedStream

template <typename SourceStream, typename Encoding>
class DecodedStream {
public:
    DecodedStream(SourceStream& ss) : ss_(ss), codepoint_() { Decode(); }
    unsigned Peek() { return codepoint_; }
    unsigned Take() {
        unsigned c = codepoint_;
        if (c) // No further decoding when '\0'
            Decode();
        return c;
    }

private:
    void Decode() {
        if (!Encoding::Decode(ss_, &codepoint_))
            codepoint_ = 0;
    }

    SourceStream& ss_;
    unsigned codepoint_;
};

///////////////////////////////////////////////////////////////////////////////
// GenericRegex

static const SizeType kRe