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

static const SizeType kRegexInvalidState = ~SizeType(0);  //!< Represents an invalid index in GenericRegex::State::out, out1
static const SizeType kRegexInvalidRange = ~SizeType(0);

template <typename Encoding, typename Allocator>
class GenericRegexSearch;

//! Regular expression engine with subset of ECMAscript grammar.
/*!
    Supported regular expression syntax:
    - \c ab     Concatenation
    - \c a|b    Alternation
    - \c a?     Zero or one
    - \c a*     Zero or more
    - \c a+     One or more
    - \c a{3}   Exactly 3 times
    - \c a{3,}  At least 3 times
    - \c a{3,5} 3 to 5 times
    - \c (ab)   Grouping
    - \c ^a     At the beginning
    - \c a$     At the end
    - \c .      Any character
    - \c [abc]  Character classes
    - \c [a-c]  Character class range
    - \c [a-z0-9_] Character class combination
    - \c [^abc] Negated character classes
    - \c [^a-c] Negated character class range
    - \c [\b]   Backspace (U+0008)
    - \c \\| \\\\ ...  Escape characters
    - \c \\f Form feed (U+000C)
    - \c \\n Line feed (U+000A)
    - \c \\r Carriage return (U+000D)
    - \c \\t Tab (U+0009)
    - \c \\v Vertical tab (U+000B)

    \note This is a Thompson NFA engine, implemented with reference to 
        Cox, Russ. "Regular Expression Matching Can Be Simple And Fast (but is slow in Java, Perl, PHP, Python, Ruby,...).", 
        https://swtch.com/~rsc/regexp/regexp1.html 
*/
template <typename Encoding, typename Allocator = CrtAllocator>
class GenericRegex {
public:
    typedef Encoding EncodingType;
    typedef typename Encoding::Ch Ch;
    template <typename, typename> friend class GenericRegexSearch;

    GenericRegex(const Ch* source, Allocator* allocator = 0) : 
        states_(allocator, 256), ranges_(allocator, 256), root_(kRegexInvalidState), stateCount_(), rangeCount_(), 
        anchorBegin_(), anchorEnd_()
    {
        GenericStringStream<Encoding> ss(source);
        DecodedStream<GenericStringStream<Encoding>, Encoding> ds(ss);
        Parse(ds);
    }

    ~GenericRegex() {}

    bool IsValid() const {
        return root_ != kRegexInvalidState;
    }

private:
    enum Operator {
        kZeroOrOne,
        kZeroOrMore,
        kOneOrMore,
        kConcatenation,
        kAlternation,
        kLeftParenthesis
    };

    static const unsigned kAnyCharacterClass = 0xFFFFFFFF;   //!< For '.'
    static const unsigned kRangeCharacterClass = 0xFFFFFFFE;
    static const unsigned kRangeNegationFlag = 0x80000000;

    struct Range {
        unsigned start; // 
        unsigned end;
        SizeType next;
    };

    struct State {
        SizeType out;     //!< Equals to kInvalid for matching state
        SizeType out1;    //!< Equals to non-kInvalid for split
        SizeType rangeStart;
        unsigned codepoint;
    };

    struct Frag {
        Frag(SizeType s, SizeType o, SizeType m) : start(s), out(o), minIndex(m) {}
        SizeType start;
        SizeType out; //!< link-list of all output states
        SizeType minIndex;
    };

    State& GetState(SizeType index) {
        RAPIDJSON_ASSERT(index < stateCount_);
        return states_.template Bottom<State>()[index];
    }

    const State& GetState(SizeType index) const {
        RAPIDJSON_ASSERT(index < stateCount_);
        return states_.template Bottom<State>()[index];
    }

    Range& GetRange(SizeType index) {
        RAPIDJSON_ASSERT(index < rangeCount_);
        return ranges_.template Bottom<Range>()[index];
    }

    const Range& GetRange(SizeType index) const {
        RAPIDJSON_ASSERT(index < rangeCount_);
        return ranges_.template Bottom<Range>()[index];
    }

    template <typename InputStream>
    void Parse(DecodedStream<InputStream, Encoding>& ds) {
        Allocator allocator;
        Stack<Allocator> operandStack(&allocator, 256);     // Frag
        Stack<Allocator> operatorStack(&allocator, 256);    // Operator
        Stack<Allocator> atomCountStack(&allocator, 256);   // unsigned (Atom per parenthesis)

        *atomCountStack.template Push<unsigned>() = 0;

        unsigned codepoint;
        while (ds.Peek() !=