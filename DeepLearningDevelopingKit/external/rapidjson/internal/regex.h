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
        while (ds.Peek() != 0) {
            switch (codepoint = ds.Take()) {
                case '^':
                    anchorBegin_ = true;
                    break;

                case '$':
                    anchorEnd_ = true;
                    break;

                case '|':
                    while (!operatorStack.Empty() && *operatorStack.template Top<Operator>() < kAlternation)
                        if (!Eval(operandStack, *operatorStack.template Pop<Operator>(1)))
                            return;
                    *operatorStack.template Push<Operator>() = kAlternation;
                    *atomCountStack.template Top<unsigned>() = 0;
                    break;

                case '(':
                    *operatorStack.template Push<Operator>() = kLeftParenthesis;
                    *atomCountStack.template Push<unsigned>() = 0;
                    break;

                case ')':
                    while (!operatorStack.Empty() && *operatorStack.template Top<Operator>() != kLeftParenthesis)
                        if (!Eval(operandStack, *operatorStack.template Pop<Operator>(1)))
                            return;
                    if (operatorStack.Empty())
                        return;
                    operatorStack.template Pop<Operator>(1);
                    atomCountStack.template Pop<unsigned>(1);
                    ImplicitConcatenation(atomCountStack, operatorStack);
                    break;

                case '?':
                    if (!Eval(operandStack, kZeroOrOne))
                        return;
                    break;

                case '*':
                    if (!Eval(operandStack, kZeroOrMore))
                        return;
                    break;

                case '+':
                    if (!Eval(operandStack, kOneOrMore))
                        return;
                    break;

                case '{':
                    {
                        unsigned n, m;
                        if (!ParseUnsigned(ds, &n))
                            return;

                        if (ds.Peek() == ',') {
                            ds.Take();
                            if (ds.Peek() == '}')
                                m = kInfinityQuantifier;
                            else if (!ParseUnsigned(ds, &m) || m < n)
                                return;
                        }
                        else
                            m = n;

                        if (!EvalQuantifier(operandStack, n, m) || ds.Peek() != '}')
                            return;
                        ds.Take();
                    }
                    break;

                case '.':
                    PushOperand(operandStack, kAnyCharacterClass);
                    ImplicitConcatenation(atomCountStack, operatorStack);
                    break;

                case '[':
                    {
                        SizeType range;
                        if (!ParseRange(ds, &range))
                            return;
                        SizeType s = NewState(kRegexInvalidState, kRegexInvalidState, kRangeCharacterClass);
                        GetState(s).rangeStart = range;
                        *operandStack.template Push<Frag>() = Frag(s, s, s);
                    }
                    ImplicitConcatenation(atomCountStack, operatorStack);
                    break;

                case '\\': // Escape character
                    if (!CharacterEscape(ds, &codepoint))
                        return; // Unsupported escape character
                    // fall through to default

                default: // Pattern character
                    PushOperand(operandStack, codepoint);
                    ImplicitConcatenation(atomCountStack, operatorStack);
            }
        }

        while (!operatorStack.Empty())
            if (!Eval(operandStack, *operatorStack.template Pop<Operator>(1)))
                return;

        // Link the operand to matching state.
        if (operandStack.GetSize() == sizeof(Frag)) {
            Frag* e = operandStack.template Pop<Frag>(1);
            Patch(e->out, NewState(kRegexInvalidState, kRegexInvalidState, 0));
            root_ = e->start;

#if RAPIDJSON_REGEX_VERBOSE
            printf("root: %d\n", root_);
            for (SizeType i = 0; i < stateCount_ ; i++) {
                State& s = GetState(i);
                printf("[%2d] out: %2d out1: %2d c: '%c'\n", i, s.out, s.out1, (char)s.codepoint);
            }
            printf("\n");
#endif
        }
    }

    SizeType NewState(SizeType out, SizeType out1, unsigned codepoint) {
        State* s = states_.template Push<State>();
        s->out = out;
        s->out1 = out1;
        s->codepoint = codepoint;
        s->rangeStart = kRegexInvalidRange;
        return stateCount_++;
    }

    void PushOperand(Stack<Allocator>& operandStack, unsigned codepoint) {
        SizeType s = NewState(kRegexInvalidState, kRegexInvalidState, codepoint);
        *operandStack.template Push<Frag>() = Frag(s, s, s);
    }

    void ImplicitConcatenation(Stack<Allocator>& atomCountStack, Stack<Allocator>& operatorStack) {
        if (*atomCountStack.template Top<unsigned>())
            *operatorStack.template Push<Operator>() = kConcatenation;
        (*atomCountStack.template Top<unsigned>())++;
    }

    SizeType Append(SizeType l1, SizeType l2) {
        SizeType old = l1;
        while (GetState(l1).out != kRegexInvalidState)
            l1 = GetState(l1).out;
        GetState(l1).out = l2;
        return old;
    }

    void Patch(SizeType l, SizeType s) {
        for (SizeType next; l != kRegexInvalidState; l = next) {
            next = GetState(l).out;
            GetState(l).out = s;
        }
    }

    bool Eval(Stack<Allocator>& operandStack, Operator op) {
        switch (op) {
            case kConcatenation:
                RAPIDJSON_ASSERT(operandStack.GetSize() >= sizeof(Frag) * 2);
                {
                    Frag e2 = *operandStack.template Pop<Frag>(1);
                    Frag e1 = *operandStack.template Pop<Frag>(1);
                    Patch(e1.out, e2.start);
                    *operandStack.template Push<Frag>() = Frag(e1.start, e2.out, Min(e1.minIndex, e2.minIndex));
                }
                return true;

            case kAlternation:
                if (operandStack.GetSize() >= sizeof(Frag) * 2) {
                    Frag e2 = *operandStack.template Pop<Frag>(1);
                    Frag e1 = *operandStack.template Pop<Frag>(1);
                    SizeType s = NewState(e1.start, e2.start, 0);
                    *operandStack.template Push<Frag>() = Frag(s, Append(e1.out, e2.out), Min(e1.minIndex, e2.minIndex));
                    return true;
                }
                return false;

            case kZeroOrOne:
                if (operandStack.GetSize() >= sizeof(Frag)) {
                    Frag e = *operandStack.template Pop<Frag>(1);
                    SizeType s = NewState(kRegexInvalidState, e.start, 0);
                    *operandStack.template Push<Frag>() = Frag(s, Append(e.out, s), e.minIndex);
                    return true;
                }
                return false;

            case kZeroOrMore:
                if (operandStack.GetSize() >= sizeof(Frag)) {
                    Frag e = *operandStack.template Pop<Frag>(1);
                    SizeType s = NewState(kRegexInvalidState, e.start, 0);
                    Patch(e.out, s);
                    *operandStack.template Push<Frag>() = Frag(s, s, e.minIndex);
                    return true;
                }
                return false;

            default: 
                RAPIDJSON_ASSERT(op == kOneOrMore);
                if (operandStack.GetSize() >= sizeof(Frag)) {
                    Frag e = *operandStack.template Pop<Frag>(1);
                    SizeType s = NewState(kRegexInvalidState, e.start, 0);
                    Patch(e.out, s);
                    *operandStack.template Push<Frag>() = Frag(e.start, s, e.minIndex);
                    return true;
                }
                return false;
        }
    }

    bool EvalQuantifier(Stack<Allocator>& operandStack, unsigned n, unsigned m) {
        RAPIDJSON_ASSERT(n <= m);
        RAPIDJSON_ASSERT(operandSta