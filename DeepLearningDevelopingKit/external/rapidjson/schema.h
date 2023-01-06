
// Tencent is pleased to support the open source community by making RapidJSON available->
// 
// Copyright (C) 2015 THL A29 Limited, a Tencent company, and Milo Yip-> All rights reserved->
//
// Licensed under the MIT License (the "License"); you may not use this file except
// in compliance with the License-> You may obtain a copy of the License at
//
// http://opensource->org/licenses/MIT
//
// Unless required by applicable law or agreed to in writing, software distributed 
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR 
// CONDITIONS OF ANY KIND, either express or implied-> See the License for the 
// specific language governing permissions and limitations under the License->

#ifndef RAPIDJSON_SCHEMA_H_
#define RAPIDJSON_SCHEMA_H_

#include "document.h"
#include "pointer.h"
#include "stringbuffer.h"
#include <cmath> // abs, floor

#if !defined(RAPIDJSON_SCHEMA_USE_INTERNALREGEX)
#define RAPIDJSON_SCHEMA_USE_INTERNALREGEX 1
#else
#define RAPIDJSON_SCHEMA_USE_INTERNALREGEX 0
#endif

#if !RAPIDJSON_SCHEMA_USE_INTERNALREGEX && defined(RAPIDJSON_SCHEMA_USE_STDREGEX) && (__cplusplus >=201103L || (defined(_MSC_VER) && _MSC_VER >= 1800))
#define RAPIDJSON_SCHEMA_USE_STDREGEX 1
#else
#define RAPIDJSON_SCHEMA_USE_STDREGEX 0
#endif

#if RAPIDJSON_SCHEMA_USE_INTERNALREGEX
#include "internal/regex.h"
#elif RAPIDJSON_SCHEMA_USE_STDREGEX
#include <regex>
#endif

#if RAPIDJSON_SCHEMA_USE_INTERNALREGEX || RAPIDJSON_SCHEMA_USE_STDREGEX
#define RAPIDJSON_SCHEMA_HAS_REGEX 1
#else
#define RAPIDJSON_SCHEMA_HAS_REGEX 0
#endif

#ifndef RAPIDJSON_SCHEMA_VERBOSE
#define RAPIDJSON_SCHEMA_VERBOSE 0
#endif

#if RAPIDJSON_SCHEMA_VERBOSE
#include "stringbuffer.h"
#endif

RAPIDJSON_DIAG_PUSH

#if defined(__GNUC__)
RAPIDJSON_DIAG_OFF(effc++)
#endif

#ifdef __clang__
RAPIDJSON_DIAG_OFF(weak-vtables)
RAPIDJSON_DIAG_OFF(exit-time-destructors)
RAPIDJSON_DIAG_OFF(c++98-compat-pedantic)
RAPIDJSON_DIAG_OFF(variadic-macros)
#elif defined(_MSC_VER)
RAPIDJSON_DIAG_OFF(4512) // assignment operator could not be generated
#endif

RAPIDJSON_NAMESPACE_BEGIN

///////////////////////////////////////////////////////////////////////////////
// Verbose Utilities

#if RAPIDJSON_SCHEMA_VERBOSE

namespace internal {

inline void PrintInvalidKeyword(const char* keyword) {
    printf("Fail keyword: %s\n", keyword);
}

inline void PrintInvalidKeyword(const wchar_t* keyword) {
    wprintf(L"Fail keyword: %ls\n", keyword);
}

inline void PrintInvalidDocument(const char* document) {
    printf("Fail document: %s\n\n", document);
}

inline void PrintInvalidDocument(const wchar_t* document) {
    wprintf(L"Fail document: %ls\n\n", document);
}

inline void PrintValidatorPointers(unsigned depth, const char* s, const char* d) {
    printf("S: %*s%s\nD: %*s%s\n\n", depth * 4, " ", s, depth * 4, " ", d);
}

inline void PrintValidatorPointers(unsigned depth, const wchar_t* s, const wchar_t* d) {
    wprintf(L"S: %*ls%ls\nD: %*ls%ls\n\n", depth * 4, L" ", s, depth * 4, L" ", d);
}

} // namespace internal

#endif // RAPIDJSON_SCHEMA_VERBOSE

///////////////////////////////////////////////////////////////////////////////
// RAPIDJSON_INVALID_KEYWORD_RETURN

#if RAPIDJSON_SCHEMA_VERBOSE
#define RAPIDJSON_INVALID_KEYWORD_VERBOSE(keyword) internal::PrintInvalidKeyword(keyword)
#else
#define RAPIDJSON_INVALID_KEYWORD_VERBOSE(keyword)
#endif

#define RAPIDJSON_INVALID_KEYWORD_RETURN(keyword)\
RAPIDJSON_MULTILINEMACRO_BEGIN\
    context.invalidKeyword = keyword.GetString();\
    RAPIDJSON_INVALID_KEYWORD_VERBOSE(keyword.GetString());\
    return false;\
RAPIDJSON_MULTILINEMACRO_END

///////////////////////////////////////////////////////////////////////////////
// Forward declarations

template <typename ValueType, typename Allocator>
class GenericSchemaDocument;

namespace internal {

template <typename SchemaDocumentType>
class Schema;

///////////////////////////////////////////////////////////////////////////////
// ISchemaValidator

class ISchemaValidator {
public:
    virtual ~ISchemaValidator() {}
    virtual bool IsValid() const = 0;
};

///////////////////////////////////////////////////////////////////////////////
// ISchemaStateFactory

template <typename SchemaType>
class ISchemaStateFactory {
public:
    virtual ~ISchemaStateFactory() {}
    virtual ISchemaValidator* CreateSchemaValidator(const SchemaType&) = 0;
    virtual void DestroySchemaValidator(ISchemaValidator* validator) = 0;
    virtual void* CreateHasher() = 0;
    virtual uint64_t GetHashCode(void* hasher) = 0;
    virtual void DestroryHasher(void* hasher) = 0;
    virtual void* MallocState(size_t size) = 0;
    virtual void FreeState(void* p) = 0;
};

///////////////////////////////////////////////////////////////////////////////
// IValidationErrorHandler

template <typename SchemaType>
class IValidationErrorHandler {
public:
    typedef typename SchemaType::Ch Ch;
    typedef typename SchemaType::SValue SValue;

    virtual ~IValidationErrorHandler() {}

    virtual void NotMultipleOf(int64_t actual, const SValue& expected) = 0;
    virtual void NotMultipleOf(uint64_t actual, const SValue& expected) = 0;
    virtual void NotMultipleOf(double actual, const SValue& expected) = 0;
    virtual void AboveMaximum(int64_t actual, const SValue& expected, bool exclusive) = 0;
    virtual void AboveMaximum(uint64_t actual, const SValue& expected, bool exclusive) = 0;
    virtual void AboveMaximum(double actual, const SValue& expected, bool exclusive) = 0;
    virtual void BelowMinimum(int64_t actual, const SValue& expected, bool exclusive) = 0;
    virtual void BelowMinimum(uint64_t actual, const SValue& expected, bool exclusive) = 0;
    virtual void BelowMinimum(double actual, const SValue& expected, bool exclusive) = 0;

    virtual void TooLong(const Ch* str, SizeType length, SizeType expected) = 0;
    virtual void TooShort(const Ch* str, SizeType length, SizeType expected) = 0;
    virtual void DoesNotMatch(const Ch* str, SizeType length) = 0;

    virtual void DisallowedItem(SizeType index) = 0;
    virtual void TooFewItems(SizeType actualCount, SizeType expectedCount) = 0;
    virtual void TooManyItems(SizeType actualCount, SizeType expectedCount) = 0;
    virtual void DuplicateItems(SizeType index1, SizeType index2) = 0;

    virtual void TooManyProperties(SizeType actualCount, SizeType expectedCount) = 0;
    virtual void TooFewProperties(SizeType actualCount, SizeType expectedCount) = 0;
    virtual void StartMissingProperties() = 0;
    virtual void AddMissingProperty(const SValue& name) = 0;
    virtual bool EndMissingProperties() = 0;
    virtual void PropertyViolations(ISchemaValidator** subvalidators, SizeType count) = 0;
    virtual void DisallowedProperty(const Ch* name, SizeType length) = 0;

    virtual void StartDependencyErrors() = 0;
    virtual void StartMissingDependentProperties() = 0;
    virtual void AddMissingDependentProperty(const SValue& targetName) = 0;
    virtual void EndMissingDependentProperties(const SValue& sourceName) = 0;
    virtual void AddDependencySchemaError(const SValue& souceName, ISchemaValidator* subvalidator) = 0;
    virtual bool EndDependencyErrors() = 0;

    virtual void DisallowedValue() = 0;
    virtual void StartDisallowedType() = 0;
    virtual void AddExpectedType(const typename SchemaType::ValueType& expectedType) = 0;
    virtual void EndDisallowedType(const typename SchemaType::ValueType& actualType) = 0;
    virtual void NotAllOf(ISchemaValidator** subvalidators, SizeType count) = 0;
    virtual void NoneOf(ISchemaValidator** subvalidators, SizeType count) = 0;
    virtual void NotOneOf(ISchemaValidator** subvalidators, SizeType count) = 0;
    virtual void Disallowed() = 0;
};


///////////////////////////////////////////////////////////////////////////////
// Hasher

// For comparison of compound value
template<typename Encoding, typename Allocator>
class Hasher {
public:
    typedef typename Encoding::Ch Ch;

    Hasher(Allocator* allocator = 0, size_t stackCapacity = kDefaultSize) : stack_(allocator, stackCapacity) {}

    bool Null() { return WriteType(kNullType); }
    bool Bool(bool b) { return WriteType(b ? kTrueType : kFalseType); }
    bool Int(int i) { Number n; n.u.i = i; n.d = static_cast<double>(i); return WriteNumber(n); }
    bool Uint(unsigned u) { Number n; n.u.u = u; n.d = static_cast<double>(u); return WriteNumber(n); }
    bool Int64(int64_t i) { Number n; n.u.i = i; n.d = static_cast<double>(i); return WriteNumber(n); }
    bool Uint64(uint64_t u) { Number n; n.u.u = u; n.d = static_cast<double>(u); return WriteNumber(n); }
    bool Double(double d) { 
        Number n; 
        if (d < 0) n.u.i = static_cast<int64_t>(d);
        else       n.u.u = static_cast<uint64_t>(d); 
        n.d = d;
        return WriteNumber(n);
    }

    bool RawNumber(const Ch* str, SizeType len, bool) {
        WriteBuffer(kNumberType, str, len * sizeof(Ch));
        return true;
    }

    bool String(const Ch* str, SizeType len, bool) {
        WriteBuffer(kStringType, str, len * sizeof(Ch));
        return true;
    }

    bool StartObject() { return true; }
    bool Key(const Ch* str, SizeType len, bool copy) { return String(str, len, copy); }
    bool EndObject(SizeType memberCount) { 
        uint64_t h = Hash(0, kObjectType);
        uint64_t* kv = stack_.template Pop<uint64_t>(memberCount * 2);
        for (SizeType i = 0; i < memberCount; i++)
            h ^= Hash(kv[i * 2], kv[i * 2 + 1]);  // Use xor to achieve member order insensitive
        *stack_.template Push<uint64_t>() = h;
        return true;
    }
    
    bool StartArray() { return true; }
    bool EndArray(SizeType elementCount) { 
        uint64_t h = Hash(0, kArrayType);
        uint64_t* e = stack_.template Pop<uint64_t>(elementCount);
        for (SizeType i = 0; i < elementCount; i++)
            h = Hash(h, e[i]); // Use hash to achieve element order sensitive
        *stack_.template Push<uint64_t>() = h;
        return true;
    }

    bool IsValid() const { return stack_.GetSize() == sizeof(uint64_t); }

    uint64_t GetHashCode() const {
        RAPIDJSON_ASSERT(IsValid());
        return *stack_.template Top<uint64_t>();
    }

private:
    static const size_t kDefaultSize = 256;
    struct Number {
        union U {
            uint64_t u;
            int64_t i;
        }u;
        double d;
    };

    bool WriteType(Type type) { return WriteBuffer(type, 0, 0); }
    
    bool WriteNumber(const Number& n) { return WriteBuffer(kNumberType, &n, sizeof(n)); }
    
    bool WriteBuffer(Type type, const void* data, size_t len) {
        // FNV-1a from http://isthe.com/chongo/tech/comp/fnv/
        uint64_t h = Hash(RAPIDJSON_UINT64_C2(0x84222325, 0xcbf29ce4), type);
        const unsigned char* d = static_cast<const unsigned char*>(data);
        for (size_t i = 0; i < len; i++)
            h = Hash(h, d[i]);
        *stack_.template Push<uint64_t>() = h;
        return true;
    }

    static uint64_t Hash(uint64_t h, uint64_t d) {
        static const uint64_t kPrime = RAPIDJSON_UINT64_C2(0x00000100, 0x000001b3);
        h ^= d;
        h *= kPrime;
        return h;
    }

    Stack<Allocator> stack_;
};

///////////////////////////////////////////////////////////////////////////////
// SchemaValidationContext

template <typename SchemaDocumentType>
struct SchemaValidationContext {
    typedef Schema<SchemaDocumentType> SchemaType;
    typedef ISchemaStateFactory<SchemaType> SchemaValidatorFactoryType;
    typedef IValidationErrorHandler<SchemaType> ErrorHandlerType;
    typedef typename SchemaType::ValueType ValueType;
    typedef typename ValueType::Ch Ch;

    enum PatternValidatorType {
        kPatternValidatorOnly,
        kPatternValidatorWithProperty,
        kPatternValidatorWithAdditionalProperty
    };

    SchemaValidationContext(SchemaValidatorFactoryType& f, ErrorHandlerType& eh, const SchemaType* s) :
        factory(f),
        error_handler(eh),
        schema(s),
        valueSchema(),
        invalidKeyword(),
        hasher(),
        arrayElementHashCodes(),
        validators(),
        validatorCount(),
        patternPropertiesValidators(),
        patternPropertiesValidatorCount(),
        patternPropertiesSchemas(),
        patternPropertiesSchemaCount(),
        valuePatternValidatorType(kPatternValidatorOnly),
        propertyExist(),
        inArray(false),
        valueUniqueness(false),
        arrayUniqueness(false)
    {
    }

    ~SchemaValidationContext() {
        if (hasher)
            factory.DestroryHasher(hasher);
        if (validators) {
            for (SizeType i = 0; i < validatorCount; i++)
                factory.DestroySchemaValidator(validators[i]);
            factory.FreeState(validators);
        }
        if (patternPropertiesValidators) {
            for (SizeType i = 0; i < patternPropertiesValidatorCount; i++)
                factory.DestroySchemaValidator(patternPropertiesValidators[i]);
            factory.FreeState(patternPropertiesValidators);
        }
        if (patternPropertiesSchemas)
            factory.FreeState(patternPropertiesSchemas);
        if (propertyExist)
            factory.FreeState(propertyExist);
    }

    SchemaValidatorFactoryType& factory;
    ErrorHandlerType& error_handler;
    const SchemaType* schema;
    const SchemaType* valueSchema;
    const Ch* invalidKeyword;
    void* hasher; // Only validator access
    void* arrayElementHashCodes; // Only validator access this
    ISchemaValidator** validators;
    SizeType validatorCount;
    ISchemaValidator** patternPropertiesValidators;
    SizeType patternPropertiesValidatorCount;
    const SchemaType** patternPropertiesSchemas;
    SizeType patternPropertiesSchemaCount;
    PatternValidatorType valuePatternValidatorType;
    PatternValidatorType objectPatternValidatorType;
    SizeType arrayElementIndex;
    bool* propertyExist;
    bool inArray;
    bool valueUniqueness;
    bool arrayUniqueness;
};

///////////////////////////////////////////////////////////////////////////////
// Schema

template <typename SchemaDocumentType>
class Schema {
public:
    typedef typename SchemaDocumentType::ValueType ValueType;
    typedef typename SchemaDocumentType::AllocatorType AllocatorType;
    typedef typename SchemaDocumentType::PointerType PointerType;
    typedef typename ValueType::EncodingType EncodingType;
    typedef typename EncodingType::Ch Ch;
    typedef SchemaValidationContext<SchemaDocumentType> Context;
    typedef Schema<SchemaDocumentType> SchemaType;
    typedef GenericValue<EncodingType, AllocatorType> SValue;
    typedef IValidationErrorHandler<Schema> ErrorHandler;
    friend class GenericSchemaDocument<ValueType, AllocatorType>;

    Schema(SchemaDocumentType* schemaDocument, const PointerType& p, const ValueType& value, const ValueType& document, AllocatorType* allocator) :
        allocator_(allocator),
        uri_(schemaDocument->GetURI(), *allocator),
        pointer_(p),
        typeless_(schemaDocument->GetTypeless()),
        enum_(),
        enumCount_(),
        not_(),
        type_((1 << kTotalSchemaType) - 1), // typeless
        validatorCount_(),
        notValidatorIndex_(),
        properties_(),
        additionalPropertiesSchema_(),
        patternProperties_(),
        patternPropertyCount_(),
        propertyCount_(),
        minProperties_(),
        maxProperties_(SizeType(~0)),
        additionalProperties_(true),
        hasDependencies_(),
        hasRequired_(),
        hasSchemaDependencies_(),
        additionalItemsSchema_(),
        itemsList_(),
        itemsTuple_(),
        itemsTupleCount_(),
        minItems_(),
        maxItems_(SizeType(~0)),
        additionalItems_(true),
        uniqueItems_(false),
        pattern_(),
        minLength_(0),
        maxLength_(~SizeType(0)),
        exclusiveMinimum_(false),
        exclusiveMaximum_(false)
    {
        typedef typename SchemaDocumentType::ValueType ValueType;
        typedef typename ValueType::ConstValueIterator ConstValueIterator;
        typedef typename ValueType::ConstMemberIterator ConstMemberIterator;

        if (!value.IsObject())
            return;

        if (const ValueType* v = GetMember(value, GetTypeString())) {
            type_ = 0;
            if (v->IsString())
                AddType(*v);
            else if (v->IsArray())
                for (ConstValueIterator itr = v->Begin(); itr != v->End(); ++itr)
                    AddType(*itr);
        }

        if (const ValueType* v = GetMember(value, GetEnumString()))
            if (v->IsArray() && v->Size() > 0) {
                enum_ = static_cast<uint64_t*>(allocator_->Malloc(sizeof(uint64_t) * v->Size()));
                for (ConstValueIterator itr = v->Begin(); itr != v->End(); ++itr) {
                    typedef Hasher<EncodingType, MemoryPoolAllocator<> > EnumHasherType;
                    char buffer[256 + 24];
                    MemoryPoolAllocator<> hasherAllocator(buffer, sizeof(buffer));
                    EnumHasherType h(&hasherAllocator, 256);
                    itr->Accept(h);
                    enum_[enumCount_++] = h.GetHashCode();
                }
            }

        if (schemaDocument) {
            AssignIfExist(allOf_, *schemaDocument, p, value, GetAllOfString(), document);
            AssignIfExist(anyOf_, *schemaDocument, p, value, GetAnyOfString(), document);
            AssignIfExist(oneOf_, *schemaDocument, p, value, GetOneOfString(), document);
        }

        if (const ValueType* v = GetMember(value, GetNotString())) {
            schemaDocument->CreateSchema(&not_, p.Append(GetNotString(), allocator_), *v, document);
            notValidatorIndex_ = validatorCount_;
            validatorCount_++;
        }

        // Object

        const ValueType* properties = GetMember(value, GetPropertiesString());
        const ValueType* required = GetMember(value, GetRequiredString());
        const ValueType* dependencies = GetMember(value, GetDependenciesString());
        {
            // Gather properties from properties/required/dependencies
            SValue allProperties(kArrayType);

            if (properties && properties->IsObject())
                for (ConstMemberIterator itr = properties->MemberBegin(); itr != properties->MemberEnd(); ++itr)
                    AddUniqueElement(allProperties, itr->name);
            
            if (required && required->IsArray())
                for (ConstValueIterator itr = required->Begin(); itr != required->End(); ++itr)
                    if (itr->IsString())
                        AddUniqueElement(allProperties, *itr);

            if (dependencies && dependencies->IsObject())
                for (ConstMemberIterator itr = dependencies->MemberBegin(); itr != dependencies->MemberEnd(); ++itr) {
                    AddUniqueElement(allProperties, itr->name);
                    if (itr->value.IsArray())
                        for (ConstValueIterator i = itr->value.Begin(); i != itr->value.End(); ++i)
                            if (i->IsString())
                                AddUniqueElement(allProperties, *i);
                }

            if (allProperties.Size() > 0) {
                propertyCount_ = allProperties.Size();
                properties_ = static_cast<Property*>(allocator_->Malloc(sizeof(Property) * propertyCount_));
                for (SizeType i = 0; i < propertyCount_; i++) {
                    new (&properties_[i]) Property();
                    properties_[i].name = allProperties[i];
                    properties_[i].schema = typeless_;
                }
            }
        }

        if (properties && properties->IsObject()) {
            PointerType q = p.Append(GetPropertiesString(), allocator_);
            for (ConstMemberIterator itr = properties->MemberBegin(); itr != properties->MemberEnd(); ++itr) {
                SizeType index;
                if (FindPropertyIndex(itr->name, &index))
                    schemaDocument->CreateSchema(&properties_[index].schema, q.Append(itr->name, allocator_), itr->value, document);
            }
        }

        if (const ValueType* v = GetMember(value, GetPatternPropertiesString())) {
            PointerType q = p.Append(GetPatternPropertiesString(), allocator_);
            patternProperties_ = static_cast<PatternProperty*>(allocator_->Malloc(sizeof(PatternProperty) * v->MemberCount()));
            patternPropertyCount_ = 0;

            for (ConstMemberIterator itr = v->MemberBegin(); itr != v->MemberEnd(); ++itr) {
                new (&patternProperties_[patternPropertyCount_]) PatternProperty();
                patternProperties_[patternPropertyCount_].pattern = CreatePattern(itr->name);
                schemaDocument->CreateSchema(&patternProperties_[patternPropertyCount_].schema, q.Append(itr->name, allocator_), itr->value, document);
                patternPropertyCount_++;
            }
        }

        if (required && required->IsArray())
            for (ConstValueIterator itr = required->Begin(); itr != required->End(); ++itr)
                if (itr->IsString()) {
                    SizeType index;
                    if (FindPropertyIndex(*itr, &index)) {
                        properties_[index].required = true;
                        hasRequired_ = true;
                    }
                }

        if (dependencies && dependencies->IsObject()) {
            PointerType q = p.Append(GetDependenciesString(), allocator_);
            hasDependencies_ = true;
            for (ConstMemberIterator itr = dependencies->MemberBegin(); itr != dependencies->MemberEnd(); ++itr) {
                SizeType sourceIndex;
                if (FindPropertyIndex(itr->name, &sourceIndex)) {
                    if (itr->value.IsArray()) {
                        properties_[sourceIndex].dependencies = static_cast<bool*>(allocator_->Malloc(sizeof(bool) * propertyCount_));
                        std::memset(properties_[sourceIndex].dependencies, 0, sizeof(bool)* propertyCount_);
                        for (ConstValueIterator targetItr = itr->value.Begin(); targetItr != itr->value.End(); ++targetItr) {
                            SizeType targetIndex;
                            if (FindPropertyIndex(*targetItr, &targetIndex))
                                properties_[sourceIndex].dependencies[targetIndex] = true;
                        }
                    }
                    else if (itr->value.IsObject()) {
                        hasSchemaDependencies_ = true;
                        schemaDocument->CreateSchema(&properties_[sourceIndex].dependenciesSchema, q.Append(itr->name, allocator_), itr->value, document);
                        properties_[sourceIndex].dependenciesValidatorIndex = validatorCount_;
                        validatorCount_++;
                    }
                }
            }
        }

        if (const ValueType* v = GetMember(value, GetAdditionalPropertiesString())) {
            if (v->IsBool())
                additionalProperties_ = v->GetBool();
            else if (v->IsObject())
                schemaDocument->CreateSchema(&additionalPropertiesSchema_, p.Append(GetAdditionalPropertiesString(), allocator_), *v, document);
        }

        AssignIfExist(minProperties_, value, GetMinPropertiesString());
        AssignIfExist(maxProperties_, value, GetMaxPropertiesString());

        // Array
        if (const ValueType* v = GetMember(value, GetItemsString())) {
            PointerType q = p.Append(GetItemsString(), allocator_);
            if (v->IsObject()) // List validation
                schemaDocument->CreateSchema(&itemsList_, q, *v, document);
            else if (v->IsArray()) { // Tuple validation
                itemsTuple_ = static_cast<const Schema**>(allocator_->Malloc(sizeof(const Schema*) * v->Size()));
                SizeType index = 0;
                for (ConstValueIterator itr = v->Begin(); itr != v->End(); ++itr, index++)
                    schemaDocument->CreateSchema(&itemsTuple_[itemsTupleCount_++], q.Append(index, allocator_), *itr, document);
            }
        }

        AssignIfExist(minItems_, value, GetMinItemsString());
        AssignIfExist(maxItems_, value, GetMaxItemsString());

        if (const ValueType* v = GetMember(value, GetAdditionalItemsString())) {
            if (v->IsBool())
                additionalItems_ = v->GetBool();
            else if (v->IsObject())
                schemaDocument->CreateSchema(&additionalItemsSchema_, p.Append(GetAdditionalItemsString(), allocator_), *v, document);
        }

        AssignIfExist(uniqueItems_, value, GetUniqueItemsString());

        // String
        AssignIfExist(minLength_, value, GetMinLengthString());
        AssignIfExist(maxLength_, value, GetMaxLengthString());

        if (const ValueType* v = GetMember(value, GetPatternString()))
            pattern_ = CreatePattern(*v);

        // Number
        if (const ValueType* v = GetMember(value, GetMinimumString()))
            if (v->IsNumber())
                minimum_.CopyFrom(*v, *allocator_);

        if (const ValueType* v = GetMember(value, GetMaximumString()))
            if (v->IsNumber())
                maximum_.CopyFrom(*v, *allocator_);

        AssignIfExist(exclusiveMinimum_, value, GetExclusiveMinimumString());
        AssignIfExist(exclusiveMaximum_, value, GetExclusiveMaximumString());

        if (const ValueType* v = GetMember(value, GetMultipleOfString()))
            if (v->IsNumber() && v->GetDouble() > 0.0)
                multipleOf_.CopyFrom(*v, *allocator_);
    }

    ~Schema() {
        AllocatorType::Free(enum_);
        if (properties_) {
            for (SizeType i = 0; i < propertyCount_; i++)
                properties_[i].~Property();
            AllocatorType::Free(properties_);
        }
        if (patternProperties_) {
            for (SizeType i = 0; i < patternPropertyCount_; i++)
                patternProperties_[i].~PatternProperty();
            AllocatorType::Free(patternProperties_);
        }
        AllocatorType::Free(itemsTuple_);
#if RAPIDJSON_SCHEMA_HAS_REGEX
        if (pattern_) {
            pattern_->~RegexType();
            AllocatorType::Free(pattern_);
        }
#endif
    }

    const SValue& GetURI() const {
        return uri_;
    }

    const PointerType& GetPointer() const {
        return pointer_;
    }

    bool BeginValue(Context& context) const {
        if (context.inArray) {
            if (uniqueItems_)
                context.valueUniqueness = true;

            if (itemsList_)
                context.valueSchema = itemsList_;
            else if (itemsTuple_) {
                if (context.arrayElementIndex < itemsTupleCount_)
                    context.valueSchema = itemsTuple_[context.arrayElementIndex];
                else if (additionalItemsSchema_)
                    context.valueSchema = additionalItemsSchema_;
                else if (additionalItems_)
                    context.valueSchema = typeless_;
                else {
                    context.error_handler.DisallowedItem(context.arrayElementIndex);
                    RAPIDJSON_INVALID_KEYWORD_RETURN(GetItemsString());
                }
            }
            else
                context.valueSchema = typeless_;

            context.arrayElementIndex++;
        }
        return true;
    }

    RAPIDJSON_FORCEINLINE bool EndValue(Context& context) const {
        if (context.patternPropertiesValidatorCount > 0) {
            bool otherValid = false;
            SizeType count = context.patternPropertiesValidatorCount;
            if (context.objectPatternValidatorType != Context::kPatternValidatorOnly)
                otherValid = context.patternPropertiesValidators[--count]->IsValid();

            bool patternValid = true;
            for (SizeType i = 0; i < count; i++)
                if (!context.patternPropertiesValidators[i]->IsValid()) {
                    patternValid = false;
                    break;
                }

            if (context.objectPatternValidatorType == Context::kPatternValidatorOnly) {
                if (!patternValid) {
                    context.error_handler.PropertyViolations(context.patternPropertiesValidators, count);
                    RAPIDJSON_INVALID_KEYWORD_RETURN(GetPatternPropertiesString());
                }
            }
            else if (context.objectPatternValidatorType == Context::kPatternValidatorWithProperty) {
                if (!patternValid || !otherValid) {
                    context.error_handler.PropertyViolations(context.patternPropertiesValidators, count + 1);
                    RAPIDJSON_INVALID_KEYWORD_RETURN(GetPatternPropertiesString());
                }
            }
            else if (!patternValid && !otherValid) { // kPatternValidatorWithAdditionalProperty)
                context.error_handler.PropertyViolations(context.patternPropertiesValidators, count + 1);
                RAPIDJSON_INVALID_KEYWORD_RETURN(GetPatternPropertiesString());
            }
        }

        if (enum_) {
            const uint64_t h = context.factory.GetHashCode(context.hasher);
            for (SizeType i = 0; i < enumCount_; i++)
                if (enum_[i] == h)
                    goto foundEnum;
            context.error_handler.DisallowedValue();
            RAPIDJSON_INVALID_KEYWORD_RETURN(GetEnumString());
            foundEnum:;
        }

        if (allOf_.schemas)
            for (SizeType i = allOf_.begin; i < allOf_.begin + allOf_.count; i++)
                if (!context.validators[i]->IsValid()) {
                    context.error_handler.NotAllOf(&context.validators[allOf_.begin], allOf_.count);
                    RAPIDJSON_INVALID_KEYWORD_RETURN(GetAllOfString());
                }
        
        if (anyOf_.schemas) {
            for (SizeType i = anyOf_.begin; i < anyOf_.begin + anyOf_.count; i++)
                if (context.validators[i]->IsValid())
                    goto foundAny;
            context.error_handler.NoneOf(&context.validators[anyOf_.begin], anyOf_.count);
            RAPIDJSON_INVALID_KEYWORD_RETURN(GetAnyOfString());
            foundAny:;
        }

        if (oneOf_.schemas) {
            bool oneValid = false;
            for (SizeType i = oneOf_.begin; i < oneOf_.begin + oneOf_.count; i++)
                if (context.validators[i]->IsValid()) {
                    if (oneValid) {
                        context.error_handler.NotOneOf(&context.validators[oneOf_.begin], oneOf_.count);
                        RAPIDJSON_INVALID_KEYWORD_RETURN(GetOneOfString());
                    } else
                        oneValid = true;
                }
            if (!oneValid) {
                context.error_handler.NotOneOf(&context.validators[oneOf_.begin], oneOf_.count);
                RAPIDJSON_INVALID_KEYWORD_RETURN(GetOneOfString());
            }
        }

        if (not_ && context.validators[notValidatorIndex_]->IsValid()) {
            context.error_handler.Disallowed();
            RAPIDJSON_INVALID_KEYWORD_RETURN(GetNotString());
        }

        return true;
    }

    bool Null(Context& context) const {
        if (!(type_ & (1 << kNullSchemaType))) {
            DisallowedType(context, GetNullString());
            RAPIDJSON_INVALID_KEYWORD_RETURN(GetTypeString());
        }
        return CreateParallelValidator(context);
    }
    
    bool Bool(Context& context, bool) const {
        if (!(type_ & (1 << kBooleanSchemaType))) {
            DisallowedType(context, GetBooleanString());
            RAPIDJSON_INVALID_KEYWORD_RETURN(GetTypeString());
        }
        return CreateParallelValidator(context);
    }

    bool Int(Context& context, int i) const {
        if (!CheckInt(context, i))
            return false;
        return CreateParallelValidator(context);
    }

    bool Uint(Context& context, unsigned u) const {
        if (!CheckUint(context, u))
            return false;
        return CreateParallelValidator(context);
    }

    bool Int64(Context& context, int64_t i) const {
        if (!CheckInt(context, i))
            return false;
        return CreateParallelValidator(context);
    }

    bool Uint64(Context& context, uint64_t u) const {
        if (!CheckUint(context, u))
            return false;
        return CreateParallelValidator(context);
    }

    bool Double(Context& context, double d) const {
        if (!(type_ & (1 << kNumberSchemaType))) {
            DisallowedType(context, GetNumberString());
            RAPIDJSON_INVALID_KEYWORD_RETURN(GetTypeString());
        }

        if (!minimum_.IsNull() && !CheckDoubleMinimum(context, d))
            return false;

        if (!maximum_.IsNull() && !CheckDoubleMaximum(context, d))
            return false;
        
        if (!multipleOf_.IsNull() && !CheckDoubleMultipleOf(context, d))
            return false;
        
        return CreateParallelValidator(context);
    }
    
    bool String(Context& context, const Ch* str, SizeType length, bool) const {
        if (!(type_ & (1 << kStringSchemaType))) {
            DisallowedType(context, GetStringString());
            RAPIDJSON_INVALID_KEYWORD_RETURN(GetTypeString());
        }

        if (minLength_ != 0 || maxLength_ != SizeType(~0)) {
            SizeType count;
            if (internal::CountStringCodePoint<EncodingType>(str, length, &count)) {
                if (count < minLength_) {
                    context.error_handler.TooShort(str, length, minLength_);
                    RAPIDJSON_INVALID_KEYWORD_RETURN(GetMinLengthString());
                }
                if (count > maxLength_) {
                    context.error_handler.TooLong(str, length, maxLength_);
                    RAPIDJSON_INVALID_KEYWORD_RETURN(GetMaxLengthString());
                }
            }
        }

        if (pattern_ && !IsPatternMatch(pattern_, str, length)) {
            context.error_handler.DoesNotMatch(str, length);
            RAPIDJSON_INVALID_KEYWORD_RETURN(GetPatternString());
        }

        return CreateParallelValidator(context);
    }

    bool StartObject(Context& context) const {
        if (!(type_ & (1 << kObjectSchemaType))) {
            DisallowedType(context, GetObjectString());
            RAPIDJSON_INVALID_KEYWORD_RETURN(GetTypeString());
        }

        if (hasDependencies_ || hasRequired_) {
            context.propertyExist = static_cast<bool*>(context.factory.MallocState(sizeof(bool) * propertyCount_));
            std::memset(context.propertyExist, 0, sizeof(bool) * propertyCount_);
        }

        if (patternProperties_) { // pre-allocate schema array
            SizeType count = patternPropertyCount_ + 1; // extra for valuePatternValidatorType
            context.patternPropertiesSchemas = static_cast<const SchemaType**>(context.factory.MallocState(sizeof(const SchemaType*) * count));
            context.patternPropertiesSchemaCount = 0;
            std::memset(context.patternPropertiesSchemas, 0, sizeof(SchemaType*) * count);
        }

        return CreateParallelValidator(context);
    }
    
    bool Key(Context& context, const Ch* str, SizeType len, bool) const {
        if (patternProperties_) {
            context.patternPropertiesSchemaCount = 0;
            for (SizeType i = 0; i < patternPropertyCount_; i++)
                if (patternProperties_[i].pattern && IsPatternMatch(patternProperties_[i].pattern, str, len)) {
                    context.patternPropertiesSchemas[context.patternPropertiesSchemaCount++] = patternProperties_[i].schema;
                    context.valueSchema = typeless_;
                }
        }

        SizeType index;
        if (FindPropertyIndex(ValueType(str, len).Move(), &index)) {
            if (context.patternPropertiesSchemaCount > 0) {
                context.patternPropertiesSchemas[context.patternPropertiesSchemaCount++] = properties_[index].schema;
                context.valueSchema = typeless_;
                context.valuePatternValidatorType = Context::kPatternValidatorWithProperty;
            }
            else
                context.valueSchema = properties_[index].schema;

            if (context.propertyExist)
                context.propertyExist[index] = true;

            return true;
        }

        if (additionalPropertiesSchema_) {
            if (additionalPropertiesSchema_ && context.patternPropertiesSchemaCount > 0) {
                context.patternPropertiesSchemas[context.patternPropertiesSchemaCount++] = additionalPropertiesSchema_;
                context.valueSchema = typeless_;
                context.valuePatternValidatorType = Context::kPatternValidatorWithAdditionalProperty;
            }
            else
                context.valueSchema = additionalPropertiesSchema_;
            return true;
        }
        else if (additionalProperties_) {
            context.valueSchema = typeless_;
            return true;
        }

        if (context.patternPropertiesSchemaCount == 0) { // patternProperties are not additional properties
            context.error_handler.DisallowedProperty(str, len);
            RAPIDJSON_INVALID_KEYWORD_RETURN(GetAdditionalPropertiesString());
        }

        return true;
    }

    bool EndObject(Context& context, SizeType memberCount) const {
        if (hasRequired_) {
            context.error_handler.StartMissingProperties();
            for (SizeType index = 0; index < propertyCount_; index++)
                if (properties_[index].required && !context.propertyExist[index])
                    context.error_handler.AddMissingProperty(properties_[index].name);
            if (context.error_handler.EndMissingProperties())
                RAPIDJSON_INVALID_KEYWORD_RETURN(GetRequiredString());
        }

        if (memberCount < minProperties_) {
            context.error_handler.TooFewProperties(memberCount, minProperties_);
            RAPIDJSON_INVALID_KEYWORD_RETURN(GetMinPropertiesString());
        }

        if (memberCount > maxProperties_) {
            context.error_handler.TooManyProperties(memberCount, maxProperties_);
            RAPIDJSON_INVALID_KEYWORD_RETURN(GetMaxPropertiesString());
        }

        if (hasDependencies_) {
            context.error_handler.StartDependencyErrors();
            for (SizeType sourceIndex = 0; sourceIndex < propertyCount_; sourceIndex++) {
                const Property& source = properties_[sourceIndex];
                if (context.propertyExist[sourceIndex]) {
                    if (source.dependencies) {
                        context.error_handler.StartMissingDependentProperties();
                        for (SizeType targetIndex = 0; targetIndex < propertyCount_; targetIndex++)
                            if (source.dependencies[targetIndex] && !context.propertyExist[targetIndex])
                                context.error_handler.AddMissingDependentProperty(properties_[targetIndex].name);
                        context.error_handler.EndMissingDependentProperties(source.name);
                    }
                    else if (source.dependenciesSchema) {
                        ISchemaValidator* dependenciesValidator = context.validators[source.dependenciesValidatorIndex];
                        if (!dependenciesValidator->IsValid())
                            context.error_handler.AddDependencySchemaError(source.name, dependenciesValidator);
                    }
                }
            }
            if (context.error_handler.EndDependencyErrors())
                RAPIDJSON_INVALID_KEYWORD_RETURN(GetDependenciesString());
        }

        return true;
    }

    bool StartArray(Context& context) const {
        if (!(type_ & (1 << kArraySchemaType))) {
            DisallowedType(context, GetArrayString());
            RAPIDJSON_INVALID_KEYWORD_RETURN(GetTypeString());
        }

        context.arrayElementIndex = 0;
        context.inArray = true;

        return CreateParallelValidator(context);
    }

    bool EndArray(Context& context, SizeType elementCount) const {
        context.inArray = false;
        
        if (elementCount < minItems_) {
            context.error_handler.TooFewItems(elementCount, minItems_);
            RAPIDJSON_INVALID_KEYWORD_RETURN(GetMinItemsString());
        }
        
        if (elementCount > maxItems_) {
            context.error_handler.TooManyItems(elementCount, maxItems_);
            RAPIDJSON_INVALID_KEYWORD_RETURN(GetMaxItemsString());
        }

        return true;
    }

    // Generate functions for string literal according to Ch
#define RAPIDJSON_STRING_(name, ...) \
    static const ValueType& Get##name##String() {\
        static const Ch s[] = { __VA_ARGS__, '\0' };\
        static const ValueType v(s, static_cast<SizeType>(sizeof(s) / sizeof(Ch) - 1));\
        return v;\
    }

    RAPIDJSON_STRING_(Null, 'n', 'u', 'l', 'l')
    RAPIDJSON_STRING_(Boolean, 'b', 'o', 'o', 'l', 'e', 'a', 'n')
    RAPIDJSON_STRING_(Object, 'o', 'b', 'j', 'e', 'c', 't')
    RAPIDJSON_STRING_(Array, 'a', 'r', 'r', 'a', 'y')
    RAPIDJSON_STRING_(String, 's', 't', 'r', 'i', 'n', 'g')
    RAPIDJSON_STRING_(Number, 'n', 'u', 'm', 'b', 'e', 'r')
    RAPIDJSON_STRING_(Integer, 'i', 'n', 't', 'e', 'g', 'e', 'r')
    RAPIDJSON_STRING_(Type, 't', 'y', 'p', 'e')
    RAPIDJSON_STRING_(Enum, 'e', 'n', 'u', 'm')
    RAPIDJSON_STRING_(AllOf, 'a', 'l', 'l', 'O', 'f')
    RAPIDJSON_STRING_(AnyOf, 'a', 'n', 'y', 'O', 'f')
    RAPIDJSON_STRING_(OneOf, 'o', 'n', 'e', 'O', 'f')
    RAPIDJSON_STRING_(Not, 'n', 'o', 't')
    RAPIDJSON_STRING_(Properties, 'p', 'r', 'o', 'p', 'e', 'r', 't', 'i', 'e', 's')
    RAPIDJSON_STRING_(Required, 'r', 'e', 'q', 'u', 'i', 'r', 'e', 'd')
    RAPIDJSON_STRING_(Dependencies, 'd', 'e', 'p', 'e', 'n', 'd', 'e', 'n', 'c', 'i', 'e', 's')
    RAPIDJSON_STRING_(PatternProperties, 'p', 'a', 't', 't', 'e', 'r', 'n', 'P', 'r', 'o', 'p', 'e', 'r', 't', 'i', 'e', 's')
    RAPIDJSON_STRING_(AdditionalProperties, 'a', 'd', 'd', 'i', 't', 'i', 'o', 'n', 'a', 'l', 'P', 'r', 'o', 'p', 'e', 'r', 't', 'i', 'e', 's')
    RAPIDJSON_STRING_(MinProperties, 'm', 'i', 'n', 'P', 'r', 'o', 'p', 'e', 'r', 't', 'i', 'e', 's')
    RAPIDJSON_STRING_(MaxProperties, 'm', 'a', 'x', 'P', 'r', 'o', 'p', 'e', 'r', 't', 'i', 'e', 's')
    RAPIDJSON_STRING_(Items, 'i', 't', 'e', 'm', 's')
    RAPIDJSON_STRING_(MinItems, 'm', 'i', 'n', 'I', 't', 'e', 'm', 's')
    RAPIDJSON_STRING_(MaxItems, 'm', 'a', 'x', 'I', 't', 'e', 'm', 's')
    RAPIDJSON_STRING_(AdditionalItems, 'a', 'd', 'd', 'i', 't', 'i', 'o', 'n', 'a', 'l', 'I', 't', 'e', 'm', 's')
    RAPIDJSON_STRING_(UniqueItems, 'u', 'n', 'i', 'q', 'u', 'e', 'I', 't', 'e', 'm', 's')
    RAPIDJSON_STRING_(MinLength, 'm', 'i', 'n', 'L', 'e', 'n', 'g', 't', 'h')
    RAPIDJSON_STRING_(MaxLength, 'm', 'a', 'x', 'L', 'e', 'n', 'g', 't', 'h')
    RAPIDJSON_STRING_(Pattern, 'p', 'a', 't', 't', 'e', 'r', 'n')
    RAPIDJSON_STRING_(Minimum, 'm', 'i', 'n', 'i', 'm', 'u', 'm')
    RAPIDJSON_STRING_(Maximum, 'm', 'a', 'x', 'i', 'm', 'u', 'm')
    RAPIDJSON_STRING_(ExclusiveMinimum, 'e', 'x', 'c', 'l', 'u', 's', 'i', 'v', 'e', 'M', 'i', 'n', 'i', 'm', 'u', 'm')
    RAPIDJSON_STRING_(ExclusiveMaximum, 'e', 'x', 'c', 'l', 'u', 's', 'i', 'v', 'e', 'M', 'a', 'x', 'i', 'm', 'u', 'm')
    RAPIDJSON_STRING_(MultipleOf, 'm', 'u', 'l', 't', 'i', 'p', 'l', 'e', 'O', 'f')

#undef RAPIDJSON_STRING_

private:
    enum SchemaValueType {
        kNullSchemaType,
        kBooleanSchemaType,
        kObjectSchemaType,
        kArraySchemaType,
        kStringSchemaType,
        kNumberSchemaType,
        kIntegerSchemaType,
        kTotalSchemaType
    };

#if RAPIDJSON_SCHEMA_USE_INTERNALREGEX
        typedef internal::GenericRegex<EncodingType, AllocatorType> RegexType;
#elif RAPIDJSON_SCHEMA_USE_STDREGEX
        typedef std::basic_regex<Ch> RegexType;
#else
        typedef char RegexType;
#endif

    struct SchemaArray {
        SchemaArray() : schemas(), count() {}
        ~SchemaArray() { AllocatorType::Free(schemas); }
        const SchemaType** schemas;
        SizeType begin; // begin index of context.validators
        SizeType count;
    };

    template <typename V1, typename V2>
    void AddUniqueElement(V1& a, const V2& v) {
        for (typename V1::ConstValueIterator itr = a.Begin(); itr != a.End(); ++itr)
            if (*itr == v)
                return;
        V1 c(v, *allocator_);
        a.PushBack(c, *allocator_);
    }

    static const ValueType* GetMember(const ValueType& value, const ValueType& name) {
        typename ValueType::ConstMemberIterator itr = value.FindMember(name);
        return itr != value.MemberEnd() ? &(itr->value) : 0;
    }

    static void AssignIfExist(bool& out, const ValueType& value, const ValueType& name) {
        if (const ValueType* v = GetMember(value, name))
            if (v->IsBool())
                out = v->GetBool();
    }

    static void AssignIfExist(SizeType& out, const ValueType& value, const ValueType& name) {
        if (const ValueType* v = GetMember(value, name))
            if (v->IsUint64() && v->GetUint64() <= SizeType(~0))
                out = static_cast<SizeType>(v->GetUint64());
    }

    void AssignIfExist(SchemaArray& out, SchemaDocumentType& schemaDocument, const PointerType& p, const ValueType& value, const ValueType& name, const ValueType& document) {
        if (const ValueType* v = GetMember(value, name)) {
            if (v->IsArray() && v->Size() > 0) {
                PointerType q = p.Append(name, allocator_);
                out.count = v->Size();
                out.schemas = static_cast<const Schema**>(allocator_->Malloc(out.count * sizeof(const Schema*)));
                memset(out.schemas, 0, sizeof(Schema*)* out.count);
                for (SizeType i = 0; i < out.count; i++)
                    schemaDocument.CreateSchema(&out.schemas[i], q.Append(i, allocator_), (*v)[i], document);
                out.begin = validatorCount_;
                validatorCount_ += out.count;