/*
 *  Copyright 2015 The WebRTC project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "sdk/android/native_api/jni/java_types.h"

#include <memory>
#include <string>
#include <utility>

#include "sdk/android/generated_external_classes_jni/ArrayList_jni.h"
#include "sdk/android/generated_external_classes_jni/Boolean_jni.h"
#include "sdk/android/generated_external_classes_jni/Double_jni.h"
#include "sdk/android/generated_external_classes_jni/Enum_jni.h"
#include "sdk/android/generated_external_classes_jni/Integer_jni.h"
#include "sdk/android/generated_external_classes_jni/Iterable_jni.h"
#include "sdk/android/generated_external_classes_jni/Iterator_jni.h"
#include "sdk/android/generated_external_classes_jni/LinkedHashMap_jni.h"
#include "sdk/android/generated_external_classes_jni/Long_jni.h"
#include "sdk/android/generated_external_classes_jni/Map_jni.h"
#include "sdk/android/generated_native_api_jni/JniHelper_jni.h"
#include "third_party/jni_zero/jni_zero.h"

namespace webrtc {

Iterable::Iterable(JNIEnv* jni, const jni_zero::JavaRef<jobject>& iterable)
    : jni_(jni), iterable_(jni, iterable) {}

Iterable::Iterable(Iterable&& other) = default;

Iterable::~Iterable() = default;

// Creates an iterator representing the end of any collection.
Iterable::Iterator::Iterator() = default;

// Creates an iterator pointing to the beginning of the specified collection.
Iterable::Iterator::Iterator(JNIEnv* jni,
                             const jni_zero::JavaRef<jobject>& iterable)
    : jni_(jni) {
  iterator_ = JNI_Iterable::Java_Iterable_iterator(jni, iterable);
  RTC_CHECK(!iterator_.is_null());
  // Start at the first element in the collection.
  ++(*this);
}

// Move constructor - necessary to be able to return iterator types from
// functions.
Iterable::Iterator::Iterator(Iterator&& other)
    : jni_(std::move(other.jni_)),
      iterator_(std::move(other.iterator_)),
      value_(std::move(other.value_)) {
  RTC_DCHECK_RUN_ON(&thread_checker_);
}

Iterable::Iterator::~Iterator() = default;

// Advances the iterator one step.
Iterable::Iterator& Iterable::Iterator::operator++() {
  RTC_DCHECK_RUN_ON(&thread_checker_);
  if (AtEnd()) {
    // Can't move past the end.
    return *this;
  }
  bool has_next = JNI_Iterator::Java_Iterator_hasNext(jni_, iterator_);
  if (!has_next) {
    iterator_ = nullptr;
    value_ = nullptr;
    return *this;
  }

  value_ = JNI_Iterator::Java_Iterator_next(jni_, iterator_);
  return *this;
}

void Iterable::Iterator::Remove() {
  JNI_Iterator::Java_Iterator_remove(jni_, iterator_);
}

// Provides a way to compare the iterator with itself and with the end iterator.
// Note: all other comparison results are undefined, just like for C++ input
// iterators.
bool Iterable::Iterator::operator==(const Iterable::Iterator& other) {
  // Two different active iterators should never be compared.
  RTC_DCHECK(this == &other || AtEnd() || other.AtEnd());
  return AtEnd() == other.AtEnd();
}

ScopedJavaLocalRef<jobject>& Iterable::Iterator::operator*() {
  RTC_CHECK(!AtEnd());
  return value_;
}

bool Iterable::Iterator::AtEnd() const {
  RTC_DCHECK_RUN_ON(&thread_checker_);
  return jni_ == nullptr || IsNull(jni_, iterator_);
}

bool IsNull(JNIEnv* jni, const jni_zero::JavaRef<jobject>& obj) {
  return jni->IsSameObject(obj.obj(), nullptr);
}

std::string GetJavaEnumName(JNIEnv* jni,
                            const jni_zero::JavaRef<jobject>& j_enum) {
  return JavaToStdString(jni, JNI_Enum::Java_Enum_name(jni, j_enum));
}

Iterable GetJavaMapEntrySet(JNIEnv* jni,
                            const jni_zero::JavaRef<jobject>& j_map) {
  return Iterable(jni, JNI_Map::Java_Map_entrySet(jni, j_map));
}

ScopedJavaLocalRef<jobject> GetJavaMapEntryKey(
    JNIEnv* jni,
    const jni_zero::JavaRef<jobject>& j_entry) {
  return jni::Java_JniHelper_getKey(jni, j_entry);
}

ScopedJavaLocalRef<jobject> GetJavaMapEntryValue(
    JNIEnv* jni,
    const jni_zero::JavaRef<jobject>& j_entry) {
  return jni::Java_JniHelper_getValue(jni, j_entry);
}

int64_t JavaToNativeLong(JNIEnv* env,
                         const jni_zero::JavaRef<jobject>& j_long) {
  return JNI_Long::Java_Long_longValue(env, j_long);
}

std::optional<bool> JavaToNativeOptionalBool(
    JNIEnv* jni,
    const jni_zero::JavaRef<jobject>& boolean) {
  if (IsNull(jni, boolean))
    return std::nullopt;
  return JNI_Boolean::Java_Boolean_booleanValue(jni, boolean);
}

std::optional<double> JavaToNativeOptionalDouble(
    JNIEnv* jni,
    const jni_zero::JavaRef<jobject>& j_double) {
  if (IsNull(jni, j_double))
    return std::nullopt;
  return JNI_Double::Java_Double_doubleValue(jni, j_double);
}

std::optional<int32_t> JavaToNativeOptionalInt(
    JNIEnv* jni,
    const jni_zero::JavaRef<jobject>& integer) {
  if (IsNull(jni, integer))
    return std::nullopt;
  return JNI_Integer::Java_Integer_intValue(jni, integer);
}

// Given a jstring, reinterprets it to a new native string.
std::string JavaToNativeString(JNIEnv* jni,
                               const jni_zero::JavaRef<jstring>& j_string) {
  const jni_zero::ScopedJavaLocalRef<jbyteArray> j_byte_array =
      jni::Java_JniHelper_getStringBytes(jni, j_string);

  const size_t len = jni->GetArrayLength(j_byte_array.obj());
  CHECK_EXCEPTION(jni) << "error during GetArrayLength";
  std::string str(len, '\0');
  jni->GetByteArrayRegion(j_byte_array.obj(), 0, len,
                          reinterpret_cast<jbyte*>(&str[0]));
  CHECK_EXCEPTION(jni) << "error during GetByteArrayRegion";
  return str;
}

std::map<std::string, std::string> JavaToNativeStringMap(
    JNIEnv* jni,
    const jni_zero::JavaRef<jobject>& j_map) {
  return JavaToNativeMap<std::string, std::string>(
      jni, j_map,
      [](JNIEnv* env, jni_zero::JavaRef<jobject> const& key,
         jni_zero::JavaRef<jobject> const& value) {
        return std::make_pair(
            JavaToNativeString(env, static_java_ref_cast<jstring>(env, key)),
            JavaToNativeString(env, static_java_ref_cast<jstring>(env, value)));
      });
}

ScopedJavaLocalRef<jobject> NativeToJavaBoolean(JNIEnv* env, bool b) {
  return JNI_Boolean::Java_Boolean_Constructor__boolean(env, b);
}

ScopedJavaLocalRef<jobject> NativeToJavaDouble(JNIEnv* env, double d) {
  return JNI_Double::Java_Double_Constructor__double(env, d);
}

ScopedJavaLocalRef<jobject> NativeToJavaInteger(JNIEnv* jni, int32_t i) {
  return JNI_Integer::Java_Integer_Constructor__int(jni, i);
}

ScopedJavaLocalRef<jobject> NativeToJavaLong(JNIEnv* env, int64_t u) {
  return JNI_Long::Java_Long_Constructor__long(env, u);
}

ScopedJavaLocalRef<jstring> NativeToJavaString(JNIEnv* env, const char* str) {
  jstring j_str = env->NewStringUTF(str);
  CHECK_EXCEPTION(env) << "error during NewStringUTF";
  return jni_zero::ScopedJavaLocalRef<jstring>(env, j_str);
}

ScopedJavaLocalRef<jstring> NativeToJavaString(JNIEnv* jni,
                                               const std::string& str) {
  return NativeToJavaString(jni, str.c_str());
}

ScopedJavaLocalRef<jobject> NativeToJavaDouble(
    JNIEnv* jni,
    const std::optional<double>& optional_double) {
  return optional_double ? NativeToJavaDouble(jni, *optional_double) : nullptr;
}

ScopedJavaLocalRef<jobject> NativeToJavaInteger(
    JNIEnv* jni,
    const std::optional<int32_t>& optional_int) {
  return optional_int ? NativeToJavaInteger(jni, *optional_int) : nullptr;
}

ScopedJavaLocalRef<jstring> NativeToJavaString(
    JNIEnv* jni,
    const std::optional<std::string>& str) {
  return str ? NativeToJavaString(jni, *str) : nullptr;
}

ScopedJavaLocalRef<jbyteArray> NativeToJavaByteArray(
    JNIEnv* env,
    ArrayView<int8_t> container) {
  jni_zero::ScopedJavaLocalRef<jbyteArray> jarray(
      env, env->NewByteArray(container.size()));
  int8_t* array_ptr =
      env->GetByteArrayElements(jarray.obj(), /*isCopy=*/nullptr);
  memcpy(array_ptr, container.data(), container.size() * sizeof(int8_t));
  env->ReleaseByteArrayElements(jarray.obj(), array_ptr, /*mode=*/0);
  return jarray;
}

ScopedJavaLocalRef<jintArray> NativeToJavaIntArray(
    JNIEnv* env,
    ArrayView<int32_t> container) {
  jni_zero::ScopedJavaLocalRef<jintArray> jarray(
      env, env->NewIntArray(container.size()));
  int32_t* array_ptr =
      env->GetIntArrayElements(jarray.obj(), /*isCopy=*/nullptr);
  memcpy(array_ptr, container.data(), container.size() * sizeof(int32_t));
  env->ReleaseIntArrayElements(jarray.obj(), array_ptr, /*mode=*/0);
  return jarray;
}

std::vector<int8_t> JavaToNativeByteArray(
    JNIEnv* env,
    const jni_zero::JavaRef<jbyteArray>& jarray) {
  int8_t* array_ptr =
      env->GetByteArrayElements(jarray.obj(), /*isCopy=*/nullptr);
  size_t array_length = env->GetArrayLength(jarray.obj());
  std::vector<int8_t> container(array_ptr, array_ptr + array_length);
  env->ReleaseByteArrayElements(jarray.obj(), array_ptr, /*mode=*/JNI_ABORT);
  return container;
}

std::vector<int32_t> JavaToNativeIntArray(
    JNIEnv* env,
    const jni_zero::JavaRef<jintArray>& jarray) {
  int32_t* array_ptr =
      env->GetIntArrayElements(jarray.obj(), /*isCopy=*/nullptr);
  size_t array_length = env->GetArrayLength(jarray.obj());
  std::vector<int32_t> container(array_ptr, array_ptr + array_length);
  env->ReleaseIntArrayElements(jarray.obj(), array_ptr, /*mode=*/JNI_ABORT);
  return container;
}

std::vector<float> JavaToNativeFloatArray(
    JNIEnv* env,
    const jni_zero::JavaRef<jfloatArray>& jarray) {
  // jfloat is a "machine-dependent native type" which represents a 32-bit
  // float. C++ makes no guarantees about the size of floating point types, and
  // some exotic architectures don't even have 32-bit floats (or even binary
  // floats), but on all architectures we care about this is a float.
  static_assert(std::is_same<float, jfloat>::value, "jfloat must be float");
  float* array_ptr =
      env->GetFloatArrayElements(jarray.obj(), /*isCopy=*/nullptr);
  size_t array_length = env->GetArrayLength(jarray.obj());
  std::vector<float> container(array_ptr, array_ptr + array_length);
  env->ReleaseFloatArrayElements(jarray.obj(), array_ptr, /*mode=*/JNI_ABORT);
  return container;
}

ScopedJavaLocalRef<jobjectArray> NativeToJavaBooleanArray(
    JNIEnv* env,
    const std::vector<bool>& container) {
  return NativeToJavaObjectArray(env, container, java_lang_Boolean_clazz(env),
                                 &NativeToJavaBoolean);
}

ScopedJavaLocalRef<jobjectArray> NativeToJavaDoubleArray(
    JNIEnv* env,
    const std::vector<double>& container) {
  jni_zero::ScopedJavaLocalRef<jobject> (*convert_function)(JNIEnv*, double) =
      &NativeToJavaDouble;
  return NativeToJavaObjectArray(env, container, java_lang_Double_clazz(env),
                                 convert_function);
}

ScopedJavaLocalRef<jobjectArray> NativeToJavaIntegerArray(
    JNIEnv* env,
    const std::vector<int32_t>& container) {
  jni_zero::ScopedJavaLocalRef<jobject> (*convert_function)(JNIEnv*, int32_t) =
      &NativeToJavaInteger;
  return NativeToJavaObjectArray(env, container, java_lang_Integer_clazz(env),
                                 convert_function);
}

ScopedJavaLocalRef<jobjectArray> NativeToJavaLongArray(
    JNIEnv* env,
    const std::vector<int64_t>& container) {
  return NativeToJavaObjectArray(env, container, java_lang_Long_clazz(env),
                                 &NativeToJavaLong);
}

ScopedJavaLocalRef<jobjectArray> NativeToJavaStringArray(
    JNIEnv* env,
    const std::vector<std::string>& container) {
  jni_zero::ScopedJavaLocalRef<jstring> (*convert_function)(
      JNIEnv*, const std::string&) = &NativeToJavaString;
  return NativeToJavaObjectArray(
      env, container,
      static_cast<jclass>(jni::Java_JniHelper_getStringClass(env).obj()),
      convert_function);
}

JavaListBuilder::JavaListBuilder(JNIEnv* env)
    : env_(env), j_list_(JNI_ArrayList::Java_ArrayList_Constructor(env)) {}

JavaListBuilder::~JavaListBuilder() = default;

void JavaListBuilder::add(const jni_zero::JavaRef<jobject>& element) {
  JNI_ArrayList::Java_ArrayList_add(env_, j_list_, element);
}

JavaMapBuilder::JavaMapBuilder(JNIEnv* env)
    : env_(env),
      j_map_(JNI_LinkedHashMap::Java_LinkedHashMap_Constructor(env)) {}

JavaMapBuilder::~JavaMapBuilder() = default;

void JavaMapBuilder::put(const jni_zero::JavaRef<jobject>& key,
                         const jni_zero::JavaRef<jobject>& value) {
  JNI_Map::Java_Map_put(env_, j_map_, key, value);
}

jlong NativeToJavaPointer(const void* ptr) {
  static_assert(sizeof(intptr_t) <= sizeof(jlong),
                "Time to rethink the use of jlongs");
  // Going through intptr_t to be obvious about the definedness of the
  // conversion from pointer to integral type.  intptr_t to jlong is a standard
  // widening by the static_assert above.
  jlong ret = reinterpret_cast<intptr_t>(ptr);
  RTC_DCHECK(reinterpret_cast<void*>(ret) == ptr);
  return ret;
}

// Given a list of jstrings, reinterprets it to a new vector of native strings.
std::vector<std::string> JavaToStdVectorStrings(
    JNIEnv* jni,
    const jni_zero::JavaRef<jobject>& list) {
  std::vector<std::string> converted_list;
  if (!list.is_null()) {
    for (const jni_zero::JavaRef<jobject>& str : Iterable(jni, list)) {
      converted_list.push_back(
          JavaToStdString(jni, jni_zero::JavaParamRef<jstring>(
                                   jni, static_cast<jstring>(str.obj()))));
    }
  }
  return converted_list;
}

}  // namespace webrtc
