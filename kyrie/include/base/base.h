#ifndef KYRIE_INCLUDE_BASE_BASE_H_
#define KYRIE_INCLUDE_BASE_BASE_H_
#include <glog/logging.h>
#include <cstdint>
#include <string>

#define UNUSED(expr) \
  do {               \
    (void)(expr);    \
  } while (0)

namespace model {
enum class BufferType : uint8_t {
  kInputTokens = 0,
  kInputEmbeddings = 1,
  kOutputRMSNorm = 2,
  kKeyCache = 3,
  kValueCache = 4,
  kQuery = 5,
  kInputPos = 6,
  kScoreStorage = 7,
  kOutputMHA = 8,
  kAttnOutput = 9,
  kW1Output = 10,
  kW2Output = 11,
  kW3Output = 12,
  kFFNRMSNorm = 13,
  kFFNOutput = 14,
  kForwardOutput = 15,
  kForwardOutputCPU = 16,
  kSinCache = 17,
  kCosCache = 18,
};
}

namespace base {
enum class DeviceType : uint8_t {
  kDeviceUnknown = 0,
  kDeviceCPU = 1,
  kDeviceCUDA = 2,
};

enum class DataType : uint8_t {
  kDataTypeUnknown = 0,
  kDataTypeFp32 = 1,
  kDataTypeInt8 = 2,
  kDataTypeInt32 = 3,
};

enum class ModelType : uint8_t {
  kModelTypeUnknown = 0,
  kModelTypeQwen2 = 1,
};

enum class TokenizerType : uint8_t {
  kTokenizerUnknown = 0,
  kTokenizerBpe = 1,
};

enum class StatusCode : uint8_t {
  kSuccess = 0,
  kFunctionUnImplement = 1,
  kPathNotValid = 2,
  kModelParseError = 3,
  kInternalError = 4,
  kKeyValueExists = 5,
  kInvalidArgument = 6,
};

inline size_t DataTypeSize(DataType data_type) {
  switch (data_type) {
    case DataType::kDataTypeFp32:
      return sizeof(float);
    case DataType::kDataTypeInt8:
      return sizeof(int8_t);
    case DataType::kDataTypeInt32:
      return sizeof(int32_t);
    default:
      return 0;
  }
}

class NoCopyable {
 protected:
  NoCopyable() = default;
  ~NoCopyable() = default;
  NoCopyable(const NoCopyable&) = delete;
  NoCopyable& operator=(const NoCopyable&) = delete;
};

class Status {
 public:
  Status(StatusCode code = StatusCode::kSuccess, std::string err_message = "");

  Status(const Status& other) = default;
  Status& operator=(const Status& other) = default;

  Status& operator=(StatusCode code);
  bool operator==(StatusCode code) const;
  bool operator!=(StatusCode code) const;

  explicit operator bool() const;
  operator StatusCode() const;

  StatusCode code() const;
  const std::string& message() const;
  void set_message(const std::string& err_msg);

 private:
  StatusCode code_ = StatusCode::kSuccess;
  std::string message_;
};

namespace error {
#define STATUS_CHECK(call)                                                                 \
  do {                                                                                     \
    const base::Status& status = call;                                                     \
    if (!status) {                                                                         \
      LOG(FATAL) << "Inference error in " << __FILE__ << ":" << __LINE__                  \
                 << "\nError code: " << static_cast<int>(status.code())                    \
                 << "\nError message: " << status.message();                               \
    }                                                                                      \
  } while (0)

inline Status Success(const std::string& msg = "") {
  return Status(StatusCode::kSuccess, msg);
}

inline Status FunctionNotImplement(const std::string& msg = "") {
  return Status(StatusCode::kFunctionUnImplement, msg);
}

inline Status PathNotValid(const std::string& msg = "") {
  return Status(StatusCode::kPathNotValid, msg);
}

inline Status ModelParseError(const std::string& msg = "") {
  return Status(StatusCode::kModelParseError, msg);
}

inline Status InternalError(const std::string& msg = "") {
  return Status(StatusCode::kInternalError, msg);
}

inline Status KeyValueExists(const std::string& msg = "") {
  return Status(StatusCode::kKeyValueExists, msg);
}

inline Status InvalidArgument(const std::string& msg = "") {
  return Status(StatusCode::kInvalidArgument, msg);
}

}  // namespace error

std::ostream& operator<<(std::ostream& os, const Status& x);

}  // namespace base
#endif  // KYRIE_INCLUDE_BASE_BASE_H_
