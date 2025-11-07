#include "base/base.h"

namespace base {

Status::Status(StatusCode code, std::string err_message)
    : code_(code), message_(std::move(err_message)) {}

Status& Status::operator=(StatusCode code) {
  code_ = code;
  return *this;
}

bool Status::operator==(StatusCode code) const {
  return code_ == code;
}

bool Status::operator!=(StatusCode code) const {
  return code_ != code;
}

Status::operator StatusCode() const {
  return code_;
}

Status::operator bool() const {
  return code_ == StatusCode::kSuccess;
}

StatusCode Status::code() const {
  return code_;
}

const std::string& Status::message() const {
  return message_;
}

void Status::set_message(const std::string& err_msg) {
  message_ = err_msg;
}

std::ostream& operator<<(std::ostream& os, const Status& x) {
  os << x.message();
  return os;
}

}  // namespace base
