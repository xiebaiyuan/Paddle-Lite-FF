#if defined(__OHOS__) && defined(LITE_WITH_LOG)

#pragma once
#include <string>

namespace ohos{
void log(std::string& info);
void log(const std::string& info);
}
#endif