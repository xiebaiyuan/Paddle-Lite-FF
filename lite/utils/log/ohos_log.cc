#if defined(__OHOS__) && defined(LITE_WITH_LOG)
#include "hilog/log.h"
#include "lite/utils/log/ohos_log.h"
namespace ohos{
//  OHOS_LOG_I("device_name %s", device_name.c_str());
#undef LOG_INFO
#undef LOG_WARN

#undef LOG_DOMAIN
#undef LOG_TAG
#define LOG_DOMAIN 0x3200 // 全局domain宏，标识业务领域
#define LOG_TAG "Paddle-Lite-OHOS" // 全局tag宏，标识模块日志tag

#define OHOS_LOG_I(format, ...) \
OH_LOG_INFO(LogType::LOG_APP, "【native】Info: " format, ##__VA_ARGS__);

#define OHOS_LOG_W(format, ...) \
OH_LOG_WARN(LogType::LOG_APP, "【native】Info: " format, ##__VA_ARGS__);

void log(std::string& info){
  OHOS_LOG_I("device_name %{public}s", info.c_str());
}
void log(const std::string& info){
  OHOS_LOG_I("device_name %{public}s", info.c_str());
}

#undef LOG_DOMAIN
#undef LOG_TAG
}
#endif

