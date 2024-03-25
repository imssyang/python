#include "python3.h"
#include <dlfcn.h>

Python3::Python3(const std::string& path)
{
    handle_ = dlopen(path.data(), RTLD_NOW);
    if (!handle_) { return; }

    //allocate_ = reinterpret_cast<decltype(allocate_)>(dlsym(handle_, "AHardwareBuffer_allocate"));
}
