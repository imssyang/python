#pragma once
#include <Python.h>
#include <string>

class Python3
{
public:
    static Python3& Instance(const std::string& path = "") {
        static Python3 instance(path);
        return instance;
    }

private:
    Python3(const std::string& path);
    Python3(const Python3&) = delete;
    ~Python3() = default;

private:
    void* handle_;
};