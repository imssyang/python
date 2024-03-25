
#pragma once

#if defined(_MSC_VER)
#    define PYBIND11_COMPILER_MSVC
#    define PYBIND11_PRAGMA(...) __pragma(__VA_ARGS__)
#    define PYBIND11_WARNING_PUSH PYBIND11_PRAGMA(warning(push))
#    define PYBIND11_WARNING_POP PYBIND11_PRAGMA(warning(pop))
#elif defined(__INTEL_COMPILER)
#    define PYBIND11_COMPILER_INTEL
#    define PYBIND11_PRAGMA(...) _Pragma(#__VA_ARGS__)
#    define PYBIND11_WARNING_PUSH PYBIND11_PRAGMA(warning push)
#    define PYBIND11_WARNING_POP PYBIND11_PRAGMA(warning pop)
#elif defined(__clang__)
#    define PYBIND11_COMPILER_CLANG
#    define PYBIND11_PRAGMA(...) _Pragma(#__VA_ARGS__)
#    define PYBIND11_WARNING_PUSH PYBIND11_PRAGMA(clang diagnostic push)
#    define PYBIND11_WARNING_POP PYBIND11_PRAGMA(clang diagnostic push)
#elif defined(__GNUC__)
#    define PYBIND11_COMPILER_GCC
#    define PYBIND11_PRAGMA(...) _Pragma(#__VA_ARGS__)
#    define PYBIND11_WARNING_PUSH PYBIND11_PRAGMA(GCC diagnostic push)
#    define PYBIND11_WARNING_POP PYBIND11_PRAGMA(GCC diagnostic pop)
#endif

#ifdef PYBIND11_COMPILER_MSVC
#    define PYBIND11_WARNING_DISABLE_MSVC(name) PYBIND11_PRAGMA(warning(disable : name))
#else
#    define PYBIND11_WARNING_DISABLE_MSVC(name)
#endif

#ifdef PYBIND11_COMPILER_CLANG
#    define PYBIND11_WARNING_DISABLE_CLANG(name) PYBIND11_PRAGMA(clang diagnostic ignored name)
#else
#    define PYBIND11_WARNING_DISABLE_CLANG(name)
#endif

#ifdef PYBIND11_COMPILER_GCC
#    define PYBIND11_WARNING_DISABLE_GCC(name) PYBIND11_PRAGMA(GCC diagnostic ignored name)
#else
#    define PYBIND11_WARNING_DISABLE_GCC(name)
#endif

#ifdef PYBIND11_COMPILER_INTEL
#    define PYBIND11_WARNING_DISABLE_INTEL(name) PYBIND11_PRAGMA(warning disable name)
#else
#    define PYBIND11_WARNING_DISABLE_INTEL(name)
#endif

#define PYBIND11_NAMESPACE_BEGIN(name)                                                            \
    namespace name {                                                                              \
    PYBIND11_WARNING_PUSH

#define PYBIND11_NAMESPACE_END(name)                                                              \
    PYBIND11_WARNING_POP                                                                          \
    }

#if !defined(PYBIND11_NAMESPACE)
#    ifdef __GNUG__
#        define PYBIND11_NAMESPACE pybind11 __attribute__((visibility("hidden")))
#    else
#        define PYBIND11_NAMESPACE pybind11
#    endif
#endif

#if !defined(PYBIND11_EXPORT)
#    if defined(WIN32) || defined(_WIN32)
#        define PYBIND11_EXPORT __declspec(dllexport)
#    else
#        define PYBIND11_EXPORT __attribute__((visibility("default")))
#    endif
#endif

#if !(defined(_MSC_VER) && __cplusplus == 199711L)
#    if __cplusplus >= 201402L
#        define PYBIND11_CPP14
#        if __cplusplus >= 201703L
#            define PYBIND11_CPP17
#            if __cplusplus >= 202002L
#                define PYBIND11_CPP20
// Please update tests/pybind11_tests.cpp `cpp_std()` when adding a macro here.
#            endif
#        endif
#    endif
#elif defined(_MSC_VER) && __cplusplus == 199711L
// MSVC sets _MSVC_LANG rather than __cplusplus (supposedly until the standard is fully
// implemented). Unless you use the /Zc:__cplusplus flag on Visual Studio 2017 15.7 Preview 3
// or newer.
#    if _MSVC_LANG >= 201402L
#        define PYBIND11_CPP14
#        if _MSVC_LANG > 201402L
#            define PYBIND11_CPP17
#            if _MSVC_LANG >= 202002L
#                define PYBIND11_CPP20
#            endif
#        endif
#    endif
#endif

#if defined(PYBIND11_CPP20)
#    define PYBIND11_CONSTINIT constinit
#    define PYBIND11_DTOR_CONSTEXPR constexpr
#else
#    define PYBIND11_CONSTINIT
#    define PYBIND11_DTOR_CONSTEXPR
#endif

// Compiler version assertions
#if defined(__clang__) && !defined(__apple_build_version__)
#    if __clang_major__ < 3 || (__clang_major__ == 3 && __clang_minor__ < 3)
#        error pybind11 requires clang 3.3 or newer
#    endif
#elif defined(__GNUG__)
#    if __GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ < 8)
#        error pybind11 requires gcc 4.8 or newer
#    endif
#endif

#include <Python.h>
#include <frameobject.h>
#include <pythread.h>

#include <cstddef>
#include <cstring>
#include <exception>
#include <forward_list>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <typeindex>
#include <unordered_map>
#include <unordered_set>
#include <vector>