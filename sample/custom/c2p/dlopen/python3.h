#pragma once

#include "common.h"

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)

class scoped_interpreter {
public:
    explicit scoped_interpreter(bool init_signal_handlers = true,
                                int argc = 0,
                                const char *const *argv = nullptr,
                                bool add_program_dir_to_path = true) {
        if (Py_IsInitialized() != 0) {
            //pybind11_fail("The interpreter is already running");
        }

        PyConfig config;
        PyConfig_InitPythonConfig(&config);
        config.parse_argv = 0;
        config.install_signal_handlers = init_signal_handlers ? 1 : 0;

        PyStatus status = PyConfig_SetBytesArgv(&config, argc, const_cast<char *const *>(argv));
        if (PyStatus_Exception(status) != 0) {
            // A failure here indicates a character-encoding failure or the python
            // interpreter out of memory. Give up.
            PyConfig_Clear(&config);
            throw std::runtime_error(PyStatus_IsError(status) != 0 ? status.err_msg
                                                                : "Failed to prepare CPython");
        }
        status = Py_InitializeFromConfig(&config);
        if (PyStatus_Exception(status) != 0) {
            PyConfig_Clear(&config);
            throw std::runtime_error(PyStatus_IsError(status) != 0 ? status.err_msg
                                                                : "Failed to init CPython");
        }
        if (add_program_dir_to_path) {
            PyRun_SimpleString("import sys, os.path; "
                            "sys.path.insert(0, "
                            "os.path.abspath(os.path.dirname(sys.argv[0])) "
                            "if sys.argv and os.path.exists(sys.argv[0]) else '')");
        }
        PyConfig_Clear(&config);
    }

    scoped_interpreter(const scoped_interpreter &) = delete;
    scoped_interpreter(scoped_interpreter &&other) noexcept { other.is_valid = false; }
    scoped_interpreter &operator=(const scoped_interpreter &) = delete;
    scoped_interpreter &operator=(scoped_interpreter &&) = delete;

    ~scoped_interpreter() {
        if (is_valid) {
            Py_Finalize();
        }
    }

private:
    bool is_valid = true;
};







/// Wrapper for Python extension modules
class module_ : public object {
public:
    PYBIND11_OBJECT_DEFAULT(module_, object, PyModule_Check)

    /// Create a new top-level Python module with the given name and docstring
    PYBIND11_DEPRECATED("Use PYBIND11_MODULE or module_::create_extension_module instead")
    explicit module_(const char *name, const char *doc = nullptr) {
        *this = create_extension_module(name, doc, new PyModuleDef());
    }

    /** \rst
        Create Python binding for a new function within the module scope. ``Func``
        can be a plain C++ function, a function pointer, or a lambda function. For
        details on the ``Extra&& ... extra`` argument, see section :ref:`extras`.
    \endrst */
    template <typename Func, typename... Extra>
    module_ &def(const char *name_, Func &&f, const Extra &...extra) {
        cpp_function func(std::forward<Func>(f),
                          name(name_),
                          scope(*this),
                          sibling(getattr(*this, name_, none())),
                          extra...);
        // NB: allow overwriting here because cpp_function sets up a chain with the intention of
        // overwriting (and has already checked internally that it isn't overwriting
        // non-functions).
        add_object(name_, func, true /* overwrite */);
        return *this;
    }

    /** \rst
        Create and return a new Python submodule with the given name and docstring.
        This also works recursively, i.e.

        .. code-block:: cpp

            py::module_ m("example", "pybind11 example plugin");
            py::module_ m2 = m.def_submodule("sub", "A submodule of 'example'");
            py::module_ m3 = m2.def_submodule("subsub", "A submodule of 'example.sub'");
    \endrst */
    module_ def_submodule(const char *name, const char *doc = nullptr) {
        const char *this_name = PyModule_GetName(m_ptr);
        if (this_name == nullptr) {
            throw error_already_set();
        }
        std::string full_name = std::string(this_name) + '.' + name;
        handle submodule = PyImport_AddModule(full_name.c_str());
        if (!submodule) {
            throw error_already_set();
        }
        auto result = reinterpret_borrow<module_>(submodule);
        if (doc && options::show_user_defined_docstrings()) {
            result.attr("__doc__") = pybind11::str(doc);
        }
        attr(name) = result;
        return result;
    }

    /// Import and return a module or throws `error_already_set`.
    static module_ import(const char *name) {
        PyObject *obj = PyImport_ImportModule(name);
        if (!obj) {
            throw error_already_set();
        }
        return reinterpret_steal<module_>(obj);
    }

    /// Reload the module or throws `error_already_set`.
    void reload() {
        PyObject *obj = PyImport_ReloadModule(ptr());
        if (!obj) {
            throw error_already_set();
        }
        *this = reinterpret_steal<module_>(obj);
    }

    /** \rst
        Adds an object to the module using the given name.  Throws if an object with the given name
        already exists.

        ``overwrite`` should almost always be false: attempting to overwrite objects that pybind11
        has established will, in most cases, break things.
    \endrst */
    PYBIND11_NOINLINE void add_object(const char *name, handle obj, bool overwrite = false) {
        if (!overwrite && hasattr(*this, name)) {
            pybind11_fail(
                "Error during initialization: multiple incompatible definitions with name \""
                + std::string(name) + "\"");
        }

        PyModule_AddObject(ptr(), name, obj.inc_ref().ptr() /* steals a reference */);
    }

    using module_def = PyModuleDef; // TODO: Can this be removed (it was needed only for Python 2)?

    /** \rst
        Create a new top-level module that can be used as the main module of a C extension.

        ``def`` should point to a statically allocated module_def.
    \endrst */
    static module_ create_extension_module(const char *name, const char *doc, module_def *def) {
        // module_def is PyModuleDef
        // Placement new (not an allocation).
        def = new (def)
            PyModuleDef{/* m_base */ PyModuleDef_HEAD_INIT,
                        /* m_name */ name,
                        /* m_doc */ options::show_user_defined_docstrings() ? doc : nullptr,
                        /* m_size */ -1,
                        /* m_methods */ nullptr,
                        /* m_slots */ nullptr,
                        /* m_traverse */ nullptr,
                        /* m_clear */ nullptr,
                        /* m_free */ nullptr};
        auto *m = PyModule_Create(def);
        if (m == nullptr) {
            if (PyErr_Occurred()) {
                throw error_already_set();
            }
            pybind11_fail("Internal error in module_::create_extension_module()");
        }
        // TODO: Should be reinterpret_steal for Python 3, but Python also steals it again when
        //       returned from PyInit_...
        //       For Python 2, reinterpret_borrow was correct.
        return reinterpret_borrow<module_>(m);
    }
};

PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)

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