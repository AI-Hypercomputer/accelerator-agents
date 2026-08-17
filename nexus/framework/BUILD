load("//devtools/python/blaze:pytype.bzl", "pytype_strict_binary", "pytype_strict_contrib_test", "pytype_strict_library")

package(
    default_applicable_licenses = ["//third_party/py/accelerator_agents:license"],
    default_visibility = ["//third_party/py/accelerator_agents:internal"],
)

licenses(["notice"])

pytype_strict_library(
    name = "utils",
    srcs = ["utils.py"],
)

pytype_strict_library(
    name = "runtime",
    srcs = ["runtime.py"],
    deps = [
        ":utils",
    ],
)

pytype_strict_binary(
    name = "runtime_bin",
    srcs = ["runtime.py"],
    main = "runtime.py",
    deps = [
        ":runtime",
    ],
)

pytype_strict_library(
    name = "compiler",
    srcs = ["compiler.py"],
    deps = [
        ":runtime",
        ":utils",
    ],
)

pytype_strict_binary(
    name = "compiler_bin",
    srcs = ["compiler.py"],
    main = "compiler.py",
    deps = [
        ":compiler",
    ],
)

pytype_strict_library(
    name = "installer",
    srcs = ["installer.py"],
    deps = [
        ":utils",
    ],
)

pytype_strict_binary(
    name = "installer_bin",
    srcs = ["installer.py"],
    main = "installer.py",
    deps = [
        ":installer",
    ],
)

pytype_strict_contrib_test(
    name = "test_utils",
    srcs = ["tests/test_utils.py"],
    main = "tests/test_utils.py",
    deps = [
        ":utils",
    ],
)

pytype_strict_contrib_test(
    name = "test_runtime",
    srcs = ["tests/test_runtime.py"],
    main = "tests/test_runtime.py",
    deps = [
        ":runtime",
        ":utils",
    ],
)

pytype_strict_contrib_test(
    name = "test_compiler",
    srcs = ["tests/test_compiler.py"],
    main = "tests/test_compiler.py",
    deps = [
        ":compiler",
        ":runtime",
        ":utils",
    ],
)

pytype_strict_contrib_test(
    name = "test_installer",
    srcs = ["tests/test_installer.py"],
    main = "tests/test_installer.py",
    deps = [
        ":compiler",
        ":installer",
        ":utils",
    ],
)
