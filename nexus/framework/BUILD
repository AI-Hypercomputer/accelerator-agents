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
    name = "coworker",
    srcs = ["coworker.py"],
    deps = [
        ":utils",
    ],
)

pytype_strict_binary(
    name = "coworker_bin",
    srcs = ["coworker.py"],
    main = "coworker.py",
    deps = [
        ":coworker",
    ],
)

pytype_strict_contrib_test(
    name = "test_coworker",
    srcs = ["tests/test_coworker.py"],
    main = "tests/test_coworker.py",
    deps = [
        ":coworker",
    ],
)
