#!/usr/bin/env python3
# Copyright (c) torchcomms-hccl contributors
#
# setup.py:基于 CMake 构建后端扩展模块

import os
import pathlib
import shlex
import subprocess
import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as build_ext_orig


class CMakeExtension(Extension):
    """自定义 Extension,不使用 setuptools 默认的构建流程"""

    def __init__(self, name):
        super().__init__(name, sources=[])


class build_ext(build_ext_orig):
    """通过 CMake 构建后端扩展模块"""

    def run(self):
        for ext in self.extensions:
            self.build_cmake(ext)
            break  # 只有一个 extension

    def build_cmake(self, ext):
        cwd = pathlib.Path().absolute()
        build_temp = pathlib.Path(self.build_temp).absolute()
        build_temp.mkdir(parents=True, exist_ok=True)
        extdir = pathlib.Path(self.get_ext_fullpath(ext.name))

        cfg = os.environ.get("CMAKE_BUILD_TYPE", "Release")

        cmake_args = [
            f"-DCMAKE_BUILD_TYPE={cfg}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir.parent.absolute()}",
            f"-DCMAKE_INSTALL_PREFIX={extdir.parent.absolute()}",
            f"-DPython3_EXECUTABLE={sys.executable}",
        ]

        build_args = ["--", "-j"]

        os.chdir(str(build_temp))
        self.spawn(["cmake", str(cwd)] + cmake_args)
        if not self.dry_run:
            self.spawn(
                ["cmake", "--build", ".", "--target", "install"] + build_args
            )
        os.chdir(str(cwd))


setup(
    ext_modules=[CMakeExtension("torchcomms_hccl._comms_hccl")],
)
