import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.install_lib import install_lib

class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        """
        Initialize the class with a source directory path and an empty list of 
        included files for this target file object
        
        Args:
            name (str): The name of the generated CMake File Object to be written out in the cmakelists file 
            sourcedir (Optional[str], optional): Path to the root source folder relative to the current working 
                directory or absolute file system paths are accepted as well. Defaults to "".
        """
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        """
        根据指定的CMake扩展进行构建。
        
        Args:
          ext (CMakeExtension): 指定的CMake扩展对象。
        
        Returns:
          None: 返回值为空，表示构建成功或失败。
        
        """
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)  # type: ignore[no-untyped-call]
        print("Path.cwd() = ", Path.cwd())
        print("ext_fullpath = ", ext_fullpath)
        extdir = ext_fullpath.parent.resolve()
        print("extdir=",extdir)

        cfg = "Release"
        bin_full_path = Path.cwd() / self.get_ext_fullpath(ext.name).split("/")[0] / "output"
        print(bin_full_path)
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
            f"-DCMAKE_BINARY_DIR={bin_full_path}",
            f"-DVERSION_INFO={self.distribution.get_version()}"  # commented out, we want this set in the CMake file
        ]
        build_args = []
        ##export MKLROOT= mkl install path
        if "MKLROOT" in os.environ:
            MKLROOT = os.environ["MKLROOT"]
            print(MKLROOT)
            cmake_args += [f"-DMKLROOT={MKLROOT}"]
        ##export CMAKE_ARGS="-DA=A_VAL -DB=B_VAL"指定编译参数
        ## export  CMAKE_ARGS="-DMKLROOT=/opt/intel/oneapi/mkl/latest/ -DBLA_VENDOR=Intel10_64lp_seq"
        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]
        ##CMAKE_CACHEFILE_DIR
        build_temp =  Path(self.build_temp) / ext.name
        print(Path(self.build_temp))
        print(ext.name)
        if not build_temp.exists():
            build_temp.mkdir(parents=True)
        
        print(ext.sourcedir,cmake_args, build_temp )
        subprocess.run(
            ["cmake", "-DUSE_PYTHON=True", ext.sourcedir] + cmake_args, cwd=build_temp, check=True
        )

        subprocess.run(
            ["cmake", "--build", "."] + build_args, cwd=build_temp, check=True
        )

        subprocess.run(
            ["cmake", "--install", "."] + build_args, cwd=build_temp, check=True
        )


class InstallCMakeLibs(install_lib):
    def run(self):
        """
        重写父类方法，执行移动库文件操作。
        
        Args:
            无参数。
        
        Returns:
            NoneType：返回值为空类型。
        """
        self.announce("Moving library files", level=5)
        super().run()

setup(
    name='puck',
    version='0.0.1',
    description='A library for efficient similarity search and clustering of dense vectors ',
    url='https://github.com/baidu/puck',
    author='Huang,Ben Yin,Jie',
    author_email='huangben@baidu.com,yinjie06@baidu.com',
    license='Apache License 2.0',
    ext_modules=[CMakeExtension("puck._puck", ".")],
    cmdclass={
        'build_ext': CMakeBuild,
        'install_lib': InstallCMakeLibs
    },
    zip_safe=False,
    py_modules=['py_puck_api'],
    python_requires='>=3.6',
)


