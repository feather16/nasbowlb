# distutils: language = c++ 

# https://qiita.com/en3/items/1f1a609c4d7c8f3066a7
# https://qiita.com/taroc/items/fc854340a5e498ceb07d

# CythonをPythonから呼び出せるようにコンパイルする

from distutils import sysconfig
from distutils.core import setup, Extension
from Cython.Build import cythonize
from numpy import get_include # cimport numpy を使うため

# GCCのパスを設定
sysconfig.get_config_vars()['CC'] = '/usr/bin/gcc -pthread -std=c++11'

ext = Extension(
    "cython_wl_kernel", 
    sources=[
        'cython_wl_kernel.pyx', 
        'wl_kernel_impl.cpp'
        ], 
    include_dirs=['.', get_include()], 
    extra_compile_args=["-O3"], 
    language="c++"
)

setup(name="cython_wl_kernel", ext_modules=cythonize([ext], language_level = "3"))