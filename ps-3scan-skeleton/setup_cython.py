from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import skeleton.io_tools

path = skeleton.io_tools.module_relative_path('skeleton/thinning.pyx')
ext_modules = [
    Extension('skeleton.thinning',
              [path],
              )]
setup(
    name='skeleton.thinning',
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules
)
