from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy
ext_modules = [
    Extension(
        "algorithms",
        ["algorithms_C.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    )
]


setup(
  name = 'algorithms',
  cmdclass = {'build_ext': build_ext},
  
 ext_modules=cythonize(ext_modules),
        include_dirs=[numpy.get_include()]
)