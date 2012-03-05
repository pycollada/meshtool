from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension("pdae_cython", ["pdae_cython.pyx"])]

setup(
  name = 'PDAE utilities c-module',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)
