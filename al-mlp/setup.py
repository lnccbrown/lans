from distutils.core import setup
from Cython.Build import cythonize
import numpy

# setup(
#         # ext_modules = cythonize("ddm_data_simulation1.pyx", annotate=True),
#         # ext_modules = cythonize("make_data_wfpt1.pyx", annotate=True),
#         # ext_modules = cythonize("cddm_data_simulation.pyx", 
#         #                         annotate = True, 
#         #                         compiler_directives = {"language_level": "3"}),
#         # ext_modules = cythonize("samplers/cslice_sampler.pyx",
#         #                       annotate = True, 
#         #                       compiler_directives = {'language_level': "3"}),
# #         ext_modules = cythonize("clba.pyx", annotate = True, compiler_directives = {'language_level': "3"}),
#         ext_modules = cythonize("cdwiener.pyx", annotate = True, compiler_directives = {'language_level': "3"}),
#         # ext_modules = cythonize("wfpt.pyx", language="c++"),
#         #ext_modules = cythonize("ckeras_to_numpy.pyx", annotate = True, compiler_directives = {'language_level': "3"}),
#         #ext_modules = cythonize("keras_to_numpy_class.pyx", annotate = True, compiler_directive = {'language_level': "3"}),
#         include_dirs = [numpy.get_include()]
#     )

# setup(
#         ext_modules = cythonize("ckeras_to_numpy.pyx", annotate = True, compiler_directives = {'language_level': "3"}),
#         include_dirs = [numpy.get_include()]
#      )

setup(
        ext_modules = cythonize("cddm_data_simulation.pyx", 
                                annotate = True, 
                                compiler_directives = {"language_level": "3"}),
        include_dirs = [numpy.get_include()]
    )

# setup(
#         ext_modules = cythonize("clba.pyx", annotate = True, compiler_directives = {'language_level': "3"}),
#         include_dirs = [numpy.get_include()]
# )