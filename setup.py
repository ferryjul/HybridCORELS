from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import os
import sys
from Cython.Build import cythonize


class build_numpy(build_ext):
    def finalize_options(self):
        build_ext.finalize_options(self)
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())

def install(gmp):
    description = 'Python module for Hybrid Rule Lists/Black-Box models, based on the Python binding of the CORELS algorithm'
    
    with open('./README.md') as f:
        long_description = f.read()

    with open('HybridCORELS/VERSION') as f:
        version = f.read().strip()

    pyx_file = 'HybridCORELS/_corels.pyx'

    source_dir = 'HybridCORELS/src/corels/src/'
    sources = ['utils.cpp', 'rulelib.cpp', 'run.cpp', 'pmap.cpp', 
               'corels.cpp', 'cache.cpp']
    
    for i in range(len(sources)):
        sources[i] = source_dir + sources[i]
    
    #sources.append('corels/_corels.cpp')
    sources.append(pyx_file)
    sources.append('HybridCORELS/src/utils.cpp')

    cpp_args = ['-Wall', '-O3', '-std=c++11']
    libraries = []

    if os.name == 'posix':
        libraries.append('m')

    if gmp:
        libraries.append('gmp')
        cpp_args.append('-DGMP')

    if os.name == 'nt':
        cpp_args.append('-D_hypot=hypot')
        if sys.version_info[0] < 3:
            raise Exception("Python 3.x is required on Windows")

    extension = Extension("HybridCORELS._corels", 
                sources = sources,
                libraries = libraries,
                include_dirs = ['HybridCORELS/src/', 'HybridCORELS/src/corels/src'],
                language = "c++",
                extra_compile_args = cpp_args)

    extensions = [extension]
    extensions = cythonize(extensions)

    numpy_version = 'numpy'

    if sys.version_info[0] < 3 or sys.version_info[1] < 5:
        numpy_version = 'numpy<=1.16'

    setup(
        name = 'HybridCORELS',
        packages = ['HybridCORELS'],
        ext_modules = extensions,
        version = version,
        author = 'Elaine Angelino, Nicholas Larus-Stone, Hongyu Yang, Cythnia Rudin, Vassilios Kaxiras, Margo Seltzer',
        author_email = 'vassilioskaxiras@gmail.com',
        description = description,
        long_description = long_description,
        setup_requires = [numpy_version],
        install_requires = [numpy_version],
        python_requires = '>=2.7',
        url = 'https://github.com/fingoldin/pycorels',
        cmdclass = {'build_ext': build_numpy},
        license = "GNU General Public License v3 (GPLv3)",
        package_dir={'HybridCORELS': 'HybridCORELS'},
        package_data={'HybridCORELS': ['VERSION']},
        classifiers = [
            "Programming Language :: C++",
            "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
            "Operating System :: OS Independent"
        ]
    )

if __name__ == "__main__":
    try:
        install(True)
    except:
        install(False)
