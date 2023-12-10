import os
import subprocess
import sys
import numpy as np

from typing import Literal


def _dummyimport():
    import Cython


try:
    from .parasortcytq import (
        index_sort_parallel_buffered,
        index_sort_parallel,
        index_sort,
    )
except Exception as e:
    cstring = r"""# distutils: language=c++
# distutils: extra_compile_args=/std:c++20 /openmp
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: language_level=3
# cython: initializedcheck=False

from cython.parallel cimport prange
cimport cython
import numpy as np
cimport numpy as np
import cython
from libcpp.vector cimport vector
np.import_array()


ctypedef fused real:
    cython.bint
    cython.char
    cython.schar
    cython.uchar
    cython.short
    cython.ushort
    cython.int
    cython.uint
    cython.long
    cython.ulong
    cython.longlong
    cython.ulonglong
    cython.float
    cython.double
    cython.longdouble
    cython.size_t
    cython.Py_ssize_t
   

cdef extern from "parallelargsort.h":
    cdef vector[size_t] sort_indexes[T](vector &v) nogil
    cdef vector[size_t] sort_indexes_parallel[T](vector &v) nogil
    cdef vector[size_t] sort_indexes_parallel_buffered[T](vector &v) nogil



cdef void _index_sort_parallel_buffered(vector[real] &my_vector2,Py_ssize_t[:] outarray,Py_ssize_t lena ):
    cdef Py_ssize_t i
    cdef vector[size_t] tmpvec= sort_indexes_parallel_buffered(my_vector2)
    with nogil:

        for i in range(lena):
            outarray[i] = tmpvec[i]

cpdef void index_sort_parallel_buffered(real[:] a, Py_ssize_t[:] outarray):
    cdef Py_ssize_t i
    cdef Py_ssize_t lena = len(a)
    cdef vector[real] my_vector 
    my_vector.reserve(lena)
    with nogil:
        for i in range(lena):
                my_vector.push_back(a[i])
    _index_sort_parallel_buffered(my_vector,outarray,lena)




cdef void _index_sort_parallel(vector[real] &my_vector2,Py_ssize_t[:] outarray,Py_ssize_t lena ):
    cdef Py_ssize_t i
    cdef vector[size_t] tmpvec= sort_indexes_parallel(my_vector2)
    with nogil:

        for i in range(lena):
            outarray[i] = tmpvec[i]
cpdef void index_sort_parallel(real[:] a, Py_ssize_t[:] outarray):
    cdef Py_ssize_t i
    cdef Py_ssize_t lena = len(a)
    cdef vector[real] my_vector 
    my_vector.reserve(lena)
    with nogil:
        for i in range(lena):
                my_vector.push_back(a[i])
    _index_sort_parallel(my_vector,outarray,lena)



cdef void _index_sort(vector[real] &my_vector2,Py_ssize_t[:] outarray,Py_ssize_t lena ):
    cdef Py_ssize_t i
    cdef vector[size_t] tmpvec= sort_indexes(my_vector2)
    with nogil:

        for i in range(lena):
            outarray[i] = tmpvec[i]
cpdef void index_sort(real[:] a, Py_ssize_t[:] outarray):
    cdef Py_ssize_t i
    cdef Py_ssize_t lena = len(a)
    cdef vector[real] my_vector 
    my_vector.reserve(lena)
    with nogil:
        for i in range(lena):
                my_vector.push_back(a[i])
    _index_sort(my_vector,outarray,lena)


"""
    pyxfile = f"parasortcytq.pyx"
    pyxfilesetup = f"parasortcytqcompiled_setup.py"

    dirname = os.path.abspath(os.path.dirname(__file__))
    pyxfile_complete_path = os.path.join(dirname, pyxfile)
    pyxfile_setup_complete_path = os.path.join(dirname, pyxfilesetup)

    if os.path.exists(pyxfile_complete_path):
        os.remove(pyxfile_complete_path)
    if os.path.exists(pyxfile_setup_complete_path):
        os.remove(pyxfile_setup_complete_path)
    with open(pyxfile_complete_path, mode="w", encoding="utf-8") as f:
        f.write(cstring)
    numpyincludefolder = np.get_include()
    compilefile = (
        """
	from setuptools import Extension, setup
	from Cython.Build import cythonize
	ext_modules = Extension(**{'py_limited_api': False, 'name': 'parasortcytq', 'sources': ['parasortcytq.pyx'], 'include_dirs': [\'"""
        + numpyincludefolder
        + """\'], 'define_macros': [], 'undef_macros': [], 'library_dirs': [], 'libraries': [], 'runtime_library_dirs': [], 'extra_objects': [], 'extra_compile_args': [], 'extra_link_args': [], 'export_symbols': [], 'swig_opts': [], 'depends': [], 'language': None, 'optional': None})

	setup(
		name='parasortcytq',
		ext_modules=cythonize(ext_modules),
	)
			"""
    )
    with open(pyxfile_setup_complete_path, mode="w", encoding="utf-8") as f:
        f.write(
            "\n".join(
                [x.lstrip().replace(os.sep, "/") for x in compilefile.splitlines()]
            )
        )
    subprocess.run(
        [sys.executable, pyxfile_setup_complete_path, "build_ext", "--inplace"],
        cwd=dirname,
        shell=True,
        env=os.environ.copy(),
    )
    try:
        from .parasortcytq import (
            index_sort_parallel_buffered,
            index_sort_parallel,
            index_sort,
        )

    except Exception as fe:
        sys.stderr.write(f"{fe}")
        sys.stderr.flush()

def parallel_argsort(
    a,
    method: Literal["parallel_buffered", "parallel", "sort"] = "parallel_buffered",
):
    b = np.zeros(a.shape, dtype=np.int64)
    if method == "parallel_buffered":
        index_sort_parallel_buffered(a, b)
    elif method == "parallel":
        index_sort_parallel(a, b)
    elif method == "sort":
        index_sort(a, b)
    return b
