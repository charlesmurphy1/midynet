import os
import sys
import subprocess
import setuptools
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

ROOT = os.path.dirname(os.path.abspath(__file__))
# for x in os.walk(ROOT):
#     print(x)


class get_pybind_include(object):
    """Helper class to determine the pybind11 include path

    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked."""

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11

        return pybind11.get_include(self.user)


def find_compiled_basegraph(build_path):
    if not os.path.isdir(build_path) or os.listdir(build_path) == []:
        raise RuntimeError("Submodule BaseGraph was not compiled.")
    lib_path = None
    for extension in [".a"]:
        if os.path.isfile(
            os.path.join(build_path, "libBaseGraph" + extension)
        ):
            lib_path = os.path.join(build_path, "libBaseGraph" + extension)
            break

    if lib_path is None:
        raise RuntimeError(
            f'Could not find libBaseGraph in "{build_path}".'
            + "Verify that the library is compiled."
        )
    return lib_path


def find_compiled_SamplableSet(build_path):
    if not os.path.isdir(build_path) or os.listdir(build_path) == []:
        raise RuntimeError("Submodule SamplableSet was not compiled.")
    lib_path = os.path.join(build_path, "libsamplableset.a")
    return lib_path


def find_files_recursively(path, ext=[]):
    if isinstance(ext, str):
        ext = [ext]
    elif not isinstance(ext, list):
        raise TypeError(
            f"type `{type(ext)}` for extension is incorrect,"
            + "expect `str` or `list`."
        )
    file_list = []

    for e in ext:
        for root, subdirs, files in os.walk(path):
            for f in files:
                if f.split(".")[-1] == e:
                    file_list.append(os.path.join(root, f))
    return file_list


ext_modules = [
    Extension(
        "basegraph",
        include_dirs=[
            get_pybind_include(),
            get_pybind_include(user=True),
            ROOT + "/_midynet/base_graph/include",
        ],
        sources=[
            ROOT + "/_midynet/base_graph/pybind_wrapper/pybind_main.cpp",
            ROOT + "/_midynet/base_graph/src/directed_multigraph.cpp",
            ROOT + "/_midynet/base_graph/src/directedgraph.cpp",
            ROOT + "/_midynet/base_graph/src/undirectedgraph.cpp",
            ROOT + "/_midynet/base_graph/src/undirected_multigraph.cpp",
            ROOT + "/_midynet/base_graph/src/fileio.cpp",
            ROOT + "/_midynet/base_graph/src/algorithms/graphpaths.cpp",
            ROOT + "/_midynet/base_graph/src/algorithms/percolation.cpp",
            ROOT + "/_midynet/base_graph/src/algorithms/randomgraphs.cpp",
            ROOT + "/_midynet/base_graph/src/metrics/general.cpp",
            ROOT + "/_midynet/base_graph/src/metrics/directed.cpp",
            ROOT + "/_midynet/base_graph/src/metrics/undirected.cpp",
        ],
        language="c++",
    ),
    Extension(
        "_SamplableSet",
        include_dirs=[
            ROOT + "/_midynet/SamplableSet/src/",
            get_pybind_include(),
            get_pybind_include(user=True),
        ],
        sources=[
            ROOT + "/_midynet/SamplableSet/src/bind_SamplableSet.cpp",
            ROOT + "/_midynet/SamplableSet/src/HashPropensity.cpp",
            ROOT + "/_midynet/SamplableSet/src/BinaryTree.cpp",
            ROOT + "/_midynet/SamplableSet/src/SamplableSet.cpp",
        ],
        language="c++",
    ),
    Extension(
        "_midynet",
        include_dirs=[
            get_pybind_include(),
            get_pybind_include(user=True),
            "./_midynet/include",
            "./_midynet/base_graph/include",
            "./_midynet/SamplableSet/src",
        ],
        sources=[
            ROOT + "/_midynet/src/rng.cpp",
            ROOT + "/_midynet/src/exceptions.cpp",
            ROOT + "/_midynet/src/generators.cpp",
            ROOT + "/_midynet/src/utility/functions.cpp",
            ROOT + "/_midynet/src/prior/sbm/block_count.cpp",
            ROOT + "/_midynet/src/prior/sbm/vertex_count.cpp",
            ROOT + "/_midynet/src/prior/sbm/block.cpp",
            ROOT + "/_midynet/src/prior/sbm/edge_count.cpp",
            ROOT + "/_midynet/src/prior/sbm/edge_matrix.cpp",
            ROOT + "/_midynet/src/prior/sbm/degree.cpp",
            ROOT + "/_midynet/src/random_graph/random_graph.cpp",
            ROOT + "/_midynet/src/random_graph/sbm.cpp",
            ROOT + "/_midynet/src/random_graph/dcsbm.cpp",
            ROOT + "/_midynet/src/dynamics/dynamics.cpp",
            ROOT + "/_midynet/src/dynamics/binary_dynamics.cpp",
            ROOT + "/_midynet/src/dynamics/cowan.cpp",
            ROOT + "/_midynet/src/dynamics/degree.cpp",
            ROOT + "/_midynet/src/dynamics/ising-glauber.cpp",
            ROOT + "/_midynet/src/dynamics/sis.cpp",
            ROOT + "/_midynet/src/proposer/edge_proposer/vertex_sampler.cpp",
            ROOT + "/_midynet/src/proposer/edge_proposer/double_edge_swap.cpp",
            ROOT + "/_midynet/src/proposer/edge_proposer/hinge_flip.cpp",
            ROOT + "/_midynet/src/proposer/edge_proposer/single_edge.cpp",
            ROOT + "/_midynet/src/proposer/block_proposer/generic.cpp",
            ROOT + "/_midynet/src/proposer/block_proposer/uniform.cpp",
            ROOT + "/_midynet/src/proposer/block_proposer/peixoto.cpp",
            ROOT + "/_midynet/src/mcmc/mcmc.cpp",
            ROOT + "/_midynet/src/mcmc/callbacks/callback.cpp",
            ROOT + "/_midynet/src/mcmc/callbacks/verbose.cpp",
            ROOT + "/_midynet/src/mcmc/callbacks/collector.cpp",
            ROOT + "/_midynet/src/mcmc/graph_mcmc.cpp",
            ROOT + "/_midynet/src/mcmc/dynamics_mcmc.cpp",
            ROOT + "/_midynet/pybind_wrapper/pybind_main.cpp",
        ],
        language="c++",
        extra_objects=[
            find_compiled_basegraph(ROOT + "/_midynet/base_graph/build"),
            find_compiled_SamplableSet(
                ROOT + "/_midynet/SamplableSet/src/build"
            ),
        ],
    ),
]


# As of Python 3.6, C Compiler has a `has_flag` method.
# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".cpp") as f:
        f.write("int main (int argc, char **argv) { return 0; }")
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


def cpp_flag(compiler):
    """Return the -std=c++[11/14/17] compiler flag.

    The newer version is prefered over c++11 (when it is available).
    """
    flags = ["-std=c++17", "-std=c++11"]

    for flag in flags:
        if has_flag(compiler, flag):
            return flag

    raise RuntimeError(
        "Unsupported compiler -- at least C++11 support " "is needed!"
    )


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""

    c_opts = {
        "msvc": ["/EHsc"],
        "unix": [],
    }
    l_opts = {
        "msvc": [],
        "unix": [],
    }

    if sys.platform == "darwin":
        darwin_opts = ["-stdlib=libc++", "-mmacosx-version-min=10.7"]
        c_opts["unix"] += darwin_opts
        l_opts["unix"] += darwin_opts

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        link_opts = self.l_opts.get(ct, [])
        if ct == "unix":
            opts.append(
                '-DVERSION_INFO="%s"' % self.distribution.get_version()
            )
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, "-fvisibility=hidden"):
                opts.append("-fvisibility=hidden")
        elif ct == "msvc":
            opts.append(
                '/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version()
            )
        for ext in self.extensions:
            ext.extra_compile_args = opts
            ext.extra_link_args = link_opts
        build_ext.build_extensions(self)


setup(
    version=0.1,
    ext_modules=ext_modules,
    extras_require={"full": ["networkx", "netrd", "graph_tool"]},
    include_package_data=True,
    cmdclass={"build_ext": BuildExt},
)
