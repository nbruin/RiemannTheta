import setuptools
from Cython.Build import cythonize
import numpy

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='RiemannTheta',
    version="0.0.1",
    author="Nils Bruin, Sohrab Ganjian",
    author_email="nbruin@sfu.ca",
    license="GPL2+",
    description="Evaluate Riemann Theta function numerically in Sagemath",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    ext_modules=cythonize("riemann_theta/riemann_theta.pyx"),
    include_dirs=[numpy.get_include()],
    zip_safe=False,
)
