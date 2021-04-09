import os
import re
import setuptools


# for simplicity we actually store the version in the __version__ attribute in the source
here = os.path.realpath(os.path.dirname(__file__))
with open(os.path.join(here, 'torchpm', '__init__.py')) as f:
    meta_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    if meta_match:
        version = meta_match.group(1)
    else:
        raise RuntimeError("Unable to find __version__ string.")


setuptools.setup(
    name="torchpm",
    version=version,
    author="Sungwoo Goo",
    author_email="yeoun9@gmail.com",
    description="Pharmacometrics in PyTorch.",
    url="https://github.com/yeoun9/torchpm",
    packages=setuptools.find_packages(),
    install_requires=['torch>=1.3.0', 'numpy>=1.19.5', 'torchdiffeq>=0.2.1', 'sympy>=1.7.1'],
    python_requires='~=3.6',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
