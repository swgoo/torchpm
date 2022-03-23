import setuptools

setuptools.setup(
    name="torchpm",
    version='0.0.3',
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
