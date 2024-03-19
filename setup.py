import setuptools

setuptools.setup(
    name="torchpm",
    version='0.1.0',
    author="Sungwoo Goo",
    author_email="swgoo@outlook.kr",
    description="Pharmacometrics in PyTorch.",
    url="https://github.com/swgoo/torchpm",
    packages=setuptools.find_packages(),
    install_requires=['torch>=1.3.0', 'numpy>=1.19.5', 'torchode>=0.2.0'],
    python_requires='>=3.8',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
