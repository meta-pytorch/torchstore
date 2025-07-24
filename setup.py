from setuptools import setup, find_packages

setup(
    name="torchstore",
    version="0.1.0",
    description="A storage solution for PyTorch tensors with distributed tensor support",
    author="Meta",
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=[
        "torch>=1.0.0",
        "monarch",  # Add version constraint if known
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
