from setuptools import setup, find_packages

setup(
    name="factcheck",
    version="0.1.0",
    description="Open-source hybrid fact verification using Knowledge Graphs + Web Search",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Jonathan Hammant",
    author_email="jhammant@gmail.com",
    url="https://github.com/jhammant/factcheck",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "requests>=2.31.0",
        "SPARQLWrapper>=2.0.0",
        "rich>=13.0.0",
        "click>=8.1.0",
    ],
    entry_points={
        "console_scripts": [
            "factcheck=factcheck.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
