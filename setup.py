from setuptools import setup, find_packages

setup(
    name="polypharmacy",
    version="0.0.1",
    author="David Bellamy",
    author_email="bellamyrd@gmail.com",
    description="Weight sharing for GNNs for side-effect prediction in drug-drug interaction graphs",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.12",
    ],
)
