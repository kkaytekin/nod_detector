import os

from setuptools import find_packages, setup

# Get the directory where setup.py is located
here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the README file
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# Get the version from the package
about = {}
with open(os.path.join(here, "src", "nod_detector", "__version__.py"), encoding="utf-8") as f:
    exec(f.read(), about)


setup(
    name="nod-detector",
    version=about["__version__"],
    author="Kagan Kucukaytekin",
    author_email="kkaytekin@gmail.com",
    description="A system for detecting nodding behavior in videos",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kkaytekin/nod_detector",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.5",
        "opencv-python>=4.5.1.48",
        "mediapipe>=0.8.9.1",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.5",
            "black>=21.12b0",
            "isort>=5.10.1",
            "flake8>=4.0.1",
            "mypy>=0.931",
            "pytest-cov>=4.0.0",
            "pytest-mock>=3.6.1",
            "pytest-xdist>=2.3.0",
            "pytest-timeout>=2.0.0",
            "pytest-benchmark>=3.4.1",
            "pytest-sugar>=0.9.4",
            "pre-commit>=3.3.3",
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "sphinxcontrib-napoleon>=0.7",
        ],
    },
    entry_points={
        "console_scripts": [
            "nod-detector=nod_detector.main:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
