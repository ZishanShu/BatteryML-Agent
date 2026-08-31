# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

from setuptools import setup, find_packages


def read_requirements(path="requirements.txt"):
    """Requirement lines from `path`, without comments or blanks.

    requirements.txt carries explanatory comments and inline "PIN:" markers
    for the versions this repository cannot state without observing the
    server, so the raw file cannot be handed to install_requires unfiltered.
    """
    requirements = []
    with open(path, encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.split("#", 1)[0].strip()
            if line:
                requirements.append(line)
    return requirements


required_packages = read_requirements()

setup(
    name="BatteryML",
    version="0.0.1",
    description="An Open-Source Tool for Machine Learning on Battery Degradation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Microsoft Corporation",
    url="https://github.com/microsoft/BatteryML",
    # scripts/ is a regular package (it has an __init__.py) so that the repo's
    # own scripts.verify always wins over any installed distribution that also
    # provides a top-level "scripts" package. It is repository tooling, not
    # part of the library, so exclude it and everything under it.
    packages=find_packages(exclude=['scripts', 'scripts.*']),
    install_requires=required_packages,
    # [ASSUMED] Derived from the syntax the codebase uses, not observed on the
    # server. Confirm from ENV-01's environment_check.json and tighten if the
    # server runs something higher.
    python_requires=">=3.9",
    # Addressing the Windows usage issue
    # See https://github.com/microsoft/BatteryML/issues/21
    entry_points={
        'console_scripts': [
            'batteryml=bin.batteryml:main',
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
    ],
)
