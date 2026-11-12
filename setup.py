from setuptools import find_packages, setup

setup(
    name="astra-irstd-core",
    version="0.1.2",
    description="ASTRA core architecture and unified target-level evaluation",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(include=["astra", "astra.*", "evaluation", "evaluation.*"]),
    python_requires=">=3.8",
    install_requires=["torch>=2.4.1", "numpy>=1.23.5", "scipy>=1.10.1", "PyYAML>=6.0", "Pillow>=10.0"],
    extras_require={"torchvision": ["torchvision>=0.19.1"]},
    entry_points={"console_scripts": ["astra-core=astra.cli:main"]},
    license="AGPL-3.0; third-party Apache-2.0 and MIT notices retained",
)
