from setuptools import setup, find_packages

setup(
    name="cryoarc",
    version="0.1.0",
    description="Cryo-EM Atomic-Resolution Conformations",
    author="Rémi Vuillemot",
    packages=find_packages(),
    python_requires=">=3.10",


    # Optional but strongly recommended
    entry_points={
        "console_scripts": [
            "cryoarc=cryoarc.cli:main",
        ],
    },

    include_package_data=True,
    zip_safe=False,
)
