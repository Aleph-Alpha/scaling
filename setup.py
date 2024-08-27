from setuptools import setup, find_packages
from pathlib import Path

reqs_dir = Path("./requirements")


def get_whitelisted_packages(sub_package: str):
    whitelisted_packages = []
    for package_name in find_packages("src"):
        if package_name.startswith(sub_package):
            whitelisted_packages.append(package_name)
    return whitelisted_packages

whitelisted_packages = get_whitelisted_packages(sub_package="scaling")

# Gather scaling requirements.
requirements_base = (reqs_dir / "base.txt").read_text().splitlines()
requirements_test = (reqs_dir / "test.txt").read_text().splitlines()

requirements_optimization = (reqs_dir / "gpu_optimization.txt").read_text().splitlines()
requirements_determined = (reqs_dir / "determined.txt").read_text().splitlines()

setup(
    name="aleph-alpha-scaling",
    url="https://github.com/Aleph-Alpha",
    author="Aleph Alpha",
    author_email="requests@aleph-alpha-ip.ai",
    install_requires=requirements_base,
    tests_require=requirements_test,
    extras_require={
        "test": requirements_test,
        "gpu_optimization": requirements_optimization,
        "determined": requirements_determined,
    },
    package_dir={"": "src"},
    packages=whitelisted_packages,
    version="0.1.0",
    license="Open Aleph License",
    description="Non-distributed transformer implementation aimed at loading neox checkpoints for inference.",
    # long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    entry_points="",
    package_data={
        # If any package contains *.json or *.typed
        "": ["*.json", "*.typed", "warnings.txt"],
    },
    include_package_data=True,
)
