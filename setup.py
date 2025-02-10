from setuptools import setup, find_packages

setup(
    name="fed-ensemble",
    version="0.1",
    packages = find_packages(),
    install_requires = [
        "flwr",
        "torch",
        "numpy",
        "ray"
    ]
)