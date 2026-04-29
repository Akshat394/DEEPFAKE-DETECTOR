from setuptools import find_packages, setup

setup(
    name="ascii_deepfake_detection",
    version="1.0.0",
    description="ASCII-Driven Hybrid Deepfake Detection Framework",
    packages=find_packages(),
    install_requires=[line.strip() for line in open("requirements.txt", encoding="utf-8") if line.strip()],
)
