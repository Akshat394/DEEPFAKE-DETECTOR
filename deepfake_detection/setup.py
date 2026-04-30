from setuptools import find_packages, setup


def read_requirements(path: str = "requirements.txt"):
    requirements = []
    with open(path, encoding="utf-8") as req_file:
        for line in req_file:
            requirement = line.strip()
            if not requirement or requirement.startswith("#") or requirement.startswith("--"):
                continue
            requirements.append(requirement)
    return requirements


setup(
    name="ascii_deepfake_detection",
    version="1.0.0",
    description="ASCII-Driven Hybrid Deepfake Detection Framework",
    packages=find_packages(),
    install_requires=read_requirements(),
)
