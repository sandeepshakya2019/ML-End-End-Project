from setuptools import setup, find_packages

HYPEN_E_DOT = "-e ."
def get_requirements(path):
    requirements = []
    with open(path) as file:
        requirements = file.readlines()
        requirements = [req.replace("\n", "")for req in requirements]
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
            
    return requirements

setup(
    name='endtoendmlproject',
    version='0.0.1',
    description='Ml Projects',
    author='sandeepKumar',
    author_email='sandeepshakya2015@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirement.txt')
)