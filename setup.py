from setuptools import find_packages, setup

LONG_DESCRIPTION = open('README.md').read()

setup(
    name='pyneat',
    version='0.2.0',

    packages=find_packages(),
    
    description='A NEAT library in Python',
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',

    author='reddragonnm',
    license='MIT',
    url='https://github.com/reddragonnm/pyneat'
)