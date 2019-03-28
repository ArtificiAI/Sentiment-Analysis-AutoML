"""
This files originate from the "New-Empty-Python-Project-Base" template:
    https://github.com/guillaume-chevalier/New-Empty-Python-Project-Base 
Created by Guillaume Chevalier:
    https://github.com/guillaume-chevalier 
License: CC0-1.0 (Public Domain)
"""

from setuptools import setup

with open('README.md') as _f:
    _README_MD = _f.read()

_VERSION = '0.1'

setup(
    name='project', # TODO: rename. 
    version=_VERSION,
    description='An empty project base.',
    long_description=_README_MD,
    classifiers=[
        # TODO: typing.
        "Typing :: Typed"
    ],
    url='https://github.com/..../....',  # TODO.
    download_url='https://github.com/.../.../tarball/{}'.format(_VERSION),  # TODO.
    author='Neuraxio Inc.',
    author_email='guillaume.chevalier@neuraxio.com',
    packages=['project'],  # TODO.
    test_suite="testing",
    setup_requires=["pytest-runner"],
    tests_require=["pytest", "pytest-cov"],
    include_package_data=True,
    license='TODO',  # TODO: set license string. 
    keywords='empty project TODO keywords'
)

