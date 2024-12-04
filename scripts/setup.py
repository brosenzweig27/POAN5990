from setuptools import find_packages, setup

setup(
    name='network_project',
    packages=find_packages(include=['library']),
    version='0.1.0',
    description='Python Library for POAN5997',
    author='Benny Rosenzweig',
    install_requires=['pandas', 'numpy', 'torch', 'torch.nn', 'random'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
)