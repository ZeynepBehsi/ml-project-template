from setuptools import find_packages, setup

setup(
    name='ml-project-template',
    packages=find_packages(),
    version='0.1.0',
    description='A standardized data science project structure for machine learning workflows',
    author='Zeynep',
    license='MIT',
    install_requires=[
        'numpy>=1.24.0',
        'pandas>=2.0.0',
        'scikit-learn>=1.3.0',
    ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    test_suite='tests',
)