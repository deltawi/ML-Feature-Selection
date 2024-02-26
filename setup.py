from setuptools import setup, find_packages

requirements = ['pip', 'logzero', 'pandas', 'scikit_learn']

setup(
    name='FeatureSelector',
    packages=find_packages(),
    install_requires = requirements,
    zip_safe=False,
)