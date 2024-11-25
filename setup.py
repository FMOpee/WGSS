from setuptools import setup, find_packages

setup(
    name='WGSS',
    version='1.0.0',
    description='A package that includes a Bengali summarizer using WGSS sentence similarity algorithm',
    author='Fahim Morshed (FMOpee)',
    author_email='f.morshed.opee@gmail.com',
    packages=find_packages(),
    install_requires=[
        'nltk',
        'numpy',
        'fasttext',
        'scikit-learn'
    ]
)
