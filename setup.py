from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name='qsim',
    version='0.0.1',
    author='Robert Daniel Nutt',
    author_email='rdnutt3@gmail.com',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/rDanielNutt/Qsim',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'cupy',
        'matplotlib',
        'numpy'
    ]
)