import os
from setuptools import setup, find_packages

long_description = '''MDDI-SCL'''

setup(
    name='MDDI-SCL',
    version='0.0.1',
    author='Shenggeng Lin',
    author_email='linsg4521@sjtu.edu.cn',
    py_modules=['task1_big','task1_small','task2_big','task2_small','task3_big''task3_small'],
    description='MDDI-SCL',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=['Program running environment requirements.txt'],
    license='MIT',
    packages=find_packages()
)
