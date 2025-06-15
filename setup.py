from setuptools import setup, find_packages

setup(
    name='basis',
    version='0.1.0',
    author='Protim Roy',
    author_email='mail@protimroy.com',
    description='Basis is a low-level PyTorch library for building neural networks from scratch',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'torch'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.12.9',
)