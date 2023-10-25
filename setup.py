from setuptools import setup

setup(
    name='NeuroGraph',
    version='2.2.0',
    description='A Python package for graph-based neuroimaging benchmarks and tools',
    author='Anwar Said',
    author_email='<anwar.said@vanderbilt.edu>',
    packages=['NeuroGraph'],
    install_requires=[
        # List any dependencies your package requires
        'boto3',
        'nilearn',
        'nibabel',
        'networkx',
        'pandas',
        'sphinx_rtd_theme'
    ],
    keywords = ['python','neuroimaging','graph machine learning'],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
    ],
)
