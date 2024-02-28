from setuptools import setup, find_packages

setup(
    name='NeuroGraph',
    version='3.0.0',
    long_description='A Python package for graph-based neuroimaging benchmarks and tools',
    long_description_content_type='text/markdown',
    author='Anwar Said',
    author_email='<anwar.said@vanderbilt.edu>',
    packages=find_packages(),
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
