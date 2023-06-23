from setuptools import setup

setup(
    name='NeuroGraph',
    version='1.0.0',
    description='A Python package for graph-based Neuroimaging benchmarks and tools',
    author='Anwar Said',
    author_email='anwar.said@vanderbilt.edu',
    packages=['neurograph'],
    install_requires=[
        # List any dependencies your package requires
        'boto3==1.26.158',
        'nilearn==0.10.1',
        'networkx==2.6',
        
    ],
)