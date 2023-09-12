from setuptools import setup
from rondo import __version__

setup(
    name='rondo',
    version=__version__,
    license='MIT License',
    description='A simple framework of neural networks for deep learning.',
    install_requires=['numpy'],
    author='Kensuke Kamata',
    author_email='kn12kamata@gmail.com',
    url='',
    packages=['rondo']
)
