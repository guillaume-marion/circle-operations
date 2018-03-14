from distutils.core import setup

setup(
    name='circle-operations',
    version='0.1dev',
    author='G. Marion',
    packages=['geom'],
    url='http://github.com/GuillaumeDMMarion/circle-operations',
    description='Some simple circle operations wrapped in a class.',
    long_description=open('README.rst').read(),
    install_requires=[
        "numpy >= 1.14.0",
    ],
)