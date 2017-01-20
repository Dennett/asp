from setuptools import setup

setup(name='funniest',
      version='0.1',
      description='Approximate Stochastic Paths',
      url='https://github.com/Dennett/asp',
      author='Michael Ramsey',
      author_email='michael.ramsey@gmail.com',
      license='GNU GPLv3',
      packages=['asp'],
      install_requires=[
      'markdown',
      'networkx',
      'numpy',
      'matplotlib',
      'itertools',
      'scipy',
      'time',
      'collections',
      'pandas',
      'math'
      ],
      zip_safe=False)