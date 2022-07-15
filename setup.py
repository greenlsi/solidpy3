from distutils.core import setup
setup(
  name='solidpy3',
  packages=['solid'],
  version='0.2',
  description='A comprehensive gradient-free optimization library',
  author='Devin Soni (original author), Román Cárdenas',
  author_email='r.cardenas@upm.es',
  url='https://github.com/greenlsi/solidpy3',
  # download_url='https://github.com/100/Solid/archive/0.1.tar.gz',
  keywords=['metaheuristic', 'optimization', 'algorithm', 'artificial intelligence', 'machine learning'],
  classifiers=[
    'Programming Language :: Python :: 3.6',
  ],
  install_requires=[
    'numpy>=1.17',
  ],
)
