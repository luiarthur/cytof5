from setuptools import setup

setup(name='vb',
      version='0.1',
      description='My VB stuff',
      url='http://github.com/Cytof5/sims/cb/vb',
      author='Arthur Lui',
      author_email='luiarthur@gmail.com',
      license='MIT',
      packages=['vb'],
      test_suite='nose.collector',
      tests_require=[ 'nose' ],
      install_requires=['torch'],
      zip_safe=False)
