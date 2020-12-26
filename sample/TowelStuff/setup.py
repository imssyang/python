from distutils.core import setup

setup(
    name='sampleTowelStuff',
    version='0.1.2',
    author='John M. Gabriele',
    author_email='jmg3000@gmail.com',
    packages=['towel_stuff', 'towel_stuff.test'],
    scripts=['bin/stowe-towels.py', 'bin/wash-towels.py'],
    url='http://pypi.python.org/pypi/TowelStuff/',
    license='LICENSE.txt',
    description='Useful towel-related stuff.',
    long_description=open('README.txt').read(),
)

