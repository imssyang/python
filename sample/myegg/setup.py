from setuptools import setup, find_packages

setup(
    name='myegg',
    version='1.0.0',
    install_requires=[
    ],
    packages=find_packages(exclude=['test']),
    entry_points={
        'console_scripts': [
            'myegg = my_egg.hello:run'
        ]
    },
    test_suite='test'
)
