from __future__ import absolute_import
from __future__ import print_function
import setuptools

REQUIRED_PACKAGES = [
    'tensorflow>=2.1.0',
    'moviepy==1.0.1',
    'numpy',
    'gym>=0.17.1'
    ]

setuptools.setup(
    name='gymboard',
    packages=['gymboard'],
    package_dir={'':'src'},
    version='0.0.1',
    author="Mishig Davaadorj",
    author_email="dmishig@gmail.com",
    description='Render OpenAI Gym envs in TensorBoard',
    install_requires=REQUIRED_PACKAGES,
    python_requires='>=3.6',
    url="https://github.com/mishig25/GymBoard",
)

# run: python setup.py sdist