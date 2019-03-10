import io
import os
import sys
from shutil import rmtree

from setuptools import find_packages, setup, Command

# Package meta-data.
NAME = 'brawlstars-ai'
DESCRIPTION = 'My short description for my project.'
URL = 'https://github.com/workofart/brawlstars-ai'
EMAIL = 'hanxiangp@gmail.com'
AUTHOR = 'Henry Pan'
REQUIRES_PYTHON = '>=3.5.0'
VERSION = '0.0.1'

REQUIRED = [
    'tensorflow', 'tflearn', 'numpy', 'opencv-python', 'pyautogui' , 'pillow', 'pandas'
]


# Where the magic happens:
setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=('tests',)),
    install_requires=REQUIRED,
    include_package_data=True,
)
