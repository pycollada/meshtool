from setuptools import find_packages, setup

install_requires = []

try: import collada
except ImportError: install_requires.append('pycollada>=0.4')

try: import PIL
except ImportError: install_requires.append('PIL')

try: import argparse
except ImportError: install_requires.append('argparse')

setup(
    name = "meshtool",
    version = "0.3",
    description = "Tool for manipulating collada meshes",
    author = "Jeff Terrace and contributors",
    author_email = 'jterrace@gmail.com',
    platforms=["any"],
    license="BSD",
    install_requires=install_requires,
    url = "https://github.com/pycollada/meshtool",
    entry_points = {
        'console_scripts':[
            'meshtool = meshtool.__main__:main'
        ]
    },
    packages = find_packages()
)
