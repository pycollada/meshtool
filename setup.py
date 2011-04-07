from setuptools import find_packages, setup

setup(
    name = "meshtool",
    version = "0.1",
    description = "Tool for manipulating collada meshes",
    author = "Jeff Terrace and contributors",
    author_email = 'jterrace@gmail.com',
    platforms=["any"],
    license="BSD",
    install_requires=['pycollada>=0.1', 'PIL'],
    url = "https://github.com/pycollada/meshtool",
    entry_points = {
        'console_scripts':[
            'meshtool = meshtool.__main__:main'
        ]
    },
    packages = find_packages()
)
