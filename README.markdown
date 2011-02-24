INSTALLATION
============
    git clone git://github.com/pycollada/meshtool.git meshtool
    cd meshtool
    git submodule init
    git submodule update
    ./meshtool-cli --help

FILTER LIST
===========
* load_collada file - Loads a collada file from file
* save_collada file - Saves the collada file to file
* print_textures - Prints a list of textures
* save_screenshot - Saves a screenshot of rendered collada to file (no textures) - depends on [Panda3d](http://www.panda3d.org/)

EXAMPLES
========
    $ ./meshtool-cli --load_collada duck.dae --print_textures
    ./duckCM.tga
    $
