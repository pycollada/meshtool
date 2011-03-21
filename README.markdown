INSTALLATION
============
    git clone git://github.com/pycollada/meshtool.git meshtool
    cd meshtool
    git submodule init
    git submodule update
    ./meshtool-cli --help

FILTER LIST
===========
* ``--load_collada file`` - Loads a collada file
* ``--save_collada file`` - Saves a collada file
* ``--print_textures`` - Prints a list of the embedded images in the mesh
* ``--save_screenshot file`` - Saves a screenshot of the rendered collada file - depends on [Panda3d](http://www.panda3d.org/)
* ``--save_rotate_screenshots file N W H`` - Saves N screenshots of size WxH, rotating evenly spaced around the object between shots. Each screenshot file will be file.n.png - depends on [Panda3d](http://www.panda3d.org/)
* ``--viewer`` - Uses panda3d to bring up a viewer - depends on [Panda3d](http://www.panda3d.org/)
* ``--print_json`` - Prints a bunch of information aobut the mesh in a JSON format

EXAMPLES
========
    $ ./meshtool-cli --load_collada duck.dae --print_textures
    ./duckCM.tga
    $
