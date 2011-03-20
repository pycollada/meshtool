INSTALLATION
============
    git clone git://github.com/pycollada/meshtool.git meshtool
    cd meshtool
    git submodule init
    git submodule update
    ./meshtool-cli --help

FILTER LIST
===========
* <pre>--load_collada file</pre> - Loads a collada file
* <pre>--save_collada file</pre> - Saves a collada file
* <pre>--print_textures</pre> - Prints a list of the embedded images in the mesh
* <pre>--save_screenshot file</pre> - Saves a screenshot of the rendered collada file - depends on [Panda3d](http://www.panda3d.org/)
* <pre>--save_rotate_screenshots file N W H</pre> - Saves N screenshots of size WxH, rotating evenly spaced around the object between shots. Each screenshot file will be file.n.png - depends on [Panda3d](http://www.panda3d.org/)
* <pre>--viewer</pre> - Uses panda3d to bring up a viewer - depends on [Panda3d](http://www.panda3d.org/)

EXAMPLES
========
    $ ./meshtool-cli --load_collada duck.dae --print_textures
    ./duckCM.tga
    $
