INSTALLATION
============
    git clone git://github.com/pycollada/meshtool.git meshtool
    cd meshtool
    git submodule init
    git submodule update
    ./meshtool-cli --help

FILTER LIST
===========

Load
----
* ``--load_collada file`` - Loads a collada file

Save
----
* ``--save_collada file`` - Saves a collada file

Print
-----
* ``--print_info`` - Prints a bunch of information about the mesh to the console
* ``--print_json`` - Prints a bunch of information about the mesh in a JSON format
* ``--print_textures`` - Prints a list of the embedded images in the mesh

View
----
* ``--viewer`` - Uses panda3d to bring up a viewer - depends on [Panda3d](http://www.panda3d.org/)

Screenshots
-----------
* ``--save_screenshot file`` - Saves a screenshot of the rendered collada file - depends on [Panda3d](http://www.panda3d.org/)
* ``--save_rotate_screenshots file N W H`` - Saves N screenshots of size WxH, rotating evenly spaced around the object between shots. Each screenshot file will be file.n.png - depends on [Panda3d](http://www.panda3d.org/)

Mesh Operations
---------------
* ``--combine_effects`` - Combines identical effects
* ``--combine_materials`` - Combines identical materials
* ``--combine_primitives`` - Combines primitives within a geometry if they have the same sources and scene material mapping (triangle sets only)
* ``--strip_lines`` - Strips any lines from the document
* ``--triangulate`` - Replaces any polylist or polygons with triangles
* ``--generate_normals`` - Generates normals for any triangle sets that don't have any

Texture Operations
------------------
* ``--save_mipmaps`` - Saves mipmaps to disk in concatenated PNG format in the same location as textures but with an added .mipmap

EXAMPLES
========
    $ ./meshtool-cli --load_collada duck.dae --print_textures
    ./duckCM.tga
    $
