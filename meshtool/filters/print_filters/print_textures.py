from meshtool.args import *
from meshtool.filters.base_filters import *

def getTextures(mesh):
    texs = []
    for t in mesh.images:
        texs.append(t.path)
    return texs

def FilterGenerator():
    class PrintTexturesFilter(OpFilter):
        def __init__(self):
            super(PrintTexturesFilter, self).__init__('print_textures', 'Prints a list of the embedded images in the mesh')
        def apply(self, mesh):
            for t in getTextures(mesh):
                print t
            return mesh
    return PrintTexturesFilter()
