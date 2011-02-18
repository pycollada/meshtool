from args import *
from ..base_filters import *

def FilterGenerator():
    class PrintTexturesFilter(OpFilter):
        def __init__(self):
            super(PrintTexturesFilter, self).__init__('print_textures', 'Prints a list of the embedded images in the mesh')
        def apply(self, mesh):
            for t in mesh.images:
                print t.path
            return mesh
    return PrintTexturesFilter()
