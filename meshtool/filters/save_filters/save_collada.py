from args import *
from ..base_filters import *
import os
from xml.etree import ElementTree

def FilterGenerator():
    class ColladaSaveFilter(SaveFilter):
        def __init__(self):
            super(ColladaSaveFilter, self).__init__('save_collada', 'Saves a collada file')
        def apply(self, mesh, filename):
            if os.path.exists(filename):
                raise FilterException("specified filename already exists")
            #mesh.save()
            ElementTree.ElementTree(mesh.root).write(filename)
            return mesh
    return ColladaSaveFilter()
