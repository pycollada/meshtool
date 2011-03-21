from meshtool.args import *
from meshtool.filters.base_filters import *
import os

def FilterGenerator():
    class ColladaSaveFilter(SaveFilter):
        def __init__(self):
            super(ColladaSaveFilter, self).__init__('save_collada', 'Saves a collada file')
        def apply(self, mesh, filename):
            if os.path.exists(filename):
                raise FilterException("specified filename already exists")
            f = open(filename, 'w')
            mesh.root.write(f)
            f.close()
            return mesh
    return ColladaSaveFilter()
