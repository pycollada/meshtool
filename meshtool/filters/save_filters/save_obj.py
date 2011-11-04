from meshtool.args import *
from meshtool.filters.base_filters import *
import os
import collada
import save_obj_util

def FilterGenerator():
    class ObjSaveFilter(SaveFilter):
        def __init__(self):
            super(ObjSaveFilter, self).__init__('save_obj', 'Saves a mesh as an OBJ file')

            self.arguments.append(FileArgument(
                    'mtlfilename', 'Where to save the material properties'))

        def apply(self, mesh, filename, mtlfilename):
            if os.path.exists(filename):
                raise FilterException("Specified mesh filename already exists")

            if os.path.exists(mtlfilename):
                raise FilterException("Specified material filename already exists")

            # Handle materials first, iterating through all materials
            fmtl = open(mtlfilename, 'w')
            save_obj_util.write_mtl(mesh, fmtl)
            fmtl.close()

            f = open(filename, 'w')
            rel_mtlfilename = os.path.relpath(mtlfilename, os.path.dirname(filename))
            save_obj_util.write_obj(mesh, rel_mtlfilename, f)
            f.close()

            return mesh
    return ObjSaveFilter()
from meshtool.filters import factory
factory.register(FilterGenerator().name, FilterGenerator)
