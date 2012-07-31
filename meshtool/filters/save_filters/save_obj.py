from meshtool.filters.base_filters import SaveFilter, FilterException
import os
import save_obj_util

def FilterGenerator():
    class ObjSaveFilter(SaveFilter):
        def __init__(self):
            super(ObjSaveFilter, self).__init__('save_obj', 'Saves a mesh as an OBJ file')

        def apply(self, mesh, filename):
            if os.path.exists(filename):
                raise FilterException("Specified mesh filename already exists")

            dotloc = filename.rfind('.')
            if dotloc == -1:
                mtlfilename = filename
            else:
                mtlfilename = filename[:dotloc]
            mtlfilename += '.mtl'
            if os.path.exists(mtlfilename):
                raise FilterException("Material filename already exists")

            # Handle materials first, iterating through all materials
            fmtl = open(mtlfilename, 'w')
            save_obj_util.write_mtl(mesh, fmtl, mtlfilename)
            fmtl.close()

            f = open(filename, 'w')
            rel_mtlfilename = os.path.relpath(mtlfilename, os.path.dirname(filename))
            save_obj_util.write_obj(mesh, rel_mtlfilename, f)
            f.close()

            return mesh
    return ObjSaveFilter()
from meshtool.filters import factory
factory.register(FilterGenerator().name, FilterGenerator)
