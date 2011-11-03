from meshtool.args import *
from meshtool.filters.base_filters import *
import os
import collada

def FilterGenerator():
    class ObjSaveFilter(SaveFilter):
        def __init__(self):
            super(ObjSaveFilter, self).__init__('save_obj', 'Saves a mesh as an OBJ file')

        def apply(self, mesh, filename):
            if os.path.exists(filename):
                raise FilterException("specified filename already exists")

            f = open(filename, 'w')

            # Iterate through all primitives in each geometry instance
            vert_offset = 1
            norm_offset = 1
            tc_offset = 1
            for boundgeom in mesh.scene.objects('geometry'):
                print >>f, "# %s - %s" % (boundgeom.original.name, boundgeom.original.id)
                for boundprim in boundgeom.primitives():
                    # Write transformed vertices, normals, texcoords
                    for vi in range(boundprim.vertex.shape[0]):
                        v = boundprim.vertex[vi,:]
                        print >>f, "v %f %f %f" % (v[0], v[1], v[2])

                    if boundprim.normal is not None:
                        for vi in range(boundprim.normal.shape[0]):
                            v = boundprim.normal[vi,:]
                            print >>f, "vn %f %f %f" % (v[0], v[1], v[2])

                    if boundprim.texcoordset is not None:
                        if len(boundprim.texcoordset) > 1:
                            raise FilterException("OBJ only supports one texture coordinate set.")
                        for vi in range(boundprim.texcoordset[0].shape[0]):
                            v = boundprim.texcoordset[0][vi,:]
                            print >>f, "vt %f %f" % (v[0], v[1])

                    # Write transformed primitives
                    for face in boundprim:
                        print >>f, "f ",
                        if face.indices is not None and face.normal_indices is not None and face.texcoord_indices is not None:
                            for vidx,nidx,tcidx in zip(face.indices,face.normal_indices,face.texcoord_indices[0]):
                                print >>f, "%d/%d/%d " % (vidx + vert_offset, tcidx + tc_offset, nidx + norm_offset),
                        elif face.indices is not None and face.normal_indices is not None:
                            for vidx,nidx in zip(face.indices,face.normal_indices):
                                print >>f, "%d//%d " % (vidx + vert_offset, nidx + norm_offset),
                        else:
                            for vidx in face.indices:
                                print >>f, "%d " % (vidx + vert_offset),
                        print >>f

                    # Finally, update offsets
                    vert_offset += boundprim.vertex.shape[0]
                    if boundprim.normal is not None:
                        norm_offset += boundprim.normal.shape[0]
                    if boundprim.texcoordset is not None:
                        tc_offset += boundprim.texcoordset[0].shape[0]
            f.close()

            return mesh
    return ObjSaveFilter()
from meshtool.filters import factory
factory.register(FilterGenerator().name, FilterGenerator)
