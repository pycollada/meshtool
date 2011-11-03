from meshtool.args import *
from meshtool.filters.base_filters import *
import os
import collada

def FilterGenerator():
    class ObjSaveFilter(SaveFilter):
        def __init__(self):
            super(ObjSaveFilter, self).__init__('save_obj', 'Saves a mesh as an OBJ file')

            self.arguments.append(FileArgument(
                    'mtlfilename', 'Where to save the material properties'))

        @staticmethod
        def materialParameterAsFloat(value, default=None):
            if isinstance(value, collada.material.Map):
                return default
            elif isinstance(value, tuple):
                return sum(value)/float(len(value))
            elif isinstance(value, float):
                return value
            else:
                return default

        @staticmethod
        def formatMaterialField(field_name, value):
            if isinstance(value, collada.material.Map):
                return "map_%s %s" % (field_name, value.sampler.surface.image.path)
            elif isinstance(value, tuple):
                return "%s %f %f %f" % (field_name, value[0], value[1], value[2])
            elif isinstance(value, float):
                return "%s %f" % (field_name, value)
            else:
                return None


        def apply(self, mesh, filename, mtlfilename):
            if os.path.exists(filename):
                raise FilterException("Specified mesh filename already exists")

            if os.path.exists(mtlfilename):
                raise FilterException("Specified material filename already exists")

            # Handle materials first, iterating through all materials
            fmtl = open(mtlfilename, 'w')
            for mtl in mesh.materials:
                print >>fmtl, "newmtl", mtl.id
                if mtl.effect.ambient is not None:
                    print >>fmtl, ObjSaveFilter.formatMaterialField('Ka', mtl.effect.ambient)
                if mtl.effect.diffuse is not None:
                    print >>fmtl, ObjSaveFilter.formatMaterialField('Kd', mtl.effect.diffuse)
                if mtl.effect.specular is not None:
                    print >>fmtl, ObjSaveFilter.formatMaterialField('Ks', mtl.effect.specular)
                if mtl.effect.shininess is not None:
                    print >>fmtl, ObjSaveFilter.formatMaterialField('Ns', mtl.effect.shininess)
                # d and Tr are both used for transparency
                if mtl.effect.transparent is not None:
                    transparent_float = ObjSaveFilter.materialParameterAsFloat(mtl.effect.transparent, default=1.0)
                    print >>fmtl, ObjSaveFilter.formatMaterialField('d', transparent_float)
                    print >>fmtl, ObjSaveFilter.formatMaterialField('Tr', transparent_float)

                # Illumination model: 1 = diffuse, 2 = with specular
                illum_model = 1 if mtl.effect.shadingtype in ['lambert', 'constant'] else 2
                print >>fmtl, "illum", illum_model

                print >>fmtl
            fmtl.close()

            f = open(filename, 'w')
            print >>f, "mtllib", os.path.relpath(mtlfilename, os.path.dirname(filename))

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

                    # Start using the right material
                    print >>f, "usemtl", boundprim.material.id

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
