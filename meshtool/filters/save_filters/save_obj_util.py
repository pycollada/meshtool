import collada
import os.path

def materialParameterAsFloat(value, default=None):
    if isinstance(value, collada.material.Map):
        return default
    elif isinstance(value, tuple):
        return sum(value)/float(len(value))
    elif isinstance(value, float):
        return value
    else:
        return default

def formatMaterialField(field_name, value, mesh=None, mtlfilename=None):
    if isinstance(value, collada.material.Map):
        rel_img_path = value.sampler.surface.image.path
        # This let's relative paths get worked out, but since
        # sometimes this data is not going into files (e.g. when it's
        # going into a zip), this approach only works sometimes. For
        # other situations, like zips, you are responsible for making
        # sure the path is setup properly.
        if mesh is not None and mesh.filename is not None and mtlfilename is not None:
            # First we need to get the real relative path to the
            # texture by combining relative path to source dae +
            # relative path from dae to texture
            rel_img_path = os.path.join( os.path.dirname(mesh.filename), rel_img_path )
            # Then, use that to convert it to be relative to the mtl file.
            rel_img_path = os.path.relpath(rel_img_path, os.path.dirname(mtlfilename))
        return "map_%s %s" % (field_name, rel_img_path)
    elif isinstance(value, tuple):
        return "%s %f %f %f" % (field_name, value[0], value[1], value[2])
    elif isinstance(value, float):
        return "%s %f" % (field_name, value)
    else:
        return None


def write_mtl(mesh, fmtl, mtlfilename=None):
    """Write Wavefront OBJ-style materials to a file-like object.

    :param collada.Collada mesh:
      The collada mesh to get materials from
    :param fmtl:
      A file-like object to write material information to.
    :param str mtlfilename:
      The path to the material file being written, if it is truly a
      file. This is used to ensure paths to textures are correct.
    """

    for mtl in mesh.materials:
        print >>fmtl, "newmtl", mtl.id
        if mtl.effect.ambient is not None:
            print >>fmtl, formatMaterialField('Ka', mtl.effect.ambient, mesh, mtlfilename)
        if mtl.effect.diffuse is not None:
            print >>fmtl, formatMaterialField('Kd', mtl.effect.diffuse, mesh, mtlfilename)
        if mtl.effect.specular is not None:
            print >>fmtl, formatMaterialField('Ks', mtl.effect.specular, mesh, mtlfilename)
        if mtl.effect.shininess is not None:
            print >>fmtl, formatMaterialField('Ns', mtl.effect.shininess, mesh, mtlfilename)
        # d and Tr are both used for transparency
        if mtl.effect.transparent is not None:
            transparent_float = materialParameterAsFloat(mtl.effect.transparent, default=1.0)
            print >>fmtl, formatMaterialField('d', transparent_float, mesh, mtlfilename)
            print >>fmtl, formatMaterialField('Tr', transparent_float, mesh, mtlfilename)

        # Illumination model: 1 = diffuse, 2 = with specular
        illum_model = 1 if mtl.effect.shadingtype in ['lambert', 'constant'] else 2
        print >>fmtl, "illum", illum_model

        print >>fmtl


def write_obj(mesh, mtlfilename, f):
    """Write Wavefront OBJ contents of mesh to a file-like object."""

    print >>f, "mtllib", mtlfilename

    # Iterate through all primitives in each geometry instance
    vert_offset = 1
    norm_offset = 1
    tc_offset = 1
    for boundgeom in mesh.scene.objects('geometry'):
        print >>f, "# %s - %s" % (boundgeom.original.name, boundgeom.original.id)
        for boundprim in boundgeom.primitives():
            # Determine the properties of these primitives we're going
            # to use
            emit_normals = boundprim.normal is not None
            emit_texcoords = boundprim.texcoordset is not None and len(boundprim.texcoordset) > 0
            if emit_texcoords and len(boundprim.texcoordset) > 1:
                raise FilterException("OBJ only supports one texture coordinate set.")

            # Write transformed vertices, normals, texcoords
            for vi in range(boundprim.vertex.shape[0]):
                v = boundprim.vertex[vi,:]
                print >>f, "v %f %f %f" % (v[0], v[1], v[2])

            if emit_normals:
                for vi in range(boundprim.normal.shape[0]):
                    v = boundprim.normal[vi,:]
                    print >>f, "vn %f %f %f" % (v[0], v[1], v[2])

            if emit_texcoords:
                for vi in range(boundprim.texcoordset[0].shape[0]):
                    v = boundprim.texcoordset[0][vi,:]
                    print >>f, "vt %f %f" % (v[0], v[1])

            # Start using the right material
            print >>f, "usemtl", boundprim.material.id

            # Write transformed primitives
            for face in boundprim:
                print >>f, "f ",
                if emit_normals and emit_texcoords:
                    for vidx,nidx,tcidx in zip(face.indices,face.normal_indices,face.texcoord_indices[0]):
                        print >>f, "%d/%d/%d " % (vidx + vert_offset, tcidx + tc_offset, nidx + norm_offset),
                elif emit_normals:
                    for vidx,nidx in zip(face.indices,face.normal_indices):
                        print >>f, "%d//%d " % (vidx + vert_offset, nidx + norm_offset),
                else:
                    for vidx in face.indices:
                        print >>f, "%d " % (vidx + vert_offset),
                print >>f

            # Finally, update offsets
            vert_offset += boundprim.vertex.shape[0]
            if emit_normals:
                norm_offset += boundprim.normal.shape[0]
            if emit_texcoords:
                tc_offset += boundprim.texcoordset[0].shape[0]
