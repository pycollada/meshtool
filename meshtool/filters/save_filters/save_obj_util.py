import collada

def materialParameterAsFloat(value, default=None):
    if isinstance(value, collada.material.Map):
        return default
    elif isinstance(value, tuple):
        return sum(value)/float(len(value))
    elif isinstance(value, float):
        return value
    else:
        return default

def formatMaterialField(field_name, value):
    if isinstance(value, collada.material.Map):
        return "map_%s %s" % (field_name, value.sampler.surface.image.path)
    elif isinstance(value, tuple):
        return "%s %f %f %f" % (field_name, value[0], value[1], value[2])
    elif isinstance(value, float):
        return "%s %f" % (field_name, value)
    else:
        return None


def write_mtl(mesh, fmtl):
    """Write Wavefront OBJ-style materials to a file-like object."""

    for mtl in mesh.materials:
        print >>fmtl, "newmtl", mtl.id
        if mtl.effect.ambient is not None:
            print >>fmtl, formatMaterialField('Ka', mtl.effect.ambient)
        if mtl.effect.diffuse is not None:
            print >>fmtl, formatMaterialField('Kd', mtl.effect.diffuse)
        if mtl.effect.specular is not None:
            print >>fmtl, formatMaterialField('Ks', mtl.effect.specular)
        if mtl.effect.shininess is not None:
            print >>fmtl, formatMaterialField('Ns', mtl.effect.shininess)
        # d and Tr are both used for transparency
        if mtl.effect.transparent is not None:
            transparent_float = materialParameterAsFloat(mtl.effect.transparent, default=1.0)
            print >>fmtl, formatMaterialField('d', transparent_float)
            print >>fmtl, formatMaterialField('Tr', transparent_float)

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
