from meshtool.filters.base_filters import SaveFilter, FilterException
from itertools import chain
import os
import collada
import numpy

def FilterGenerator():
    class PlySaveFilter(SaveFilter):
        def __init__(self):
            super(PlySaveFilter, self).__init__('save_ply', 'Saves a collada model in PLY format')

        def uniqify_multidim_indexes(self, sourcedata, indices, return_map=False):
            """Just like numpy.unique except that it works with multi-dimensional arrays.
               Given a source array and indexes into the source array, will keep only unique
               indices into the source array, rewriting the indices to point to the new
               compressed source array"""
            unique_data, index_map = numpy.unique(sourcedata.view([('',sourcedata.dtype)]*sourcedata.shape[1]), return_inverse=True)
            index_map = numpy.cast['int32'](index_map)
            if return_map:
                return unique_data.view(sourcedata.dtype).reshape(-1,sourcedata.shape[1]), index_map[indices], index_map
            return unique_data.view(sourcedata.dtype).reshape(-1,sourcedata.shape[1]), index_map[indices]

        def aggregate_dae(self, mesh):
            all_vertices = []
            all_vert_indices = []
            vertex_offset = 0
            num_prims = 0

            for boundgeom in chain(mesh.scene.objects('geometry'), mesh.scene.objects('controller')):
                if isinstance(boundgeom, collada.controller.BoundController):
                    boundgeom = boundgeom.geometry

                for boundprim in boundgeom.primitives():
                    if boundprim.vertex_index is None or len(boundprim.vertex_index) == 0:
                        continue
                    if not isinstance(boundprim, collada.triangleset.BoundTriangleSet):
                        continue

                    all_vertices.append(boundprim.vertex)
                    all_vert_indices.append(boundprim.vertex_index + vertex_offset)
                    vertex_offset += len(boundprim.vertex)
                    num_prims += 1

            all_vertices = numpy.concatenate(all_vertices)
            all_vert_indices = numpy.concatenate(all_vert_indices)
            all_vertices, all_vert_indices = self.uniqify_multidim_indexes(all_vertices, all_vert_indices)

            return all_vertices, all_vert_indices

        def apply(self, mesh, filename):
            if os.path.exists(filename):
                raise FilterException("specified filename already exists")

            all_vertices, all_faces = self.aggregate_dae(mesh);

            outputfile = open(filename, "w");
            outputfile.write("ply\n");
            outputfile.write("format ascii 1.0\n");
            outputfile.write( "element vertex %d\n" % len(all_vertices));
            outputfile.write("property float x\n");
            outputfile.write("property float y\n");
            outputfile.write("property float z\n");
            outputfile.write("element face %d\n" %  len(all_faces));
            outputfile.write("property list uchar int vertex_indices\n");
            outputfile.write("end_header\n");

            for vertex in all_vertices:
                outputfile.write( "%(v0)f %(v1)f %(v2)f\n" % {'v0':vertex[0], 'v1':vertex[1], 'v2':vertex[2]} )

            for face in all_faces:
                outputfile.write( "3 %(v0)d %(v1)d %(v2)d\n" % {'v0':face[0], 'v1':face[1], 'v2':face[2] }  )

            outputfile.close()
            return mesh

    return PlySaveFilter()

from meshtool.filters import factory
factory.register(FilterGenerator().name, FilterGenerator)
