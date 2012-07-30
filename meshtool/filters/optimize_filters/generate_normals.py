from meshtool.filters.base_filters import OptimizationFilter
import collada
import numpy

def generateNormals(mesh):
    for geom in mesh.geometries:
        prims_by_src = {}
        for prim in geom.primitives:
            if type(prim) is collada.triangleset.TriangleSet and prim.normal is None and len(prim) > 0:
                vertex_source = prim.sources['VERTEX'][0][2]
                if vertex_source in prims_by_src:
                    prims_by_src[vertex_source].append(prim)
                else:
                    prims_by_src[vertex_source] = [prim]

        for srcid, primlist in prims_by_src.iteritems():
            vertex = geom.sourceById[srcid[1:]].data
            norms = numpy.zeros( vertex.shape, dtype=vertex.dtype )
            
            #combine all of the vertex indices for each primitive to one array
            concat_arrays = []
            for prim in primlist:
                concat_arrays.append(prim._vertex_index)
            combined_index = numpy.concatenate(concat_arrays)
            
            tris = vertex[combined_index]
            
            #calculate the per-face normals and apply each equally to the vertices
            n = numpy.cross( tris[::,1] - tris[::,0], tris[::,2] - tris[::,0] )
            collada.util.normalize_v3(n)
            norms[ combined_index[:,0] ] += n
            norms[ combined_index[:,1] ] += n
            norms[ combined_index[:,2] ] += n
            collada.util.normalize_v3(norms)

            #now let's create a source for this new normal data and add it
            unique_src = srcid[1:] + '-normals'
            while unique_src in geom.sourceById:
                unique_src += '-x'
            normal_src = collada.source.FloatSource(unique_src, norms, ('X', 'Y', 'Z'))
            geom.sourceById[normal_src.id] = normal_src
            
            for prim in primlist:
                #add the normal source to the list of inputs
                input_list = prim.getInputList()
                input_list.addInput(prim.sources['VERTEX'][0][0], 'NORMAL', '#' + unique_src)
                
                #delete the primitive from the geometry
                for i, p in enumerate(geom.primitives):
                    if p == prim:
                        todel = i
                        break
                del geom.primitives[todel]
                
                #create and append new one
                newtriset = geom.createTriangleSet(prim.index, input_list, prim.material)
                geom.primitives.append(newtriset)


def FilterGenerator():
    class GenerateNormalsFilter(OptimizationFilter):
        def __init__(self):
            super(GenerateNormalsFilter, self).__init__('generate_normals', "Generates normals for any triangle sets that don't have any")
        def apply(self, mesh):
            generateNormals(mesh)
            return mesh
    return GenerateNormalsFilter()
from meshtool.filters import factory
factory.register(FilterGenerator().name, FilterGenerator)