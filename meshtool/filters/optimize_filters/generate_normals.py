from meshtool.args import *
from meshtool.filters.base_filters import *
import collada
import numpy

def generateNormals(mesh):
    for geom in mesh.geometries:
        prims = []
        for prim in geom.primitives:
            if type(prim) is collada.triangleset.TriangleSet and prim.normal is None:
                prim.generateNormals()
                
                #get the generated normals values and indexes
                normal_source_vals = prim.normal
                normal_source_vals.shape = -1
                normal_index = prim.normal_index
                normal_index.shape = (-1, 3, 1)
                
                #add the normals source to the geometry
                unique_src = 'tri-normal-src'
                while unique_src in geom.sourceById:
                    unique_src += '-x'
                normal_src = collada.source.FloatSource(unique_src, normal_source_vals, ('X', 'Y', 'Z'))
                geom.sourceById[normal_src.id] = normal_src
                
                #add the normal source to the list of inputs
                input_list = prim.getInputList()
                maxinput = max([input[0] for input in input_list.getList()])
                input_list.addInput(maxinput+1, 'NORMAL', '#' + unique_src)
                               
                #append the new index to the indexes array
                all_indexes = prim.index
                all_indexes = numpy.append(all_indexes, normal_index, 2)
                all_indexes = all_indexes.flatten()
                
                #and finally create and append
                newtriset = geom.createTriangleSet(all_indexes, input_list, prim.material)
                prims.append(newtriset)
            else:
                prims.append(prim)
        geom.primitives = prims

def FilterGenerator():
    class GenerateNormalsFilter(OpFilter):
        def __init__(self):
            super(GenerateNormalsFilter, self).__init__('generate_normals', "Generates normals for any triangle sets that don't have any")
        def apply(self, mesh):
            generateNormals(mesh)
            return mesh
    return GenerateNormalsFilter()
