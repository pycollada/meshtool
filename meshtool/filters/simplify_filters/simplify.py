from meshtool.args import *
from meshtool.filters.base_filters import *
import os
import random
import numpy
from .mesh_simplification import MeshSimplification
import collada

def FilterGenerator():
    class ColladaSimplifyFilter(OpFilter):
        def __init__(self):
            super(ColladaSimplifyFilter, self).__init__('simplify', 'Simplifies a collada file')
        def apply(self, mesh):
            prim = mesh.geometries[0].primitives[0]
            if not isinstance(prim, collada.triangleset.TriangleSet):
                raise FilterException("Must triangulate before simplifying")
            s = MeshSimplification(prim.vertex, prim.vertex_index)
            for i in range(len(prim.vertex)*4/5):
                s.contractOnce()
                # if not s.isValid(): raise Exception
            floatsource = collada.source.FloatSource("foobar", numpy.array(s.vertices), ('X', 'Y', 'Z'))
            mesh.geometries[0].sourceById["foobar"] = floatsource
            il = collada.source.InputList()
            il.addInput(0, "VERTEX", "#foobar")
            indices = numpy.array(s.triangles).flatten()
            new_prim = mesh.geometries[0].createTriangleSet(indices, il, "somematerial")
            mesh.geometries[0].primitives[0] = new_prim
            return mesh
    return ColladaSimplifyFilter()
