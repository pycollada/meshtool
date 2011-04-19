from meshtool.args import *
from meshtool.filters.base_filters import *
import os
import random
import numpy
from .mesh_simplification import MeshSimplification
from .progress_printer import ProgressPrinter
import collada

def FilterGenerator():
    class ColladaSimplifyFilter(OpFilter):
        def __init__(self):
            super(ColladaSimplifyFilter, self).__init__('simplify', 'Simplifies a collada file')
            self.arguments.append(FileArgument('percent', 'Percentage of vertices to simplify to (0-100)'))
        def apply(self, mesh, percent):
            prim = mesh.geometries[0].primitives[0]
            if not isinstance(prim, collada.triangleset.TriangleSet):
                raise FilterException("Must triangulate before simplifying")
            percent = float(percent)
            if percent <= 0 or percent >= 100:
                raise FilterException("Percent must be between 0 and 100, exclusive: "+str(percent))
            print "Initializing..."
            s = MeshSimplification(prim.vertex, prim.vertex_index)
            num_contract = int(len(prim.vertex)*(100-percent)/100)
            print "Simplifying..."
            progress = ProgressPrinter(num_contract)
            for i in range(num_contract):
                progress.step()
                s.contractOnce()
                # if not s.isValid(): raise Exception
            print "Creating new primitive..."
            floatsource = collada.source.FloatSource("foobar", numpy.array(s.vertices), ('X', 'Y', 'Z'))
            mesh.geometries[0].sourceById["foobar"] = floatsource
            il = collada.source.InputList()
            il.addInput(0, "VERTEX", "#foobar")
            indices = numpy.array(s.triangles).flatten()
            new_prim = mesh.geometries[0].createTriangleSet(indices, il, "somematerial")
            mesh.geometries[0].primitives[0] = new_prim
            print "Done."
            return mesh
    return ColladaSimplifyFilter()
