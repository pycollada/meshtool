from meshtool.args import *
from meshtool.filters.base_filters import *
import os
import random
import numpy
from .aggregate_simplifier import AggregateSimplifier
from .progress_printer import ProgressPrinter
import collada
import pickle
from .util import geomReplacePrimitive

def FilterGenerator():
    class ColladaSimplifyFilter(OpFilter):
        def __init__(self):
            super(ColladaSimplifyFilter, self).__init__(
                'simplify', 'Uses mesh simplificaiton algorithm to remove vertex information ' + 
                'from the mesh. Removes percent percentage of the vertices and saves our ' +
                'the results into filename. The file can be loaded with --load_pm.')
            self.arguments.append(FilterArgument(
                    'percent', 'Percentage of vertices to simplify to (0-100)'))
            self.arguments.append(FileArgument(
                    'filename', 'Where to save the progressive mesh'))
        def apply(self, mesh, percent, filename):
            percent = float(percent)
            if percent <= 0 or percent >= 100:
                raise FilterException(
                    "Percent must be between 0 and 100, exclusive: " + 
                    str(percent))
            s = AggregateSimplifier()
            print "Initializing..."
            for g_i in range(len(mesh.geometries)):
                geom = mesh.geometries[g_i]
                for p_i in range(len(geom.primitives)):
                    prim = geom.primitives[p_i]
                    if not isinstance(prim, collada.triangleset.TriangleSet):
                        raise FilterException("Must triangulate before simplifying")
                    if prim.texcoordset is not None:
                        corner_attributes = [indset for indset
                                             in prim.texcoord_indexset]
                        attribute_sources = [sl for sl in prim.texcoordset]
                    else:
                        corner_attributes = []
                        attribute_sources = []
                    if prim.normal is not None:
                        corner_attributes.append(prim.normal_index)
                        attribute_sources.append(prim.normal)
                    print "geometries[%d].primitives[%d]"%(g_i, p_i)
                    s.addPrimitive((g_i, p_i), prim.vertex, prim.vertex_index,
                                   corner_attributes, attribute_sources)
            num_contract = int(s.num_vertices*(100-percent)/100)
            print "Simplifying..."
            progress = ProgressPrinter(num_contract)
            for i in range(num_contract):
                progress.step()
                s.contractOnce()
                # if not s.isValid(): raise Exception
            print "Creating new primitives..."
            for (g_i, p_i) in s.simplifiers:
                print "geometries[%d].primitives[%d]"%(g_i, p_i)
                ms = s.simplifiers[(g_i, p_i)]
                geomReplacePrimitive(mesh.geometries[g_i], p_i, ms.vertices,
                                     ms.triangles, ms.attributes,
                                     ms.attribute_sources)
            print "Generating progressive mesh..."
            pm = s.generatePM()
            pickle.dump(pm, open(filename, "wb"))
            print "Done."
            return mesh
    return ColladaSimplifyFilter()
