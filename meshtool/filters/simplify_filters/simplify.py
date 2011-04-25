from meshtool.args import *
from meshtool.filters.base_filters import *
import os
import random
import numpy
from .mesh_simplification import MeshSimplification
from .progress_printer import ProgressPrinter
import collada
import pickle
from .util import geomReplacePrimitive

def getSourceIdOrCreate(geom, source, new_id, components):
    for id in geom.sourceById:
        if numpy.array_equal(geom.sourceById[id], source):
            return id
    src = collada.source.FloatSource(new_id, source, components)
    geom.sourceById[new_id] = src
    return new_id

def FilterGenerator():
    class ColladaSimplifyFilter(OpFilter):
        def __init__(self):
            super(ColladaSimplifyFilter, self).__init__(
                'simplify', 'Simplifies a collada file')
            self.arguments.append(FilterArgument(
                    'percent', 'Percentage of vertices to simplify to (0-100)'))
            self.arguments.append(FileArgument(
                    'filename', 'Where to save the progressive mesh'))
        def apply(self, mesh, percent, filename):
            geom = mesh.geometries[0]
            prim = geom.primitives[0]
            if not isinstance(prim, collada.triangleset.TriangleSet):
                raise FilterException("Must triangulate before simplifying")
            percent = float(percent)
            if percent <= 0 or percent >= 100:
                raise FilterException(
                    "Percent must be between 0 and 100, exclusive: " + 
                    str(percent))
            print "Initializing..."
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
            s = MeshSimplification(prim.vertex, prim.vertex_index,
                                   corner_attributes, attribute_sources)
            num_contract = int(len(prim.vertex)*(100-percent)/100)
            print "Simplifying..."
            progress = ProgressPrinter(num_contract)
            for i in range(num_contract):
                progress.step()
                s.contractOnce()
                # if not s.isValid(): raise Exception
            print "Creating new primitive..."
            geomReplacePrimitive(geom, 0, s.vertices, s.triangles, s.attributes,
                                 s.attribute_sources)
            print "Generating progressive mesh..."
            pm = s.generatePM()
            pickle.dump(pm, open(filename, "wb"))
            print "Done."
            return mesh
    return ColladaSimplifyFilter()
