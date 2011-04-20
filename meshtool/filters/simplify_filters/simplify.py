from meshtool.args import *
from meshtool.filters.base_filters import *
import os
import random
import numpy
from .mesh_simplification import MeshSimplification
from .progress_printer import ProgressPrinter
import collada

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
            super(ColladaSimplifyFilter, self).__init__('simplify', 'Simplifies a collada file')
            self.arguments.append(FileArgument('percent', 'Percentage of vertices to simplify to (0-100)'))
        def apply(self, mesh, percent):
            geom = mesh.geometries[0]
            prim = geom.primitives[0]
            if not isinstance(prim, collada.triangleset.TriangleSet):
                raise FilterException("Must triangulate before simplifying")
            percent = float(percent)
            if percent <= 0 or percent >= 100:
                raise FilterException("Percent must be between 0 and 100, exclusive: "+str(percent))
            print "Initializing..."
            if prim.texcoordset is not None:
                corner_attributes = [indset for indset in prim.texcoord_indexset]
            else:
                corner_attributes = []
            if prim.normal is not None:
                corner_attributes.append(prim.normal_index)
            s = MeshSimplification(prim.vertex, prim.vertex_index, corner_attributes)
            num_contract = int(len(prim.vertex)*(100-percent)/100)
            print "Simplifying..."
            progress = ProgressPrinter(num_contract)
            for i in range(num_contract):
                progress.step()
                s.contractOnce()
                # if not s.isValid(): raise Exception
            print "Creating new primitive..."
            il = collada.source.InputList()
            new_id = geom.id+"-vertex"
            while new_id in geom.sourceById:
                new_id += "-x"
            vertexsource = collada.source.FloatSource(new_id, numpy.array(s.vertices), ('X', 'Y', 'Z'))
            geom.sourceById[new_id] = vertexsource
            offset = 0
            il.addInput(offset, "VERTEX", "#"+new_id)
            offset += 1
            if prim.normal is not None:
                il.addInput(offset, "NORMAL", prim.sources['NORMAL'][0][2])
                offset += 1
            if prim.texcoordset is not None:
                texcoord_indexset = []
                last = len(s.corner_attributes) - 1
                for i in range(last):
                    il.addInput(offset, "TEXCOORD", prim.sources['TEXCOORD'][i][2])
                    offset += 1
                    texcoord_indexset.append(numpy.array(s.corner_attributes[i]))
            indices = numpy.array(s.triangles)
            indices.shape = (-1,3,1)
            if prim.normal is not None:
                normal_index = numpy.array(s.corner_attributes[last])
                normal_index.shape = (-1,3,1)
                indices = numpy.append(indices, normal_index, 2)
            if prim.texcoordset is not None:
                for texcoord_index in texcoord_indexset:
                    texcoord_index.shape = (-1,3,1)
                    indices = numpy.append(indices, texcoord_index, 2)
            new_prim = geom.createTriangleSet(indices.flatten(), il, prim.material)
            geom.primitives[0] = new_prim
            print "Done."
            return mesh
    return ColladaSimplifyFilter()
