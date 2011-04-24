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
            super(ColladaSimplifyFilter, self).__init__(
                'simplify', 'Simplifies a collada file')
            self.arguments.append(FilterArgument(
                    'percent', 'Percentage of vertices to simplify to (0-100)'))
        def apply(self, mesh, percent):
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
            il = collada.source.InputList()
            new_id = geom.id+"-vertex"
            while new_id in geom.sourceById:
                new_id += "-x"
            vertexsource = collada.source.FloatSource(
                new_id, numpy.array(s.vertices), ('X', 'Y', 'Z'))
            geom.sourceById[new_id] = vertexsource
            offset = 0
            il.addInput(offset, "VERTEX", "#"+new_id)
            offset += 1
            last = len(s.attributes) - 1
            if prim.normal is not None:
                new_id = geom.id+"-normal"
                while new_id in geom.sourceById:
                    new_id += "-x"
                normalsource = collada.source.FloatSource(
                    new_id, numpy.array(s.attribute_sources[last]),
                    ('X', 'Y', 'Z'))
                geom.sourceById[new_id] = normalsource
                il.addInput(offset, "NORMAL", "#"+new_id)
                offset += 1
            if prim.texcoordset is not None:
                texcoord_indexset = []
                for i in range(last):
                    new_id = geom.id+"-texcoord"+str(i)
                    while new_id in geom.sourceById:
                        new_id += "-x"
                    texcoordsource = collada.source.FloatSource(
                        new_id, numpy.array(s.attribute_sources[i]),
                        ('S', 'T'))
                    geom.sourceById[new_id] = texcoordsource
                    il.addInput(offset, "TEXCOORD", "#"+new_id)
                    offset += 1
                    texcoord_indexset.append(numpy.array(s.attributes[i]))
            indices = numpy.array(s.triangles)
            indices.shape = (-1,3,1)
            if prim.normal is not None:
                normal_index = numpy.array(s.attributes[last])
                normal_index.shape = (-1,3,1)
                indices = numpy.append(indices, normal_index, 2)
            if prim.texcoordset is not None:
                for texcoord_index in texcoord_indexset:
                    texcoord_index.shape = (-1,3,1)
                    indices = numpy.append(indices, texcoord_index, 2)
            new_prim = geom.createTriangleSet(indices.flatten(), il, prim.material)
            geom.primitives[0] = new_prim

            # Clean up unused sources
            referenced_sources = {}
            for prim in geom.primitives:
                for semantic in prim.sources:
                    for input in prim.sources[semantic]:
                        referenced_sources[input[2][1:]] = True
            unreferenced_sources = []
            for id in geom.sourceById:
                if id not in referenced_sources:
                    unreferenced_sources.append(id)
            for id in unreferenced_sources:
                del geom.sourceById[id]
            print "Done."
            return mesh
    return ColladaSimplifyFilter()
