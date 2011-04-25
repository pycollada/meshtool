from meshtool.args import *
from meshtool.filters.base_filters import *
import pickle
import copy
import numpy
from .util import geomReplacePrimitive
from pprint import pprint

def FilterGenerator():
    class ColladaLoadPmFilter(OpFilter):
        def __init__(self):
            super(ColladaLoadPmFilter, self).__init__(
                'load_pm', 'Loads a PM')
            self.arguments.append(FilterArgument(
                    'percent', 'Percentage of PM to restore'))
            self.arguments.append(FileArgument(
                    'filename', 'PM file to load'))
        def apply(self, mesh, percent, filename):
            pm = pickle.load(open(filename, "rb"))
            geom = mesh.geometries[0]
            prim = geom.primitives[0]
            vertices = [copy.copy(v) for v in prim.vertex]
            triangles = [copy.copy(tri) for tri in prim.vertex_index]
            if prim.texcoordset is not None:
                attributes = [[copy.copy(tri) for tri in indset]
                              for indset in prim.texcoord_indexset]
                attribute_sources = [[copy.copy(a) for a in sl]
                                     for sl in prim.texcoordset]
            else:
                attributes = []
                attribute_sources = []
            if prim.normal is not None:
                attributes.append([copy.copy(tri)
                                   for tri in prim.normal_index])
                attribute_sources.append([copy.copy(n)
                                          for n in prim.normal])
            percent = float(percent)
            if percent < 0 or percent > 100:
                raise FilterException('Percent must be between 0 and 100.')
            num_restored = int(len(pm)*percent/100.0)
            for k in range(num_restored):
                rec = pm[k]
                split_index, coords, changed_triangles, \
                    new_triangles_opp_v, new_triangles_flip, \
                    new_triangles_attr = rec
                new_index = len(vertices)
                vertices.append(coords)
                for tri_i in changed_triangles:
                    for i in (0,1,2):
                        if triangles[tri_i][i] == split_index:
                            triangles[tri_i][i] = new_index
                for i in range(len(new_triangles_opp_v)):
                    opp_index = new_triangles_opp_v[i]
                    flip = new_triangles_flip[i]
                    tri = [new_index, split_index, opp_index]
                    if flip:
                        tri[0], tri[1] = tri[1], tri[0]
                    triangles.append(numpy.array(tri))
                for i in range(len(new_triangles_attr)):
                    for attr in new_triangles_attr[i]:
                        new_attr = []
                        for a in attr:
                            if type(a) == int:
                                new_attr.append(a)
                            else:
                                new_index = len(attribute_sources[i])
                                new_attr.append(new_index)
                                attribute_sources[i].append(a)
                        attributes[i].append(new_attr)
            geomReplacePrimitive(geom, 0, vertices, triangles,
                                 attributes, attribute_sources)
            return mesh
    return ColladaLoadPmFilter()
