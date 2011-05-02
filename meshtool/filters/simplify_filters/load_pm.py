from meshtool.args import *
from meshtool.filters.base_filters import *
import pickle
import copy
import numpy
from .util import geomReplacePrimitive
from pprint import pprint
from .progress_printer import ProgressPrinter

def FilterGenerator():
    class ColladaLoadPmFilter(OpFilter):
        def __init__(self):
            super(ColladaLoadPmFilter, self).__init__(
                'load_pm', 'Loads percent percentage of the extra vertex information ' + 
                'from filename (saved with --simplify) back into the loaded mesh.')
            self.arguments.append(FilterArgument(
                    'percent', 'Percentage of PM to restore'))
            self.arguments.append(FileArgument(
                    'filename', 'PM file to load'))
        def apply(self, mesh, percent, filename):
            percent = float(percent)
            if percent < 0 or percent > 100:
                raise FilterException('Percent must be between 0 and 100.')
            print "Reading PM file..."
            pm = pickle.load(open(filename, "rb"))
            print "Initializing..."
            prims = {}
            for i in range(len(mesh.geometries)):
                geom = mesh.geometries[i]
                for j in range(len(geom.primitives)):
                    prim = geom.primitives[j]
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
                    prims[(i,j)] = (vertices, triangles, attributes,
                                    attribute_sources)

            print "Restoring mesh..."
            num_restored = int(len(pm)*percent/100.0)
            progress = ProgressPrinter(num_restored)
            for k in range(num_restored):
                progress.step()
                rec = pm[k]
                key, split_index, coords, changed_triangles, \
                    new_triangles_opp_v, new_triangles_flip, \
                    new_triangles_attr = rec
                vertices, triangles, attributes, attribute_sources = prims[key]
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
            for (g_i, p_i) in prims:
                vertices, triangles, attributes, attribute_sources = \
                    prims[(g_i, p_i)]
                geom = mesh.geometries[g_i]
                geomReplacePrimitive(geom, p_i, vertices, triangles,
                                     attributes, attribute_sources)
            return mesh
    return ColladaLoadPmFilter()
