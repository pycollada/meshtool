from meshtool.args import *
from meshtool.filters.base_filters import *
import collada

def triangulate(mesh):
    for geom in mesh.geometries:
        triprims = []
        for prim in geom.primitives:
            if isinstance(prim, collada.polylist.Polylist):
                triprims.append(prim.triangleset())
            else:
                triprims.append(prim)
        geom.primitives = triprims

def FilterGenerator():
    class TriangulateFilter(OpFilter):
        def __init__(self):
            super(TriangulateFilter, self).__init__('triangulate', 'Replaces any polylist or polygons with triangles')
        def apply(self, mesh):
            triangulate(mesh)
            return mesh
    return TriangulateFilter()
