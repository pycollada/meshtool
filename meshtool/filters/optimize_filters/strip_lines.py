from meshtool.args import *
from meshtool.filters.base_filters import *
import collada

def stripLines(mesh):
    for geom in mesh.geometries:
        geom.primitives = [prim for prim in geom.primitives
                           if type(prim) is not collada.lineset.LineSet]

def FilterGenerator():
    class StripLinesFilter(OpFilter):
        def __init__(self):
            super(StripLinesFilter, self).__init__('strip_lines', 'Strips any lines from the document')
        def apply(self, mesh):
            stripLines(mesh)
            return mesh
    return StripLinesFilter()
