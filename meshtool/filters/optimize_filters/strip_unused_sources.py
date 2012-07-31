from meshtool.filters.base_filters import OptimizationFilter
import collada

def stripUnusedSources(mesh):
    for geom in mesh.geometries:
        srcid_count = {}
        for srcid, src in geom.sourceById.iteritems():
            if isinstance(src, collada.source.Source):
                srcid_count[srcid] = 0
        for prim in geom.primitives:
            for semantic, inputs in prim.sources.iteritems():
                for offset, semantic, srcid, setid, src in inputs:
                    srcid_count[srcid[1:]] += 1
        for srcid, count in srcid_count.iteritems():
            if count == 0:
                del geom.sourceById[srcid]

def FilterGenerator():
    class StripUnusedSourcesFilter(OptimizationFilter):
        def __init__(self):
            super(StripUnusedSourcesFilter, self).__init__('strip_unused_sources', "Strips any source arrays from geometries if they aren't referenced by any primitives")
        def apply(self, mesh):
            stripUnusedSources(mesh)
            return mesh
    return StripUnusedSourcesFilter()
from meshtool.filters import factory
factory.register(FilterGenerator().name, FilterGenerator)