from meshtool.filters.base_filters import OptimizationFilter
import collada
import numpy
import inspect

#after numpy 1.3, unique1d was renamed to unique
args, varargs, keywords, defaults = inspect.getargspec(numpy.unique)    
if 'return_inverse' not in args:
    numpy.unique = numpy.unique1d

def optimizeSources(mesh):
    
    for geom in mesh.geometries:
        
        src_by_semantic = {}
        
        for prim_index, prim in enumerate(geom.primitives):
            
            #only do triangle sets
            if type(prim) is not collada.triangleset.TriangleSet:
                continue
            
            #rebuild the input list, changing the sources
            inpl = prim.getInputList().getList()
            for offset, semantic, srcid, setid in inpl:
                src = geom.sourceById[srcid[1:]]
                semantic_sources = src_by_semantic.get(semantic, {})
                semantic_sources[src.id] = src
                src_by_semantic[semantic] = semantic_sources
        
        for semantic, sources in src_by_semantic.iteritems():            
            source_placements = {}
            
            to_concat = []
            cur_index = 0
            for src in sources.itervalues():
                srcid = src.id
                components = src.components
                to_concat.append(src.data)
                source_placements[src.id] = cur_index
                cur_index += len(src.data)
            new_data = numpy.concatenate(to_concat)
            
            #makes the array unique and returns index locations for previous data
            unique_data, index_locs = numpy.unique( new_data.view([('',new_data.dtype)]*new_data.shape[1]), return_inverse=True)
            unique_data = unique_data.view(new_data.dtype).reshape(-1,new_data.shape[1])
            
            base_source_name = srcid + '-unique'
            source_name = base_source_name
            ct = 0
            while source_name in geom.sourceById:
                source_name = '%s-%d' % (base_source_name, ct)
                ct += 1
            new_source = collada.source.FloatSource(source_name, unique_data, components)
            geom.sourceById[source_name] = new_source
            
            src_by_semantic[semantic] = (source_placements, source_name, unique_data, index_locs)

        prims_to_delete = []
        prims_to_add = []

        for prim_index, prim in enumerate(geom.primitives):
            
            #only do triangle sets
            if type(prim) is not collada.triangleset.TriangleSet:
                continue
            
            #rebuild the input list, changing the sources
            old_input_list = prim.getInputList().getList()
            inpl = collada.source.InputList()
            new_index = numpy.arange(len(old_input_list)*len(prim.index)*3)
            new_index.shape = (len(prim.index), 3, len(old_input_list))
            new_offset = 0
            for offset, semantic, srcid, setid in old_input_list:
                (source_placements, source_name, unique_data, index_locs) = src_by_semantic[semantic]
                map_start = source_placements[srcid[1:]]
                map_end = map_start + len(geom.sourceById[srcid[1:]])
                new_index[:,:,new_offset] = index_locs[map_start:map_end][prim.index[:,:,offset]]
                offset = new_offset
                new_offset += 1
                srcid = "#" + source_name
                inpl.addInput(offset, semantic, srcid, setid)
                
            prims_to_add.append((new_index, inpl, prim.material))
            prims_to_delete.append(prim_index)
            
        #delete old ones and add new ones
        for i in sorted(prims_to_delete, reverse=True):
            del geom.primitives[i]
        for new_index, inpl, mat in prims_to_add:
            newtriset = geom.createTriangleSet(new_index, inpl, mat)
            geom.primitives.append(newtriset)
            
def FilterGenerator():
    class OptimizeSourcesFilter(OptimizationFilter):
        def __init__(self):
            super(OptimizeSourcesFilter, self).__init__('optimize_sources', "Compresses sources to unique values, updating triangleset indices")
        def apply(self, mesh):
            optimizeSources(mesh)
            return mesh
    return OptimizeSourcesFilter()
from meshtool.filters import factory
factory.register(FilterGenerator().name, FilterGenerator)