from meshtool.filters.base_filters import OptimizationFilter
import collada
import numpy

def normalizeIndices(mesh):
    
    for geom in mesh.geometries:

        prims_to_delete = []
        prims_to_add = []
        
        if len(geom.primitives) > 1:
            base_source_name = geom.id + '-vertex'
            vert_source_name = base_source_name
            ct = 0
            while vert_source_name in geom.sourceById:
                vert_source_name = '%s-%d' % (base_source_name, ct)
                ct += 1
            geom.sourceById[vert_source_name] = 'reserved'
            vert_source_data = []
            cur_vert_len = 0
        
        for prim_index, prim in enumerate(geom.primitives):
            
            #only do triangle sets
            if type(prim) is not collada.triangleset.TriangleSet:
                continue
            
            #rebuild the input list, changing the sources
            old_input_list = prim.getInputList().getList()
            inpl = collada.source.InputList()
            new_index = numpy.arange(len(prim.index)*3)
            for offset, semantic, srcid, setid in old_input_list:
                old_source = geom.sourceById[srcid[1:]]
                new_source_data = numpy.copy(old_source.data[prim.index[:,:,offset]])
                new_source_data = new_source_data.flatten()
                
                if len(geom.primitives) == 1 or semantic != 'VERTEX':
                    base_source_name = srcid[1:] + '-normalized'
                    source_name = base_source_name
                    ct = 0
                    while source_name in geom.sourceById:
                        source_name = '%s-%d' % (base_source_name, ct)
                        ct += 1
                    
                    new_source = collada.source.FloatSource(source_name, new_source_data, old_source.components)
                    geom.sourceById[source_name] = new_source
                
                    srcid = '#%s' % source_name
                    
                if len(geom.primitives) > 1 and semantic == 'VERTEX':
                    offset = 0
                    vert_source_data.append(new_source_data)
                    vert_index = numpy.arange(len(prim.index)*3) + cur_vert_len
                    new_index.shape = (len(prim.index), 3)
                    vert_index.shape = (len(prim.index), 3)
                    new_index = numpy.dstack((vert_index, new_index))
                    new_index = new_index.flatten()
                    cur_vert_len += len(prim.index)*3
                    srcid = '#%s' % vert_source_name
                elif len(geom.primitives) > 1:
                    offset = 1
                else:
                    offset = 0
                inpl.addInput(offset, semantic, srcid, setid)
 
            prims_to_add.append((new_index, inpl, prim.material))
            
            prims_to_delete.append(prim_index)
            
        if len(geom.primitives) > 1 and len(vert_source_data) > 0:
            vert_source_data = numpy.concatenate(vert_source_data)
            new_vert_source = collada.source.FloatSource(vert_source_name, vert_source_data, ('X','Y','Z'))
            geom.sourceById[vert_source_name] = new_vert_source

        #delete old ones and add new ones
        for i in sorted(prims_to_delete, reverse=True):
            del geom.primitives[i]
        for new_index, inpl, mat in prims_to_add:
            newtriset = geom.createTriangleSet(new_index, inpl, mat)
            geom.primitives.append(newtriset)
            
def FilterGenerator():
    class NormalizeIndicesFilter(OptimizationFilter):
        def __init__(self):
            super(NormalizeIndicesFilter, self).__init__('normalize_indices', "Goes through all triangle sets, changing all index values to go from 1 to N, replacing sources to be size N")
        def apply(self, mesh):
            normalizeIndices(mesh)
            return mesh
    return NormalizeIndicesFilter()
from meshtool.filters import factory
factory.register(FilterGenerator().name, FilterGenerator)