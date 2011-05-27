from meshtool.args import *
from meshtool.filters.base_filters import *
import collada
import numpy

def normalizeIndices(mesh):
    
    for geom in mesh.geometries:
        
        prims_to_delete = []
        prims_to_add = []
        
        for prim_index, prim in enumerate(geom.primitives):
            #only do triangle sets
            if type(prim) is not collada.triangleset.TriangleSet:
                continue
            
            #rebuild the input list, changing the sources
            old_input_list = prim.getInputList().getList()
            inpl = collada.source.InputList()
            new_index = numpy.arange(len(prim.index)*3)
            for offset, semantic, srcid, set in old_input_list:
                old_source = geom.sourceById[srcid[1:]]
                new_source_data = numpy.copy(old_source.data[prim.index[:,:,offset]])
                new_source_data = new_source_data.flatten()
                
                base_source_name = srcid[1:] + '-normalized'
                source_name = base_source_name
                ct = 0
                while source_name in geom.sourceById:
                    source_name = '%s-%d' % (base_source_name, ct)
                    ct += 1
                
                new_source = collada.source.FloatSource(source_name, new_source_data, old_source.components)
                geom.sourceById[source_name] = new_source
                
                srcid = '#%s' % source_name
                    
                #we can now set all the offsets to 0 and use the same index
                offset = 0
                inpl.addInput(offset, semantic, srcid, set)
 
            newtriset = geom.createTriangleSet(new_index, inpl, prim.material)
            prims_to_add.append(newtriset)
            
            prims_to_delete.append(prim_index)

        #delete old ones and add new ones
        for i in sorted(prims_to_delete, reverse=True):
            del geom.primitives[i]
        for prim in prims_to_add:
            geom.primitives.append(prim)
            
def FilterGenerator():
    class NormalizeIndicesFilter(OpFilter):
        def __init__(self):
            super(NormalizeIndicesFilter, self).__init__('normalize_indices', "Goes through all triangle sets, changing all index values to go from 1 to N, replacing sources to be size N")
        def apply(self, mesh):
            normalizeIndices(mesh)
            return mesh
    return NormalizeIndicesFilter()
