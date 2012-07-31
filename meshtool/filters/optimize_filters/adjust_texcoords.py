from meshtool.filters.base_filters import OptimizationFilter
import collada
import numpy

def adjustTexcoords(mesh):
    
    for geom in mesh.geometries:
        
        prims_to_delete = []
        prims_to_add = []
        
        for prim_index, prim in enumerate(geom.primitives):
            #only consider triangles that have texcoords
            if type(prim) is not collada.triangleset.TriangleSet or len(prim.texcoordset) < 1:
                continue
            
            texarray = prim.texcoordset[0][prim.texcoord_indexset[0]]
            
            #only care about adjust texcoords that go outside the 0 to 1 range
            if numpy.min(texarray) >= 0.0 and numpy.max(texarray) <= 1.0:
                continue
            
            # Calculate the min x value and min y value for each triangle
            # then take the floor of the min and subtract that value
            # from each triangle. This makes each triangle's texcoords
            # as close to 0 as possible without changing their effect
            x1 = texarray[:,0,0]
            x2 = texarray[:,1,0]
            x3 = texarray[:,2,0]
            y1 = texarray[:,0,1]
            y2 = texarray[:,1,1]
            y3 = texarray[:,2,1]
            
            xmin = numpy.minimum(x1, numpy.minimum(x2, x3))
            ymin = numpy.minimum(y1, numpy.minimum(y2, y3))
            
            xfloor = numpy.floor(xmin)
            yfloor = numpy.floor(ymin)
            
            texarray[:,:,0] -= xfloor[:, numpy.newaxis]
            texarray[:,:,1] -= yfloor[:, numpy.newaxis]
            texarray = texarray.flatten()
            
            #now rebuild the input list, but just changing the texcoord source
            old_input_list = prim.getInputList().getList()
            inpl = collada.source.InputList()
            new_index = numpy.copy(prim.index)
            for offset, semantic, srcid, setid in old_input_list:
                if semantic == 'TEXCOORD' and (setid == '0' or len(prim.texcoordset) == 1):
                    base_source_name = srcid[1:] + '-adjusted'
                    source_name = base_source_name
                    ct = 0
                    while source_name in geom.sourceById:
                        source_name = '%s-%d' % (base_source_name, ct)
                        ct += 1
                    
                    new_tex_src = collada.source.FloatSource(source_name, texarray, ('S', 'T'))
                    geom.sourceById[source_name] = new_tex_src
                    
                    new_tex_index = numpy.arange(len(new_index)*3)
                    new_tex_index.shape = (len(new_index), 3)
                    new_index[:,:,offset] = new_tex_index
                    
                    srcid = '#%s' % source_name
                    
                inpl.addInput(offset, semantic, srcid, setid)

            newtriset = geom.createTriangleSet(new_index, inpl, prim.material)
            prims_to_add.append(newtriset)
            
            prims_to_delete.append(prim_index)

        #delete old ones and add new ones
        for i in sorted(prims_to_delete, reverse=True):
            del geom.primitives[i]
        for prim in prims_to_add:
            geom.primitives.append(prim)
            
def FilterGenerator():
    class AdjustTexcoordsFilter(OptimizationFilter):
        def __init__(self):
            super(AdjustTexcoordsFilter, self).__init__('adjust_texcoords', "Adjusts texture coordinates of triangles so that they are as close to the 0-1 range as possible")
        def apply(self, mesh):
            adjustTexcoords(mesh)
            return mesh
    return AdjustTexcoordsFilter()
from meshtool.filters import factory
factory.register(FilterGenerator().name, FilterGenerator)