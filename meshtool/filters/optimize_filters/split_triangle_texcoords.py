from meshtool.filters.base_filters import OptimizationFilter
import collada
import numpy
from meshtool.filters.atlas_filters.make_atlases import getTexcoordToImgMapping, TexcoordSet, MAX_TILING_DIMENSION

#The maximum number of trianlges is the maximum of these two values
MAX_TRIANGLE_INCREASE = 500
MAX_TRIANGLE_MULTIPLIER = 1.10

texdata = None
vertdata = None

def point_mean(arr1, arr2):
    return (arr1 + arr2) / 2.0

def point_dist_d2(arr, p1, p2):
    """Returns an array of the distance between two points in an array of 2d points"""
    return numpy.sqrt(numpy.square(arr[:,p1,0] - arr[:,p2,0]) + numpy.square(arr[:,p1,1] - arr[:,p2,1]))

def splitTriangleTexcoords(mesh):
    global texdata, vertdata

    #gets a mapping between texture coordinate set and the image paths it references
    tex2img = getTexcoordToImgMapping(mesh)
    
    # get a mapping from path to actual image
    unique_images = {}
    for cimg in mesh.images:
        if cimg.path not in unique_images:
            unique_images[cimg.path] = cimg.pilimage
    
    for geom in mesh.geometries:
        
        prims_to_delete = []
        prims_to_add = []
        
        for prim_index, prim in enumerate(geom.primitives):
            #only consider triangles that have texcoords
            if type(prim) is not collada.triangleset.TriangleSet or len(prim.texcoordset) < 1:
                continue
            
            #only using texcoord set 0 for now
            texdata = numpy.copy(prim.texcoordset[0])
            vertdata = numpy.copy(prim.vertex)
            texarray = texdata[prim.texcoord_indexset[0]]
             
            #we only want texcoords that go from 0 to N
            if numpy.min(texarray) <= 0.0 or numpy.max(texarray) <= 2.0:
                continue
            
            texset = TexcoordSet(geom.id, prim_index, None, 0)
            #if the texset is not in the mapping, it means it never references an image
            if texset not in tex2img:
                continue
            
            #first find the vertex and texcoord indices
            oldsources = prim.getInputList().getList()
            for (offset, semantic, source, setid) in oldsources:
                if semantic == 'TEXCOORD' and (setid is None or int(setid) == 0):
                    texindex = offset
                elif semantic == 'VERTEX':
                    vertindex = offset
            
            #selector to find triangles that have texcoords already in range
            tris2keep_idx = numpy.apply_along_axis(numpy.sum, 1, numpy.apply_along_axis(numpy.sum, 2, texarray > 2.0)) == 0
            
            #create two new index arrays and copy the vert and uv indexes there so we can modify
            new_indexes = numpy.zeros(( len(prim.index), 3, 2 ), prim.index.dtype)
            num_indexes = prim.index.shape[2]
            orig_index = numpy.dstack((prim.index, new_indexes))
            orig_index[:,:,num_indexes] = orig_index[:,:,vertindex]
            vertindex = num_indexes
            orig_index[:,:,num_indexes+1] = orig_index[:,:,texindex]
            texindex = num_indexes+1
            
            #build an index for the new triangles, starting with the previous ones that don't need splitting
            new_index = orig_index[tris2keep_idx]
            #array storing index to split
            index2split = orig_index[tris2keep_idx == False]
            
            giveup = False
            atlasable = False
            while len(index2split) > 0 and not giveup:

                pt0 = index2split[:,0,:]
                pt1 = index2split[:,1,:]
                pt2 = index2split[:,2,:]
                
                def halfway_between(pt1, pt2):
                    global texdata, vertdata
                    
                    #just copy the index from one of the other points and we will fill in the new vertex and uv indices
                    halfway_pt = numpy.copy(pt1)
                    
                    #calculate halfway in texture coordinate space
                    texpt1 = texdata[pt1[:,texindex]]
                    texpt2 = texdata[pt2[:,texindex]]
                    halfway_pt_texdata = point_mean(texpt1, texpt2)
                    halfway_tex_idx = numpy.arange(len(texdata), len(texdata) + len(halfway_pt_texdata))
                    halfway_pt[:,texindex] = halfway_tex_idx
                    texdata = numpy.concatenate((texdata, halfway_pt_texdata))
                    
                    #calculate halfway in vertex coordinate space
                    vertpt1 = vertdata[pt1[:,vertindex]]
                    vertpt2 = vertdata[pt2[:,vertindex]]
                    halfway_pt_vertdata = point_mean(vertpt1, vertpt2)
                    halfway_vert_idx = numpy.arange(len(vertdata), len(vertdata) + len(halfway_pt_vertdata))
                    halfway_pt[:,vertindex] = halfway_vert_idx
                    vertdata = numpy.concatenate((vertdata, halfway_pt_vertdata))
                    
                    return halfway_pt
                    
                halfway_pt0_pt1 = halfway_between(pt0, pt1)
                halfway_pt0_pt2 = halfway_between(pt0, pt2)
                halfway_pt1_pt2 = halfway_between(pt1, pt2)
                
                #now we have 6 points, the original 3 plus the points halfway between each of the points
                # so we can now construct four triangles, splitting the original triangle into 4 pieces
                
                tris1 = numpy.dstack((pt0, halfway_pt0_pt1, halfway_pt0_pt2))
                tris1 = numpy.swapaxes(tris1, 1, 2)
                
                tris2 = numpy.dstack((pt1, halfway_pt1_pt2, halfway_pt0_pt1))
                tris2 = numpy.swapaxes(tris2, 1, 2)
                
                tris3 = numpy.dstack((pt2, halfway_pt0_pt2, halfway_pt1_pt2))
                tris3 = numpy.swapaxes(tris3, 1, 2)
                
                tris4 = numpy.dstack((halfway_pt0_pt1, halfway_pt1_pt2, halfway_pt0_pt2))
                tris4 = numpy.swapaxes(tris4, 1, 2)
                
                #this is all of the index data now - the index that we didnt have to split plus the resulting split indices
                orig_index = numpy.concatenate((new_index, tris1, tris2, tris3, tris4))
                
                #recalculate the texcoord array
                texarray = texdata[orig_index[:,:,texindex]]
                
                #we now need to readjust the texcoord array to it's as close to 0 as possible
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
                
                texdata = numpy.copy(texarray)
                texdata.shape = (len(texarray)*3, 2)
                normalized_tex_index = numpy.arange(len(texdata))
                normalized_tex_index.shape = (len(orig_index), 3)
                orig_index[:,:,texindex] = normalized_tex_index
                
                #new selector to find triangles that have texcoords in range
                tris2keep_idx = numpy.apply_along_axis(numpy.sum, 1, numpy.apply_along_axis(numpy.sum, 2, texarray > 2.0)) == 0
                
                #triangles we need to split again
                index2split = orig_index[tris2keep_idx == False]
                #triangles that are done
                new_index = orig_index[tris2keep_idx]
                
                if len(orig_index)-len(prim.index) > max(MAX_TRIANGLE_INCREASE, len(prim.index) * MAX_TRIANGLE_MULTIPLIER):
                    
                    if len(tex2img[texset]) == 1:
                        width, height = unique_images[tex2img[texset][0]].size
                        tile_x = int(numpy.ceil(numpy.max(texarray[:,0])))
                        tile_y = int(numpy.ceil(numpy.max(texarray[:,1])))
                        stretched_width = tile_x * width
                        stretched_height = tile_y * height
                        if stretched_width <= MAX_TILING_DIMENSION and stretched_height <= MAX_TILING_DIMENSION:
                            giveup = True
                            atlasable = True
                        else:
                            giveup = True
                    else:
                        giveup = True
        
            if not giveup or atlasable:
                #rebuild the input list, changing the sources
                old_input_list = prim.getInputList().getList()
                inpl = collada.source.InputList()

                for offset, semantic, srcid, setid in old_input_list:
                    if semantic == 'VERTEX':
                        base_source_name = srcid[1:] + '-trisplit'
                        source_name = base_source_name
                        ct = 0
                        while source_name in geom.sourceById:
                            source_name = '%s-%d' % (base_source_name, ct)
                            ct += 1
                        vertdata.shape = -1
                        new_vert_src = collada.source.FloatSource(source_name, vertdata, ('X','Y','Z'))
                        geom.sourceById[source_name] = new_vert_src
                        vertdata = None
                        srcid = '#%s' % source_name
                        offset = vertindex
                    elif semantic == 'TEXCOORD' and (setid is None or int(setid) == 0):
                        base_source_name = srcid[1:] + '-trisplit'
                        source_name = base_source_name
                        ct = 0
                        while source_name in geom.sourceById:
                            source_name = '%s-%d' % (base_source_name, ct)
                            ct += 1
                        texdata.shape = -1
                        new_tex_src = collada.source.FloatSource(source_name, texdata, ('S', 'T'))
                        geom.sourceById[source_name] = new_tex_src
                        texdata = None
                        srcid = '#%s' % source_name
                        offset = texindex
                    inpl.addInput(offset, semantic, srcid, setid)
                
                orig_index.shape = -1
                prims_to_add.append((orig_index, inpl, prim.material))
                prims_to_delete.append(prim_index)
                
        #delete old ones and add new ones
        for i in sorted(prims_to_delete, reverse=True):
            del geom.primitives[i]
        for new_index, inpl, mat in prims_to_add:
            newtriset = geom.createTriangleSet(new_index, inpl, mat)
            geom.primitives.append(newtriset)

            
def FilterGenerator():
    class SplitTriangleTexcoordsFilter(OptimizationFilter):
        def __init__(self):
            super(SplitTriangleTexcoordsFilter, self).__init__('split_triangle_texcoords', "Splits triangles that span multiple texcoords into multiple triangles to better help texture atlasing")
        def apply(self, mesh):
            splitTriangleTexcoords(mesh)
            return mesh
    return SplitTriangleTexcoordsFilter()
from meshtool.filters import factory
factory.register(FilterGenerator().name, FilterGenerator)