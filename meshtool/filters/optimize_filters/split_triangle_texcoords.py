from meshtool.args import *
from meshtool.filters.base_filters import *
import collada
import numpy

def point_mean(arr1, arr2):
    return (arr1 + arr2) / 2.0

def point_dist_d2(arr, p1, p2):
    """Returns an array of the distance between two points in an array of 2d points"""
    return numpy.sqrt(numpy.square(arr[:,p1,0] - arr[:,p2,0]) + numpy.square(arr[:,p1,1] - arr[:,p2,1]))

def splitTriangleTexcoords(mesh):
    
    notexcoords = 0
    alreadyin = 0
    succeeded = 0
    gaveup = 0
    
    for geom in mesh.geometries:
        
        for prim_index, prim in enumerate(geom.primitives):
            #only consider triangles that have texcoords
            if type(prim) is not collada.triangleset.TriangleSet or len(prim.texcoordset) < 1:
                notexcoords += 1
                continue
            
            #only using texcoord set 0 for now
            texdata = numpy.copy(prim.texcoordset[0])
            vertdata = numpy.copy(prim.vertex)
            texarray = texdata[prim.texcoord_indexset[0]]
             
            #we only want texcoords that go from 0 to N
            if numpy.min(texarray) <= 0.0 or numpy.max(texarray) <= 2.0:
                alreadyin += 1
                continue
            
            #first find the vertex and texcoord indices
            oldsources = prim.getInputList().getList()
            for (offset, semantic, source, set) in oldsources:
                if semantic == 'TEXCOORD' and (set is None or int(set) == 0):
                    texindex = offset
                elif semantic == 'VERTEX':
                    vertindex = offset
            
            #selector to find triangles that have texcoords already in range
            tris2keep_idx = numpy.apply_along_axis(numpy.sum, 1, numpy.apply_along_axis(numpy.sum, 2, texarray > 2.0)) == 0
            
            #array that will store the current set of all index data
            orig_index = numpy.copy(prim.index)
            #build an index for the new triangles, starting with the previous ones that don't need splitting
            new_index = orig_index[tris2keep_idx]
            #array storing index to split
            index2split = orig_index[tris2keep_idx == False]
            
            print
            print 'starting', len(prim.index), numpy.max(texarray)
            
            giveup = False
            while len(index2split) > 0 and not giveup:
                                
                texarray = texdata[index2split[:,:,texindex]]
                
                #distance between points X and Y
                distp1p2 = point_dist_d2(texarray, 0, 1)
                distp1p3 = point_dist_d2(texarray, 0, 2)
                distp2p3 = point_dist_d2(texarray, 1, 2)
                
                
                #get the point across from the longest edge, and the remaining 2 points
                
                across_long_pt = numpy.where((distp1p2 > distp1p3)[:,numpy.newaxis],
                                             numpy.where((distp2p3 > distp1p2)[:,numpy.newaxis], index2split[:,0,:], index2split[:,2,:]),
                                             numpy.where((distp2p3 > distp1p3)[:,numpy.newaxis], index2split[:,0,:], index2split[:,1,:]))
                
                other_pt1 = numpy.where((distp1p2 > distp1p3)[:,numpy.newaxis],
                                             numpy.where((distp2p3 > distp1p2)[:,numpy.newaxis], index2split[:,1,:], index2split[:,0,:]),
                                             numpy.where((distp2p3 > distp1p3)[:,numpy.newaxis], index2split[:,1,:], index2split[:,2,:]))
                
                other_pt2 = numpy.where((distp1p2 > distp1p3)[:,numpy.newaxis],
                                             numpy.where((distp2p3 > distp1p2)[:,numpy.newaxis], index2split[:,2,:], index2split[:,1,:]),
                                             numpy.where((distp2p3 > distp1p3)[:,numpy.newaxis], index2split[:,2,:], index2split[:,0,:]))
                
                #next we have to calculate the point half way between the two points adjacent to the longest edge
                
                #just copy the index from one of the other points and we will fill in the new vertex and uv indices
                halfway_pt = numpy.copy(other_pt1)
                
                #calculate halfway in texture coordinate space
                texpt1 = texdata[other_pt1[:,texindex]]
                texpt2 = texdata[other_pt2[:,texindex]]
                halfway_pt_texdata = point_mean(texpt1, texpt2)
                halfway_tex_idx = numpy.arange(len(texdata), len(texdata) + len(halfway_pt_texdata))
                halfway_pt[:,texindex] = halfway_tex_idx
                texdata = numpy.concatenate((texdata, halfway_pt_texdata))
                
                #calculate halfway in vertex coordinate space
                vertpt1 = vertdata[other_pt1[:,vertindex]]
                vertpt2 = vertdata[other_pt2[:,vertindex]]
                halfway_pt_vertdata = point_mean(vertpt1, vertpt2)
                halfway_vert_idx = numpy.arange(len(vertdata), len(vertdata) + len(halfway_pt_vertdata))
                halfway_pt[:,vertindex] = halfway_vert_idx
                vertdata = numpy.concatenate((vertdata, halfway_pt_vertdata))
                
                #now we have 4 points, the original 3 plus the point halfway between the 2 points on the longer edge
                # so we can now construct two triangles, splitting the original triangle across its longest edge
                
                tris1 = numpy.dstack((across_long_pt, other_pt1, halfway_pt))
                tris1 = numpy.swapaxes(tris1, 1, 2)
                
                tris2 = numpy.dstack((across_long_pt, other_pt2, halfway_pt))
                tris2 = numpy.swapaxes(tris2, 1, 2)
                
                #this is all of the index data now - the index that we didnt have to split plus the resulting split indices
                #print 'old keep data', len(new_index)
                #print 'newtri1', len(tris1)
                #print 'newtri2', len(tris2)
                orig_index = numpy.concatenate((new_index, tris1, tris2))
                #print 'new all', len(orig_index)
                
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
                
                if len(orig_index)-len(prim.index) > max(1000, len(prim.index) * 2):
                    print 'FAILED stopping because', len(orig_index), 'is bigger than 1000 or', len(prim.index)*2, numpy.max(texarray)
                    giveup = True
                    gaveup += 1
                #import sys
                #sys.exit(0)
        
            if not giveup:
                print 'SUCCESS', len(prim.index), '->', len(orig_index)
                succeeded += 1
        
             
    print 'no tex coords:', notexcoords
    print 'already in range (0,2):', alreadyin
    print 'succeeded with limit:', succeeded
    print 'gave up because hit limit:', gaveup   
    import sys
    sys.exit(0)

            
def FilterGenerator():
    class SplitTriangleTexcoordsFilter(OpFilter):
        def __init__(self):
            super(SplitTriangleTexcoordsFilter, self).__init__('split_triangle_texcoords', "Splits triangles that span multiple texcoords into multiple triangles to better help texture atlasing")
        def apply(self, mesh):
            splitTriangleTexcoords(mesh)
            return mesh
    return SplitTriangleTexcoordsFilter()
