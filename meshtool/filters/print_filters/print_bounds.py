import itertools
import collada
import numpy
import math

from meshtool.filters import factory
from meshtool.filters.base_filters import PrintFilter

INF = float('inf')
NEGINF = float('-inf')

def v3dist(pt1, pt2):
    """Calculates the distance between two 3d points element-wise
    along an array"""
    d = pt1 - pt2
    return math.sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2])

def v3_array_pt_dist(arr, pt):
    """Returns an array of distances between pt and every point in arr"""
    return numpy.sqrt(numpy.square(arr[:,0] - pt[0]) +
                      numpy.square(arr[:,1] - pt[1]) +
                      numpy.square(arr[:,2] - pt[2]))

def centerFromBounds(bounds):
    minpt, maxpt = bounds
    x = ((maxpt[0] - minpt[0]) / 2) + minpt[0]
    y = ((maxpt[1] - minpt[1]) / 2) + minpt[1]
    z = ((maxpt[2] - minpt[2]) / 2) + minpt[2]
    return numpy.array([x,y,z], dtype=numpy.float32)

def iter_prims(mesh):
    for boundobj in itertools.chain(mesh.scene.objects('geometry'), mesh.scene.objects('controller')):
        if isinstance(boundobj, collada.geometry.BoundGeometry):
            boundgeom = boundobj
        else:
            boundgeom = boundobj.geometry
        
        for boundprim in boundgeom.primitives():
            if isinstance(boundprim, collada.polylist.BoundPolylist):
                boundprim = boundprim.triangleset()
                
            if boundprim.vertex is None:
                continue
            
            yield boundprim

def getBoundsInfo(mesh):
    """Returns a dictionary containing:
    
     * bounds: a tuple containing two (3,) numpy arrays that are the minimum and maximum point of a bounding box
     * center: a (3,) numpy array that is the center point
     * center_farthest: a (3,) numpy array that is the point in the mesh farthest from the center
     * center_farthest_distance: the distance between the center and the farthest point
     
    """
    minx, maxx = INF, NEGINF
    miny, maxy = INF, NEGINF
    minz, maxz = INF, NEGINF
    
    for boundprim in iter_prims(mesh):
        verts = boundprim.vertex[boundprim.vertex_index]
        
        minx = min(minx, verts[:,:,0].min())
        miny = min(miny, verts[:,:,1].min())
        minz = min(minz, verts[:,:,2].min())
        
        maxx = max(maxx, verts[:,:,0].max())
        maxy = max(maxy, verts[:,:,1].max())
        maxz = max(maxz, verts[:,:,2].max())
            
    minpt = numpy.array([minx, miny, minz], dtype=numpy.float32)
    maxpt = numpy.array([maxx, maxy, maxz], dtype=numpy.float32)
    bounds = (minpt, maxpt)
    center = centerFromBounds(bounds)
    
    maxdist, maxpt = NEGINF, numpy.array([INF, INF, INF], dtype=numpy.float32)
    for boundprim in iter_prims(mesh):
        verts = boundprim.vertex[boundprim.vertex_index]
        verts.shape = (-1, 3)
        
        dists = v3_array_pt_dist(verts, center)
        maxidx = dists.argmax()
        if dists[maxidx] > maxdist:
            maxdist = dists[maxidx]
            maxpt = verts[maxidx]
    
    return {
        'bounds': bounds,
        'center': center,
        'center_farthest': maxpt,
        'center_farthest_distance': maxdist,
    }

def fmtpt(pt):
    return '<%.7g, %.7g, %.7g>' % (pt[0], pt[1], pt[2])

def printBoundsInfo(mesh):
    boundsinfo = getBoundsInfo(mesh)
    print 'Bounds: <%s, %s>' % (fmtpt(boundsinfo['bounds'][0]), fmtpt(boundsinfo['bounds'][1]))
    print 'Center: %s' % fmtpt(boundsinfo['center'])
    print 'Point farthest from center: %s at distance of %.7g' % (fmtpt(boundsinfo['center_farthest']), boundsinfo['center_farthest_distance'])

def FilterGenerator():
    class PrintBoundsFilter(PrintFilter):
        def __init__(self):
            super(PrintBoundsFilter, self).__init__('print_bounds', 'Prints bounds information about the mesh')
        def apply(self, mesh):
            printBoundsInfo(mesh)
            return mesh
    
    return PrintBoundsFilter()

factory.register(FilterGenerator().name, FilterGenerator)
