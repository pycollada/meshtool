from meshtool.args import *
from meshtool.filters.base_filters import *
import inspect
import numpy
import networkx as nx
from itertools import izip, chain, repeat, imap
import datetime
import __builtin__
from priority_queue_set import PriorityQueueSet
import gc

if not 'set' in __builtin__.__dict__:
    import sets
    set = sets.Set

def renderVerts(verts, idx):
    from meshtool.filters.panda_filters.pandacore import getVertexData, attachLights, ensureCameraAt
    from meshtool.filters.panda_filters.pandacontrols import KeyboardMovement, MouseDrag, MouseScaleZoom
    from panda3d.core import GeomTriangles, Geom, GeomNode
    from direct.showbase.ShowBase import ShowBase
    vdata = getVertexData(verts, idx)
    gprim = GeomTriangles(Geom.UHStatic)
    gprim.addConsecutiveVertices(0, 3*len(idx))
    gprim.closePrimitive()
    pgeom = Geom(vdata)
    pgeom.addPrimitive(gprim)
    node = GeomNode("primitive")
    node.addGeom(pgeom)
    p3dApp = ShowBase()
    #attachLights(render)
    geomPath = render.attachNewNode(node)
    geomPath.setRenderModeWireframe()
    ensureCameraAt(geomPath, base.camera)
    KeyboardMovement()
    #render.setShaderAuto()
    p3dApp.run()

#after python2.5, uniqu1d was renamed to unique
args, varargs, keywords, defaults = inspect.getargspec(numpy.unique)
if 'return_inverse' not in args:
    numpy.unique = numpy.unique1d

def timer():
    begintime = datetime.datetime.now()
    while True:
        curtime = datetime.datetime.now()
        yield (curtime-begintime)
        begintime = curtime
             
def merge_edges(e1, e2):
    return list(set(imap(tuple,imap(sorted, chain(e1,e2)))))
         
def calcPerimeter(pts):
    dx = pts[:,0,0]-pts[:,1,0]
    dy = pts[:,0,1]-pts[:,1,1]
    dz = pts[:,0,2]-pts[:,1,2]
    return numpy.sum(numpy.sqrt(dx*dx + dy*dy + dz*dz))

def array_mult(arr1, arr2):
    return arr1[:,0]*arr2[:,0] + arr1[:,1]*arr2[:,1] + arr2[:,2]*arr1[:,2]
def array_dot(arr1, arr2):
    return numpy.sqrt( array_mult(arr1, arr2) )

def calcFitError(pts):
    # this computes the outer product of each vector
    # in the array, element-wise, then sums up the 3x3
    # matrices
    #  A = sum{v_i * v_i^T} 
    A = numpy.sum(pts[...,None] * pts[:,None,:], 0)
    
    # b = mean(v_i)
    b = numpy.mean(pts, 0)

    # Z = A - (b * b^T) / c
    Z = A - numpy.outer(b,b) / len(pts)
    
    # n (normal of best fit plane) is the eigenvector
    # corresponding to the minimum eigenvalue
    eigvals, eigvecs = numpy.linalg.eig(Z)
    n = eigvecs[numpy.argmin(eigvals)]
    
    # d (scalar offset of best fit plane) = -n^T * b / c
    d = numpy.inner(-n, b) / len(pts)
    
    # final error is the square of the mean distance of each point to the plane
    mean_dist = numpy.mean(array_mult(n[None,:].repeat(len(pts), axis=0), pts) + d)
    Efit = mean_dist * mean_dist
    
    #print 'A', A
    #print 'b', b
    #print 'Z', Z
    #print 'eigvals', eigvals
    #print 'eigvecs', eigvecs
    #print 'n', n
    #print 'd', d
    #print 'Efit', Efit
    
    return Efit

def begin_operation():
    gc.disable()
def end_operation():
    gc.enable()

def sandler_simplify(mesh):
    all_vertices = []
    all_indices = []
    vertex_offset = 0
    t = timer()
    
    print 'building aggregated vertex and triangle list...',
    begin_operation()
    for boundgeom in mesh.scene.objects('geometry'):
        for boundprim in boundgeom.primitives():
            all_vertices.append(boundprim.vertex)
            all_indices.append(boundprim.vertex_index + vertex_offset)
            vertex_offset += len(boundprim.vertex)
            
    all_vertices = numpy.concatenate(all_vertices)
    all_indices = numpy.concatenate(all_indices)
    end_operation()
    print next(t)
    
    print 'uniqifying the list...',
    begin_operation()
    unique_data, index_map = numpy.unique(all_vertices.view([('',all_vertices.dtype)]*all_vertices.shape[1]), return_inverse=True)
    all_vertices = unique_data.view(all_vertices.dtype).reshape(-1,all_vertices.shape[1])
    all_indices = index_map[all_indices]
    end_operation()
    print next(t)
    
    print 'building vertex vertices...',
    begin_operation()
    vertexgraph = nx.Graph()
    vertexgraph.add_nodes_from(xrange(len(all_vertices)))
    end_operation()
    print next(t)
    
    print 'building vertex edges...',
    begin_operation()
    vertexgraph.add_edges_from(( (edge[0], edge[1], {facenum:True})
                                 for facenum, edge in
                                 enumerate(all_indices[:,(0,1)]) ))
    vertexgraph.add_edges_from(( (edge[0], edge[1], {facenum:True})
                                 for facenum, edge in
                                 enumerate(all_indices[:,(0,2)]) ))
    vertexgraph.add_edges_from(( (edge[0], edge[1], {facenum:True})
                                 for facenum, edge in
                                 enumerate(all_indices[:,(1,2)]) ))
    end_operation()
    print next(t)
    
    print 'building face vertices...',
    begin_operation()
    facegraph = nx.Graph()
    facegraph.add_nodes_from(( (i, {'tris':[tri], 'edges':[(tri[0], tri[1]), (tri[0], tri[2]), (tri[1], tri[2])]})
                               for i, tri in
                               enumerate(all_indices) ))
    end_operation()
    print next(t)
    
    print 'building face edges...',
    begin_operation()
    for e in vertexgraph.edges_iter(data=True):
        adjacent_faces = e[2].keys()
        if len(adjacent_faces) == 2:
            facegraph.add_edge(adjacent_faces[0], adjacent_faces[1], error=0.0)
    end_operation()
    print next(t)
    
    merge_priorities = []
    
    print 'calculating error...',
    begin_operation()
    for v1, v2 in facegraph.edges_iter():
        edges1 = facegraph.node[v1]['edges']
        edges2 = facegraph.node[v2]['edges']
        merged = numpy.array(merge_edges(edges1, edges2))
        error = calcPerimeter(all_vertices[merged])**2
        error += calcFitError(all_vertices[numpy.unique(merged.flat)])
        merge_priorities.append((error, (v1, v2)))
        facegraph.edge[v1][v2]['error'] = error
    end_operation()
    print next(t)
        
    print 'creating priority queue...',
    begin_operation()
    merge_priorities = PriorityQueueSet(merge_priorities)
    end_operation()
    print next(t)
    
    while len(merge_priorities) > 0:
        print merge_priorities.pop_smallest()


def FilterGenerator():
    class SandlerSimplificationFilter(OpFilter):
        def __init__(self):
            super(SandlerSimplificationFilter, self).__init__('sandler_simplify', 'Simplifies the mesh based on sandler, et al. method.')
        def apply(self, mesh):
            sandler_simplify(mesh)
            return mesh
    return SandlerSimplificationFilter()
from meshtool.filters import factory
factory.register(FilterGenerator().name, FilterGenerator)