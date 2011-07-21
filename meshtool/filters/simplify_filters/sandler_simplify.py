from meshtool.args import *
from meshtool.filters.base_filters import *
import inspect
import numpy
import networkx as nx
from itertools import izip, chain, repeat, imap
import datetime
import math
import __builtin__
import heapq
from render_utils import renderVerts, renderCharts
import gc

if not 'set' in __builtin__.__dict__:
    import sets
    set = sets.Set

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
         
def distinct_vertices(edges):
    return numpy.unique(numpy.array(edges))
         
def calcPerimeter(pts):
    dx = pts[:,0,0]-pts[:,1,0]
    dy = pts[:,0,1]-pts[:,1,1]
    dz = pts[:,0,2]-pts[:,1,2]
    return numpy.sum(numpy.sqrt(dx*dx + dy*dy + dz*dz))

def array_mult(arr1, arr2):
    return arr1[:,0]*arr2[:,0] + arr1[:,1]*arr2[:,1] + arr2[:,2]*arr1[:,2]

def setxor2d(x, y):
    """Same as setxor1d except it handles a 2-dimensional array"""
    return numpy.setxor1d(x.view([('',x.dtype)] * x.shape[1]), y.view([('',y.dtype)] * y.shape[1])).view(x.dtype).reshape(-1, x.shape[1])

def intersect2d(x, y):
    """Same as intersect1d except it handles a 2-dimensional array"""
    return numpy.intersect1d(x.view([('',x.dtype)] * x.shape[1]), y.view([('',y.dtype)] * y.shape[1])).view(x.dtype).reshape(-1, x.shape[1])

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
    for boundcontroller in mesh.scene.objects('controller'):
        boundgeom = boundcontroller.geometry
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
    facegraph.add_nodes_from(( (i, {'tris':[tri]})
                               for i, tri in
                               enumerate(all_indices) ))
    end_operation()
    print next(t)
    
    print 'building face edges...',
    begin_operation()
    for e in vertexgraph.edges_iter(data=True):
        v1, v2, adjacent_faces = e
        adjacent_faces = adjacent_faces.keys()
        if len(adjacent_faces) == 2:
            facegraph.add_edge(adjacent_faces[0], adjacent_faces[1], edges=[sorted((v1,v2))])
    end_operation()
    print next(t)
    
    merge_priorities = []
    maxerror = 0
    
    print 'calculating error...',
    begin_operation()
    for v1, v2 in facegraph.edges_iter():
        edges1 = numpy.array(list(chain.from_iterable(iter(e[2]['edges']) for e in facegraph.edges(v1, data=True))))
        edges2 = numpy.array(list(chain.from_iterable(iter(e[2]['edges']) for e in facegraph.edges(v2, data=True))))
        merged = setxor2d(edges1, edges2)
        if len(merged) > 0:
            error = calcPerimeter(all_vertices[merged])**2
            error += calcFitError(all_vertices[numpy.unique(merged.flat)])
            if error > maxerror: maxerror = error
            merge_priorities.append((error, (v1, v2)))
    end_operation()
    print next(t)
        
    print 'creating priority queue...',
    begin_operation()
    heapq.heapify(merge_priorities)
    end_operation()
    print next(t)
    
    print 'merging charts...',
    begin_operation()
    node_count = len(all_indices)
    while len(merge_priorities) > 0:
        (error, (face1, face2)) = heapq.heappop(merge_priorities)
        
        #this can happen if we have already merged one of these
        if face1 not in facegraph or face2 not in facegraph:
            continue
        
        # if the number of corners of the merged face is less than 3, disqualify it
        # where a "corner" is defined as a vertex with at least 3 adjacent faces
        edges1 = numpy.array(list(chain.from_iterable(iter(e[2]['edges']) for e in facegraph.edges(face1, data=True))))
        edges2 = numpy.array(list(chain.from_iterable(iter(e[2]['edges']) for e in facegraph.edges(face2, data=True))))
        combined_edges = setxor2d(edges1, edges2)
        combined_vertices = distinct_vertices(combined_edges)
        numcorners = 0
        for v in combined_vertices:
            adjacent = []
            for (vv1, vv2, adj_dict) in vertexgraph.edges_iter(v, data=True):
                adjacent.extend(adj_dict.keys())
            adjacent = numpy.unique(numpy.array(adjacent))
            numadj = len([a for a in adjacent if a != face1 and a != face2])
            if numadj >= 3:
                numcorners += 1
        if numcorners < 3:
            continue
        
        if math.log(error) / math.log(maxerror) > 0.9:
            break
        
        combined_tris = facegraph.node[face1]['tris'] + facegraph.node[face2]['tris']
        
        newface = node_count
        node_count += 1
        facegraph.add_node(newface, tris=combined_tris)
        
        edges_to_add = []
        topush = []
        
        for curface in (face1, face2):
            for edge in facegraph.edges_iter(curface):
                otherface = edge[1]
                otheredges = numpy.array(list(chain.from_iterable(iter(e[2]['edges']) for e in facegraph.edges(otherface, data=True))))
                commonedges = intersect2d(combined_edges, otheredges)
                
                connected_components_graph = nx.from_edgelist(commonedges)
                if nx.algorithms.components.connected.number_connected_components(connected_components_graph) > 1:
                    continue
                
                edges_to_add.append((newface, otherface, {'edges':commonedges}))
                
                merged = setxor2d(combined_edges, otheredges)
                if len(merged) > 0:
                    error = calcPerimeter(all_vertices[merged])**2
                    error += calcFitError(all_vertices[numpy.unique(merged.flat)])
                    if error > maxerror: maxerror = error
                    topush.append((error, (newface, otherface)))
        
        facegraph.add_edges_from(edges_to_add)
        for p in topush:
            heapq.heappush(merge_priorities, p)

        for v in combined_vertices:
            edges = vertexgraph.edges(v, data=True)
            for (vv1, vv2, facedata) in edges:
                if face1 in facedata or face2 in facedata:
                    if face1 in facedata:
                        del facedata[face1]
                    if face2 in facedata:
                        del facedata[face2]
                    facedata[newface] = True
                    vertexgraph.add_edge(vv1, vv2, attr_dict=facedata)

        facegraph.remove_node(face1)
        facegraph.remove_node(face2)

    end_operation()
    print next(t)
    
    print 'final number of faces =', len(facegraph)
    print 'final number of connected components =', nx.algorithms.components.connected.number_connected_components(facegraph)
    
    renderCharts(facegraph, all_vertices)


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