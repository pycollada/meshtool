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
from graph_utils import astar_path, dfs_interior_nodes
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

def v3dist(pt1, pt2):
    d = pt1 - pt2
    return math.sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2])

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
        corners = []
        for v in combined_vertices:
            adjacent = []
            for (vv1, vv2, adj_dict) in vertexgraph.edges_iter(v, data=True):
                adjacent.extend(adj_dict.keys())
            adjacent = numpy.unique(numpy.array(adjacent))
            numadj = len([a for a in adjacent if a != face1 and a != face2])
            if numadj >= 3:
                corners.append(v)
        if len(corners) < 3:
            continue
        
        if math.log(error) / math.log(maxerror) > 0.9:
            break
        
        combined_tris = facegraph.node[face1]['tris'] + facegraph.node[face2]['tris']
        
        newface = node_count
        node_count += 1
        facegraph.add_node(newface, corners=corners, tris=combined_tris)
        
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
    
    print 'updating corners...',
    begin_operation()
    for face, facedata in facegraph.nodes_iter(data=True):
        edges = numpy.array(list(chain.from_iterable(iter(e[2]['edges']) for e in facegraph.edges(face, data=True))))
        vertices = distinct_vertices(edges)
        corners = []
        for v in vertices:
            adjacent = []
            for (vv1, vv2, adj_dict) in vertexgraph.edges_iter(v, data=True):
                adjacent.extend(adj_dict.keys())
            adjacent = numpy.unique(numpy.array(adjacent))
            numadj = len([a for a in adjacent if a != face])
            if numadj >= 3:
                corners.append(v)
        facegraph.add_node(face, corners=corners)
    end_operation()
    print next(t)
    
    print 'computing distance between points',
    begin_operation()
    for v1, v2 in vertexgraph.edges_iter():
        vertexgraph.add_edge(v1, v2, distance=v3dist(all_vertices[v1],all_vertices[v2]))
    end_operation()
    print next(t)
    
    print 'straightening chart boundaries...',
    begin_operation()
    numstraightened = 0
    for (face1, face2, databetween) in facegraph.edges_iter(data=True):        
        edges1 = numpy.array(list(chain.from_iterable(iter(e[2]['edges']) for e in facegraph.edges(face1, data=True))))
        edges2 = numpy.array(list(chain.from_iterable(iter(e[2]['edges']) for e in facegraph.edges(face2, data=True))))
        combined_edges = setxor2d(edges1, edges2)
        shared_edges = numpy.array(databetween['edges'])
        shared_vertices = numpy.unique(shared_edges)

        corners1 = facegraph.node[face1]['corners']
        corners2 = facegraph.node[face2]['corners']
        combined_corners = [c for c in corners1 if c in corners2 and c in shared_edges]
        
        #this can happen if only a single vertex is shared
        # in that case, we can't straighten
        if len(combined_corners) < 2:
            #print 'edges1', edges1
            #print 'edges2', edges2
            #print 'corners1', corners2
            #print 'corners2', corners2
            #print 'combined', combined_corners
            #print 'shared_edges', shared_edges
            continue

        assert(len(combined_corners) <= 2)
        
        edges1 = setxor2d(edges1, shared_edges)
        edges2 = setxor2d(edges2, shared_edges)
        stop_nodes = set(combined_edges.flatten())
        
        corner1, corner2 = combined_corners
        straightened_path = astar_path(vertexgraph, corner1, corner2,
                                       heuristic=lambda x,y: v3dist(all_vertices[x], all_vertices[y]),
                                       weight='distance', exclude=stop_nodes)
        
        # if we already have the shortest path, nothing to do
        if set(shared_vertices) == set(straightened_path):
            continue
        
        new_combined_edges = []
        for i in range(len(straightened_path)-1):
            new_combined_edges.append(sorted((straightened_path[i], straightened_path[i+1])))
        new_combined_edges = numpy.array(new_combined_edges)
        new_edges1 = setxor2d(edges1, new_combined_edges)
        new_edges2 = setxor2d(edges2, new_combined_edges)
        
        # This can happen if one of the faces is a small and the
        # boundary is convex. The shortest path actually encompasses
        # the smaller face, but this would be equivalent to merging the
        # two faces. If we didn't merge these two in the previous step,
        # it was because the cost was too high or it would violate one of
        # the constraints, so just ignore this 
        if len(new_edges1) == 0 or len(new_edges2) == 0:
            continue
                    
        #print 'corner', corner1, 'to corner', corner2
        #print 'stop if hitting', stop_nodes
        
        #print 'original path', shared_edges
        
        #print 'shortest path', straightened_path
        
        #print 'new_combined_edges', new_combined_edges
        #print 'newedges1', new_edges1
        #print 'newedges2', new_edges2
        
        combined_tris = facegraph.node[face1]['tris'] + facegraph.node[face2]['tris']
        boundary1 = numpy.unique(new_edges1)
        boundary2 = numpy.unique(new_edges2)
        
        nodein1 = None
        nodein2 = None
        
        for tri in combined_tris:
            for pt in range(3):
                vert = tri[pt]
                if vert in straightened_path:
                    continue
                if vert in boundary1:
                    nodein1 = tri[(pt+1) % 3]
                if vert in boundary2:
                    nodein2 = tri[(pt+1) % 3]
                if nodein1 and nodein2:
                    break
                
        if not nodein1 or not nodein2:
            print 'warning: skipping because nodein1 or nodein2 was none'
            continue
        
        #print 'nodein1', nodein1
        #print 'nodein2', nodein2
        
        vertexset1 = set(numpy.setdiff1d(boundary1,numpy.array(straightened_path)))
        vertexset2 = set(numpy.setdiff1d(boundary2,numpy.array(straightened_path)))
        
        constrained_set = set(numpy.unique(list(chain.from_iterable(combined_tris))))
        #print 'constrained_set', constrained_set
        #print 'vertexset1', vertexset1
        #print 'vertexset2', vertexset2
        
        allin1 = list(dfs_interior_nodes(vertexgraph,
                                         starting=vertexset1,
                                         boundary=numpy.concatenate((boundary1, numpy.array(straightened_path))),
                                         subset=constrained_set))
        allin2 = list(dfs_interior_nodes(vertexgraph,
                                         starting=vertexset2,
                                         boundary=numpy.concatenate((boundary2, numpy.array(straightened_path))),
                                         subset=constrained_set))
        
        #print 'allin1len', len(allin1) + len(boundary1) + len(straightened_path)
        #print 'allin2len', len(allin2) + len(boundary2) + len(straightened_path)
        
        #print 'allin1', allin1
        #print 'boundary1', boundary1
        #print 'allin2', allin2
        #print 'boundary2', boundary2
        #print 'straightpath', straightened_path
        
        vertexset1 = set(allin1).union(vertexset1).union(set(straightened_path))
        vertexset2 = set(allin2).union(vertexset2).union(set(straightened_path))
        #print 'vertexset1', vertexset1
        #print 'vertexset2', vertexset2
        tris1 = []
        tris2 = []
        for tri in combined_tris:
            if tri[0] in vertexset1 and tri[1] in vertexset1 and tri[2] in vertexset1:
                tris1.append(tri)
            elif tri[0] in vertexset2 and tri[1] in vertexset2 and tri[2] in vertexset2:
                tris2.append(tri)
            else:
                #print 'found tri', tri
                #print 'tri0 in 1?', tri[0] in vertexset1
                #print 'tri1 in 1?', tri[1] in vertexset1
                #print 'tri2 in 1?', tri[2] in vertexset1
                #print 'tri0 in 2?', tri[0] in vertexset2
                #print 'tri1 in 2?', tri[1] in vertexset2
                #print 'tri2 in 2?', tri[2] in vertexset2
                assert(False)
        
        if len(tris1) == 0 or len(tris2) == 0:
            print 'warning: one of the tris was empty, skipping'
            continue
        
        assert(len(tris1) + len(tris2) == len(combined_tris))
        
        numstraightened += 1
        facegraph.add_edge(face1, face2, edges=new_combined_edges)
        facegraph.add_node(face1, tris=tris1)
        facegraph.add_node(face2, tris=tris2)
        
        #print 'origintris1', len(facegraph.node[face1]['tris'])
        #print 'origintris2', len(facegraph.node[face2]['tris'])
        #print 'tris1', len(tris1)
        #print 'tris2', len(tris2)
        
        #import sys
        #sys.exit(0)
        #blah = raw_input()
    end_operation()
    print next(t)
    
    print 'numedges', facegraph.number_of_edges()
    print 'numstraightened', numstraightened
    
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