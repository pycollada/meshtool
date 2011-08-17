from meshtool.args import *
from meshtool.filters.base_filters import *
import inspect
import numpy
import networkx as nx
from itertools import chain, izip
import datetime
import math
import __builtin__
import heapq
from render_utils import renderVerts, renderCharts
from graph_utils import astar_path, dfs_interior_nodes, super_cycle
import gc
import sys
import random
import collada
import Image
import ImageDraw
import ImageFile
from meshtool.filters.atlas_filters.rectpack import RectPack
from StringIO import StringIO
import meshtool.filters

ImageFile.MAXBLOCK = 20*1024*1024 # default is 64k, setting to 20MB to handle large textures

#python 2.5 set
if not 'set' in __builtin__.__dict__:
    import sets
    set = sets.Set

#after python2.5, uniqu1d was renamed to unique
args, varargs, keywords, defaults = inspect.getargspec(numpy.unique)
if 'return_inverse' not in args:
    numpy.unique = numpy.unique1d

#next for py 2.5
try: next
except NameError:
    def next ( obj ): return obj.next()

#chain.from_iterable is not in 2.5
try: chain.from_iterable
except AttributeError:
    _chain = chain
    class ChainWrapper(chain):
        def __call__(self, *args, **kwargs):
            return _chain(*args, **kwargs)
        @classmethod
        def from_iterable(self, iterables):
            for it in iterables:
                for element in it:
                    yield element
    chain = ChainWrapper()

#itertools.combinations is not in python 2.5
try:
    from itertools import combinations
except ImportError:
    def combinations(iterable, r):
        pool = tuple(iterable)
        n = len(pool)
        if r > n:
            return
        indices = range(r)
        yield tuple(pool[i] for i in indices)
        while True:
            for i in reversed(range(r)):
                if indices[i] != i + n - r:
                    break
            else:
                return
            indices[i] += 1
            for j in range(i+1, r):
                indices[j] = indices[j-1] + 1
            yield tuple(pool[i] for i in indices)

def timer():
    begintime = datetime.datetime.now()
    while True:
        curtime = datetime.datetime.now()
        yield (curtime-begintime)
        begintime = curtime
         
def calcPerimeter(pts):
    """Calculates the perimeter of an area by 
    summing the distance between the points in
    a set of edges"""
    dx = pts[:,0,0]-pts[:,1,0]
    dy = pts[:,0,1]-pts[:,1,1]
    dz = pts[:,0,2]-pts[:,1,2]
    return numpy.sum(numpy.sqrt(dx*dx + dy*dy + dz*dz))

def v3dist(pt1, pt2):
    """Calculates the distance between two 3d points element-wise
    along an array"""
    d = pt1 - pt2
    return math.sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2])

def array_mult(arr1, arr2):
    return arr1[:,0]*arr2[:,0] + arr1[:,1]*arr2[:,1] + arr2[:,2]*arr1[:,2]
def array_dot(arr1, arr2):
    return numpy.sqrt( array_mult(arr1, arr2) )

def calcFitError(pts):
    """
    Calculates E_fit, which is an error value from
    
     - Hierarchical Face Clustering on Polygonal Surfaces
     - Michael Garland, et al.
     - See section 3.2
    
    Computes the best fit plane through a set of points
    and then calculates the squared mean distance between
    the points and the plane as the error value"""
    
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
    eigvals, eigvecs = numpy.linalg.eigh(Z)
    n = eigvecs[numpy.argmin(eigvals)]
    
    # d (scalar offset of best fit plane) = -n^T * b / c
    d = numpy.inner(-n, b)
    
    # final error is the square of the mean distance of each point to the plane
    mean_dist = numpy.mean(numpy.abs(numpy.dot(pts, n) + d))
    Efit = mean_dist * mean_dist
    
    return Efit

def tri_areas_3d(arr):
    crosses = numpy.cross(arr[:,0] - arr[:,1], arr[:,0] - arr[:,2])
    return array_dot(crosses, crosses) / 2.0
def tri_areas_2d(arr):
    return ( arr[:,0,0]*(arr[:,1,1]-arr[:,2,1]) + 
             arr[:,1,0]*(arr[:,2,1]-arr[:,0,1]) + 
             arr[:,2,0]*(arr[:,0,1]-arr[:,1,1])
            ) / 2.0

def stretch_metric(t3d, t2d, return_A2d=False, flippedCheck=None, normalize=False):
    """
    Computes the texture stretch metric from
    
     - Texture Mapping Progressive Meshes
     - Pedro V. Sander, et al.
     - See section 3
    """
     
    q1 = t3d[:,0]
    q2 = t3d[:,1]
    q3 = t3d[:,2]
    s1 = t2d[:,0,0]
    s2 = t2d[:,1,0]
    s3 = t2d[:,2,0]
    t1 = t2d[:,0,1]
    t2 = t2d[:,1,1]
    t3 = t2d[:,2,1]
    A2d = ((s2-s1)*(t3-t1) - (s3-s1)*(t2-t1)) / 2.0
    A2d[A2d == 0] = numpy.inf
    assert(not(numpy.any(A2d==0) and return_A2d))
    
    S_s = (q1*(t2-t3)[:,None] + q2*(t3-t1)[:,None] + q3*(t1-t2)[:,None]) / (2.0 * A2d)[:,None]
    S_t = (q1*(s3-s2)[:,None] + q2*(s1-s3)[:,None] + q3*(s2-s1)[:,None]) / (2.0 * A2d)[:,None]
    
    L2 = numpy.sqrt((array_mult(S_s,S_s) + array_mult(S_t,S_t)) / 2.0)
    if flippedCheck is not None:
        L2[numpy.logical_xor(A2d < 0, flippedCheck < 0)] = numpy.inf
        L2[A2d == 0] = numpy.inf
    if normalize:
        A3d = tri_areas_3d(t3d)
        L2 = numpy.sqrt(numpy.sum(L2*L2*A3d) / numpy.sum(A3d)) * numpy.sqrt(numpy.sum(numpy.abs(A2d)) / numpy.sum(A3d))
    if return_A2d:
        return L2, A2d
    return L2

def drawChart(chart_tris, border_verts, new_uvs, newvert2idx):
    W, H = 500, 500
    im = Image.new("RGB", (W,H), (255,255,255))
    draw = ImageDraw.Draw(im)
    for tri in chart_tris:
        for edge in [(tri[0],tri[1]), (tri[0], tri[2]), (tri[1], tri[2])]:
            pt1, pt2 = edge
            u1 = new_uvs[newvert2idx[pt1]][0]
            u2 = new_uvs[newvert2idx[pt2]][0]
            v1 = new_uvs[newvert2idx[pt1]][1]
            v2 = new_uvs[newvert2idx[pt2]][1]
            uv1 = ( u1 * W, (1.0-v1) * H )
            uv2 = ( u2 * W, (1.0-v2) * H )
            color1 = (255,0,0) if pt1 in border_verts else (0,0,255)
            color2 = (255,0,0) if pt2 in border_verts else (0,0,255)
            draw.ellipse((uv1[0]-2, uv1[1]-2, uv1[0]+2, uv1[1]+2), outline=color1, fill=color1)
            draw.ellipse((uv2[0]-2, uv2[1]-2, uv2[0]+2, uv2[1]+2), outline=color2, fill=color2)
            draw.line([uv1, uv2], fill=(0,0,0))
    del draw
    im.show()

def transformblit(src_tri, dst_tri, src_img, dst_img):
    """Pastes a triangular region from one image
    into a triangular region in another image by
    using an affine transformation"""
    ((x11,x12), (x21,x22), (x31,x32)) = src_tri
    ((y11,y12), (y21,y22), (y31,y32)) = dst_tri

    minx = int(math.floor(min(y11, y21, y31)))
    miny = int(math.floor(min(y12, y22, y32)))
    y11 -= minx
    y21 -= minx
    y31 -= minx
    y12 -= miny
    y22 -= miny
    y32 -= miny
    sizex = int(math.ceil(max(y11, y21, y31)))
    sizey = int(math.ceil(max(y12, y22, y32)))
    
    M = numpy.array([
                     [y11, y12, 1, 0, 0, 0],
                     [y21, y22, 1, 0, 0, 0],
                     [y31, y32, 1, 0, 0, 0],
                     [0, 0, 0, y11, y12, 1],
                     [0, 0, 0, y21, y22, 1],
                     [0, 0, 0, y31, y32, 1]
                ])
    
    y = numpy.array([x11, x21, x31, x12, x22, x32])

    A = numpy.linalg.solve(M, y)

    transformed = src_img.transform((sizex, sizey), Image.AFFINE, A, Image.BICUBIC)
    
    mask = Image.new('1', (sizex, sizey))
    maskdraw = ImageDraw.Draw(mask)
    maskdraw.polygon(((y11,y12), (y21,y22), (y31,y32)), outline=255, fill=255)

    dst_img.paste(transformed, (minx, miny), mask=mask)

class SanderSimplify(object):

    def __init__(self, mesh):
        self.mesh = mesh
        
        self.all_vertices = []
        self.all_normals = []
        self.all_orig_uvs = []
        
        self.all_vert_indices = []
        self.all_normal_indices = []
        self.all_orig_uv_indices = []
        
        self.index_offset = 0
        
        self.tri2material = []
        
        self.timer = timer()
        
        self.begin_operation('Building aggregated vertex and triangle list...')
        for boundgeom in chain(mesh.scene.objects('geometry'), mesh.scene.objects('controller')):
            if isinstance(boundgeom, collada.controller.BoundController):
                boundgeom = boundgeom.geometry
            for boundprim in boundgeom.primitives():
                
                self.all_vertices.append(boundprim.vertex)
                self.all_normals.append(boundprim.normal)
                self.all_vert_indices.append(boundprim.vertex_index + self.index_offset)
                self.all_normal_indices.append(boundprim.normal_index + self.index_offset)
                
                if boundprim.texcoordset and len(boundprim.texcoordset) > 0:
                    self.all_orig_uvs.append(boundprim.texcoordset[0])
                    self.all_orig_uv_indices.append(boundprim.texcoord_indexset[0] + self.index_offset)
                else:
                    self.all_orig_uv_indices.append(numpy.zeros(shape=(len(boundprim.index), 3)))
                
                self.tri2material.append((self.index_offset, boundprim.material))

                self.index_offset += len(boundprim.vertex)
                
        self.all_vertices = numpy.concatenate(self.all_vertices)
        self.all_normals = numpy.concatenate(self.all_normals)
        self.all_orig_uvs = numpy.concatenate(self.all_orig_uvs)
        self.all_vert_indices = numpy.concatenate(self.all_vert_indices)
        self.all_normal_indices = numpy.concatenate(self.all_normal_indices)
        self.all_orig_uv_indices = numpy.concatenate(self.all_orig_uv_indices)
    
        assert(len(self.all_vert_indices) == len(self.all_normal_indices) == len(self.all_orig_uv_indices))
    
        self.end_operation()

    def begin_operation(self, message):
        print message,
        sys.stdout.flush()
        gc.disable()
    def end_operation(self):
        seconds = next(self.timer).seconds
        hours = seconds // 3600
        seconds -= (hours * 3600)
        minutes = seconds // 60
        seconds -= (minutes * 60)
        if hours > 0: print "%dh" % hours,
        if hours > 0 or minutes > 0: print "%dm" % minutes,
        print "%ds" % seconds
        gc.enable()

    def uniqify_list(self):
        self.begin_operation('Uniqifying the list...')
        
        all_vertices = self.all_vertices
        
        unique_data, index_map = numpy.unique(all_vertices.view([('',all_vertices.dtype)]*all_vertices.shape[1]), return_inverse=True)
        all_vertices = unique_data.view(all_vertices.dtype).reshape(-1,all_vertices.shape[1])
        self.all_vert_indices = index_map[self.all_vert_indices]
        
        #scale to known range so error values are normalized
        all_vertices[:,0] -= numpy.min(all_vertices[:,0])
        all_vertices[:,1] -= numpy.min(all_vertices[:,1])
        all_vertices[:,2] -= numpy.min(all_vertices[:,2])
        all_vertices *= 1000.0 / numpy.max(all_vertices)
        
        self.all_vertices = all_vertices
        
        self.end_operation()

    def build_vertex_graph(self):
        self.begin_operation('Building vertex graph...')
        vertexgraph = nx.Graph()
        vertexgraph.add_nodes_from(xrange(len(self.all_vertices)))

        vertexgraph.add_edges_from(self.all_vert_indices[:,(0,1)])
        vertexgraph.add_edges_from(self.all_vert_indices[:,(0,2)])
        vertexgraph.add_edges_from(self.all_vert_indices[:,(1,2)])
        
        vertexgraph.add_nodes_from(( (edge[0], {facenum: True})
                                     for facenum, edge in
                                     enumerate(self.all_vert_indices[:,(0,1)]) ))
        vertexgraph.add_nodes_from(( (edge[0], {facenum: True})
                                     for facenum, edge in
                                     enumerate(self.all_vert_indices[:,(0,2)]) ))
        vertexgraph.add_nodes_from(( (edge[0], {facenum: True})
                                     for facenum, edge in
                                     enumerate(self.all_vert_indices[:,(1,2)]) ))
        vertexgraph.add_nodes_from(( (edge[1], {facenum: True})
                                     for facenum, edge in
                                     enumerate(self.all_vert_indices[:,(0,1)]) ))
        vertexgraph.add_nodes_from(( (edge[1], {facenum: True})
                                     for facenum, edge in
                                     enumerate(self.all_vert_indices[:,(0,2)]) ))
        vertexgraph.add_nodes_from(( (edge[1], {facenum: True})
                                     for facenum, edge in
                                     enumerate(self.all_vert_indices[:,(1,2)]) ))
        
        self.vertexgraph = vertexgraph
        self.end_operation()

    def build_face_graph(self):
        self.begin_operation('Building face graph...')
        facegraph = nx.Graph()
        facegraph.add_nodes_from(( (i, {'tris':[i], 
                                        'edges':set([tuple(sorted([tri[0], tri[1]])),
                                                     tuple(sorted([tri[1], tri[2]])),
                                                     tuple(sorted([tri[0], tri[2]]))])})
                                   for i, tri in
                                   enumerate(self.all_vert_indices) ))

        self.invalid_edges = set()
        for (v1, v2) in self.vertexgraph.edges_iter():
            adj_v1 = set(self.vertexgraph.node[v1].keys())
            adj_v2 = set(self.vertexgraph.node[v2].keys())
            adj_both = adj_v1.intersection(adj_v2)
            if len(adj_both) < 2:
                continue
            numadded = 0
            for (f1, f2) in combinations(adj_both, 2):
                f1tri = self.all_vert_indices[f1]
                f2tri = self.all_vert_indices[f2]

                #if faces are the same up to winding order, they shouldnt be considered connected
                if sorted(f1tri) == sorted(f2tri):
                    continue
                edges = [(f1tri[0], f1tri[1]), (f1tri[1], f1tri[2]), (f1tri[2], f1tri[0])]

                #check to see if the adjacent triangles traverse a shared edge in the same order
                # if they do, then they are facing opposite directions and so should not be merged
                reallyConnected = True
                for (f1v1, f1v2) in edges:
                    if not f1v1 in f2tri or not f1v2 in f2tri:
                        continue
                    wheref2v1 = numpy.where(f2tri == f1v1)[0][0]
                    nextlocf2 = wheref2v1 + 1 if wheref2v1 < 2 else 0
                    if f2tri[nextlocf2] == f1v2:
                        reallyConnected = False
                
                if reallyConnected:
                    numadded += 1
                    facegraph.add_edge(f1, f2)
            if numadded > 2:
                #edge is adjacent to more than two faces, so don't merge it
                self.invalid_edges.add(tuple(sorted([v1, v2])))
                
        self.facegraph = facegraph
        
        self.end_operation()

    def initialize_chart_merge_errors(self):
        self.merge_priorities = []
        self.maxerror = 0
        
        self.begin_operation('(Step 1 of 7) Creating priority queue for initial merges...')
        for v1, v2 in self.facegraph.edges_iter():
            edges1 = self.facegraph.node[v1]['edges']
            edges2 = self.facegraph.node[v2]['edges']
            merged = numpy.array(list(edges1.symmetric_difference(edges2)))
            if len(merged) > 0:
                error = calcPerimeter(self.all_vertices[merged])**2
                error += calcFitError(self.all_vertices[merged.flatten()])
                if error > self.maxerror: self.maxerror = error
                self.merge_priorities.append((error, (v1, v2)))

        heapq.heapify(self.merge_priorities)
        
        self.end_operation()

    def merge_charts(self):
        self.begin_operation('(Step 1 of 7) Merging charts...')
        node_count = len(self.all_vert_indices)
        while len(self.merge_priorities) > 0:
            (error, (face1, face2)) = heapq.heappop(self.merge_priorities)
            
            #this can happen if we have already merged one of these
            if face1 not in self.facegraph or face2 not in self.facegraph:
                continue
            
            edges1 = self.facegraph.node[face1]['edges']
            edges2 = self.facegraph.node[face2]['edges']
            combined_edges = edges1.symmetric_difference(edges2)
            shared_edges = edges1.intersection(edges2)
            
            #if the length of the xor of the edges of the two faces is less than either of the original
            # then it's creating some kind of collapse, so disallow
            if len(combined_edges) < len(edges1) or len(combined_edges) < len(edges2):
                continue
    
            if len(self.invalid_edges) > 0 and len(shared_edges.intersection(self.invalid_edges)) > 0:
                continue
    
            #check if boundary is more than one connected component
            connected_components_graph = nx.from_edgelist(combined_edges)
            if nx.algorithms.components.connected.number_connected_components(connected_components_graph) > 1:
                if len(edges1) == 3 and len(edges2) == 3: print 'rejecting cc1'
                continue
            if len(nx.algorithms.cycles.cycle_basis(connected_components_graph)) != 1:
                print 'invalid merge because of too many cycles'
                continue
    
            # if the number of corners of the merged face is less than 3, disqualify it
            # where a "corner" is defined as a vertex with at least 3 adjacent faces
            
            corners1 = set()
            vertices1 = set(chain.from_iterable(edges1))
            for v in vertices1:
                adj_v = set(self.vertexgraph.node[v].keys())
                numadj = self.facegraph.subgraph(adj_v).number_of_edges()
                if numadj >= 3:
                    corners1.add(v)
            
            corners2 = set()
            vertices2 = set(chain.from_iterable(edges2))
            for v in vertices2:
                adj_v = set(self.vertexgraph.node[v].keys())
                numadj = self.facegraph.subgraph(adj_v).number_of_edges()
                if numadj >= 3:
                    corners2.add(v)
            
            combined_vertices = set(chain.from_iterable(combined_edges))
            newcorners = set()
            faces_sharing_vert = set()
            for v in combined_vertices:
                adj_v = set(self.vertexgraph.node[v].keys())
                faces_sharing_vert.update(adj_v)
                numadj = self.facegraph.subgraph(adj_v).number_of_edges()
                if face1 in adj_v and face2 in adj_v:
                    numadj -= 1
                if numadj >= 3:
                    newcorners.add(v)
            faces_sharing_vert.discard(face1)
            faces_sharing_vert.discard(face2)
            
            if len(newcorners) < 3 and (len(newcorners) < len(corners1) or len(newcorners) < len(corners2)):
                continue
            
            #cutoff value was chosen which seems to work well for most models
            logrel = math.log(1 + error) / math.log(1 + self.maxerror)
            if logrel > 0.90:
                break
            #print 'error', error, 'maxerror', maxerror, 'logrel', logrel, 'merged left', len(merge_priorities), 'numfaces', len(facegraph)
            
            newface = node_count
            node_count += 1
            
            edges_to_add = []
            
            invalidmerge = False
            for otherface in faces_sharing_vert:
                if otherface not in self.facegraph:
                    continue
                otheredges = self.facegraph.node[otherface]['edges']
                otherverts = set(chain.from_iterable(otheredges))
                commonverts = combined_vertices.intersection(otherverts)
                commonedges = combined_edges.intersection(otheredges)
                
                connected_components_graph = nx.from_edgelist(commonedges)
                connected_components_graph.add_nodes_from(commonverts)
    
                #invalid merge if border between merged face and neighbor is more than one connected component
                if nx.algorithms.components.connected.number_connected_components(connected_components_graph) != 1:
                    invalidmerge = True
                    break
                
                #if there are no common edges, it just means single vertices are shared, so don't need to check rest
                if len(commonedges) == 0:
                    continue
    
                vertices = set(chain.from_iterable(otheredges))
                othernewcorners = set()
                otherprevcorners = set()
                for v in vertices:
                    adj_v = set(self.vertexgraph.node[v].keys())
                    numadj = self.facegraph.subgraph(adj_v).number_of_edges()
                    if numadj >= 3:
                        otherprevcorners.add(v)
                    if face1 in adj_v and face2 in adj_v:
                        numadj -= 1
                    if numadj >= 3:
                        othernewcorners.add(v)
                
                #invalid merge if neighbor would have less than 3 corners
                if len(othernewcorners) < 3 and len(othernewcorners) < len(otherprevcorners):
                    invalidmerge = True
                    break
                
                edges_to_add.append((newface, otherface))
    
            if invalidmerge:
                continue
            
            #only add edges to neighbors that are already neighbors
            valid_neighbors = set(self.facegraph.neighbors(face1))
            valid_neighbors.update(set(self.facegraph.neighbors(face2)))
            edges_to_add = [e for e in edges_to_add if (e[1] in valid_neighbors)]
            
            combined_tris = self.facegraph.node[face1]['tris'] + self.facegraph.node[face2]['tris']
            self.facegraph.add_node(newface, tris=combined_tris, edges=combined_edges)        
            self.facegraph.add_edges_from(edges_to_add)
            
            adj_faces = set(self.facegraph.neighbors(face1))
            adj_faces = adj_faces.union(set(self.facegraph.neighbors(face2)))
            adj_faces.remove(face1)
            adj_faces.remove(face2)
            for otherface in adj_faces:
                otheredges = self.facegraph.node[otherface]['edges']
                merged = numpy.array(list(combined_edges.symmetric_difference(otheredges)))
                if len(merged) > 0:
                    error = calcPerimeter(self.all_vertices[merged])**2
                    error += calcFitError(self.all_vertices[merged.flatten()])
                    if error > self.maxerror: self.maxerror = error
                    heapq.heappush(self.merge_priorities, (error, (newface, otherface)))
    
            for v in combined_vertices:
                if face1 in self.vertexgraph.node[v]:
                    del self.vertexgraph.node[v][face1]
                if face2 in self.vertexgraph.node[v]:
                    del self.vertexgraph.node[v][face2]
                self.vertexgraph.node[v][newface] = True
    
            self.facegraph.remove_node(face1)
            self.facegraph.remove_node(face2)
    
        self.end_operation()

    def update_corners(self, enforce=False):
        self.begin_operation('Updating corners...')
        for face, facedata in self.facegraph.nodes_iter(data=True):
            edges = facedata['edges']
            vertices = set(chain.from_iterable(edges))
            corners = set((v for v in vertices if self.facegraph.subgraph(self.vertexgraph.node[v].keys()).number_of_edges() >= 3))
            self.facegraph.node[face]['corners'] = corners
            
        if enforce:
            for (face1, face2) in self.facegraph.edges_iter():
                edges1 = self.facegraph.node[face1]['edges']
                edges2 = self.facegraph.node[face2]['edges']
                shared_edges = edges1.intersection(edges2)
                
                if len(self.invalid_edges) > 0 and len(shared_edges.intersection(self.invalid_edges)) > 0:
                    continue
                
                shared_vertices = set(chain.from_iterable(shared_edges))
                corners1 = self.facegraph.node[face1]['corners']
                corners2 = self.facegraph.node[face2]['corners']
                combined_corners = corners1.intersection(corners2).intersection(shared_vertices)
                
                giveup = False
                if len(combined_corners) == 1:
                    pt2edge = {}
                    for src, dest in shared_edges:
                        srclist = pt2edge.get(src, [])
                        srclist.append((src, dest))
                        pt2edge[src] = srclist
                        dstlist = pt2edge.get(dest, [])
                        dstlist.append((src, dest))
                        pt2edge[dest] = dstlist
                    start_path = combined_corners.pop()
                    curpt = start_path
                    shared_path = []
                    while curpt in pt2edge:
                        edge = pt2edge[curpt][0]
                        sanedge = edge
                        if edge[0] != curpt:
                            sanedge = (edge[1], edge[0])
                        nextpt = sanedge[1]
                        shared_path.append(sanedge)
                        nextopts = pt2edge[nextpt]
                        if edge not in nextopts:
                            giveup = True
                            break
                        nextopts.remove(edge)
                        if len(nextopts) > 0:
                            pt2edge[nextpt] = nextopts
                        else:
                            del pt2edge[nextpt]
                        curpt = nextpt
                    
                    if giveup:
                        continue
                    
                    end_path = shared_path[-1][1]
                    
                    self.facegraph.node[face1]['corners'].add(end_path)
                    self.facegraph.node[face2]['corners'].add(end_path)
            
        self.end_operation()

    def calc_edge_length(self):
        self.begin_operation('Computing distance between points')
        for v1, v2 in self.vertexgraph.edges_iter():
            self.vertexgraph.add_edge(v1, v2, distance=v3dist(self.all_vertices[v1],self.all_vertices[v2]))
        self.end_operation()

    def straighten_chart_boundaries(self):
        self.begin_operation('(Step 1 of 7) Straightening chart boundaries...')
        for (face1, face2) in self.facegraph.edges_iter():
            
            #can't straighten the border of a single triangle
            tris1 = self.facegraph.node[face1]['tris']
            tris2 = self.facegraph.node[face2]['tris']
            if len(tris1) <= 1 or len(tris2) <= 1:
                continue
            
            edges1 = self.facegraph.node[face1]['edges']
            edges2 = self.facegraph.node[face2]['edges']
            shared_edges = edges1.intersection(edges2)
            
            if len(self.invalid_edges) > 0 and len(shared_edges.intersection(self.invalid_edges)) > 0:
                continue
            
            #dont bother trying to straighten a single edge
            if len(shared_edges) == 1:
                continue
            
            shared_vertices = set(chain.from_iterable(shared_edges))
            corners1 = self.facegraph.node[face1]['corners']
            corners2 = self.facegraph.node[face2]['corners']
            combined_corners = corners1.intersection(corners2).intersection(shared_vertices)
            
            if len(combined_corners) < 1 or len(combined_corners) > 2:
                print 'warn combined corners wrong', len(combined_corners)
                #print 'shared edges', shared_edges
                #fakegraph = nx.Graph()
                #fakegraph.add_node(-1, tris=tris1)
                #fakegraph.add_node(-2, tris=tris2)
                #cur = 0
                #for f in self.facegraph.neighbors(face1):
                #    fakegraph.add_node(cur, tris=self.facegraph.node[f]['tris'])
                #    cur += 1
                #for f in self.facegraph.neighbors(face2):
                #    fakegraph.add_node(cur, tris=self.facegraph.node[f]['tris'])
                #    cur += 1
                #renderCharts(fakegraph, self.all_vertices, self.all_vert_indices, lineset=[shared_edges])
                
                continue
            
            giveup = False
            if len(combined_corners) == 2:
                start_path, end_path = combined_corners
            elif len(combined_corners) == 1:
                pt2edge = {}
                for src, dest in shared_edges:
                    srclist = pt2edge.get(src, [])
                    srclist.append((src, dest))
                    pt2edge[src] = srclist
                    dstlist = pt2edge.get(dest, [])
                    dstlist.append((src, dest))
                    pt2edge[dest] = dstlist
                start_path = combined_corners.pop()
                curpt = start_path
                shared_path = []
                while curpt in pt2edge:
                    edge = pt2edge[curpt][0]
                    sanedge = edge
                    if edge[0] != curpt:
                        sanedge = (edge[1], edge[0])
                    nextpt = sanedge[1]
                    shared_path.append(sanedge)
                    nextopts = pt2edge[nextpt]
                    if edge not in nextopts:
                        print 'warn corner1 bad'
                        giveup = True
                        break
                    nextopts.remove(edge)
                    if len(nextopts) > 0:
                        pt2edge[nextpt] = nextopts
                    else:
                        del pt2edge[nextpt]
                    curpt = nextpt
                
                end_path = shared_path[-1][1] 
            if giveup:
                continue
            
            edges1 = edges1.symmetric_difference(shared_edges)
            edges2 = edges2.symmetric_difference(shared_edges)
            all_verts1 = set(chain.from_iterable(self.all_vert_indices[tris1]))
            all_verts2 = set(chain.from_iterable(self.all_vert_indices[tris2]))
            stop_nodes = all_verts1.intersection(all_verts2).difference(shared_vertices)
            stop_nodes = stop_nodes.union(set(chain.from_iterable(edges1))).union(set(chain.from_iterable(edges2)))
            constrained_set = all_verts1.union(all_verts2)
            
            try:
                straightened_path = astar_path(self.vertexgraph, start_path, end_path,
                                               heuristic=lambda x,y: v3dist(self.all_vertices[x], self.all_vertices[y]),
                                               weight='distance', subset=constrained_set, exclude=stop_nodes)
            except nx.exception.NetworkXNoPath:
                print 'warn no path'
                continue
            
            # if we already have the shortest path, nothing to do
            if set(shared_vertices) == set(straightened_path):
                continue
            
            new_combined_edges = []
            for i in range(len(straightened_path)-1):
                new_combined_edges.append(tuple(sorted((straightened_path[i], straightened_path[i+1]))))
            new_combined_edges = set(new_combined_edges)
            new_edges1 = edges1.symmetric_difference(new_combined_edges)
            new_edges2 = edges2.symmetric_difference(new_combined_edges)
            
            # This can happen if the shortest path actually encompasses
            # the smaller face, but this would be equivalent to merging the
            # two faces. If we didn't merge these two in the previous step,
            # it was because the cost was too high or it would violate one of
            # the constraints, so just ignore this 
            if len(new_edges1) == 0 or len(new_edges2) == 0:
                print 'warn new edges 0'
                continue
            
            boundary1 = set(chain.from_iterable(new_edges1))
            boundary2 = set(chain.from_iterable(new_edges2))
            boundary = boundary1.union(boundary2).union(straightened_path)
            
            vertexset1 = boundary1.difference(straightened_path)
            vertexset2 = boundary2.difference(straightened_path)
            
            allin1 = list(dfs_interior_nodes(self.vertexgraph,
                                             starting=vertexset1,
                                             boundary=boundary,
                                             subset=constrained_set.difference(boundary2)))
            allin2 = list(dfs_interior_nodes(self.vertexgraph,
                                             starting=vertexset2,
                                             boundary=boundary,
                                             subset=constrained_set.difference(boundary1)))
            
            vertexset1 = set(allin1).union(vertexset1).union(straightened_path)
            vertexset2 = set(allin2).union(vertexset2).union(straightened_path)
            combined_tris = tris1 + tris2
            tris1 = []
            tris2 = []
            trisneither = []
            for tri in combined_tris:
                trivals = self.all_vert_indices[tri]
                if trivals[0] in vertexset1 and trivals[1] in vertexset1 and trivals[2] in vertexset1:
                    tris1.append(tri)
                elif trivals[0] in vertexset2 and trivals[1] in vertexset2 and trivals[2] in vertexset2:
                    tris2.append(tri)
                else:
                    trisneither.append(tri)
            
            #this can happen if the straightened path cuts off another face's edges
            if len(trisneither) != 0:
                print 'warn trisneither'
                continue
            
            # This can happen if the shortest path actually encompasses
            # the smaller face, but this would be equivalent to merging the
            # two faces. If we didn't merge these two in the previous step,
            # it was because the cost was too high or it would violate one of
            # the constraints, so just ignore this 
            if len(tris1) == 0 or len(tris2) == 0:
                print 'warn one was 0'
                continue
            
            #this can happen if the straightened path cuts off another face's edges
            if len(tris1) + len(tris2) != len(combined_tris):
                print 'warn cutoff'
                continue
    
            new_edges1 = new_edges1.union(new_combined_edges)
            new_edges2 = new_edges2.union(new_combined_edges)
    
            #if we stole edges from one face to the other, fix it
            for otherface in self.facegraph.neighbors_iter(face1):
                if otherface == face2:
                    continue
                otheredges = self.facegraph.node[otherface]['edges']
                face1otheredges = otheredges.intersection(new_edges1)
                if len(face1otheredges) == 0:
                    print 'warn otherone subsumed'
            for otherface in self.facegraph.neighbors_iter(face2):
                if otherface == face1:
                    continue
                otheredges = self.facegraph.node[otherface]['edges']
                face2otheredges = otheredges.intersection(new_edges2)
                if len(face2otheredges) == 0:
                    print 'warn otherone subsumed'
                
            #update adjaceny in vertex graph for swapped
            orig_verts1 = set(chain.from_iterable(self.facegraph.node[face1]['edges']))
            orig_verts2 = set(chain.from_iterable(self.facegraph.node[face2]['edges']))
            new_verts1 = set(chain.from_iterable(new_edges1))
            new_verts2 = set(chain.from_iterable(new_edges2))
            
            for v1lost in orig_verts1.difference(new_verts1):
                del self.vertexgraph.node[v1lost][face1]
            for v2lost in orig_verts2.difference(new_verts2):
                del self.vertexgraph.node[v2lost][face2]
            for v1added in new_verts1.difference(orig_verts1):
                self.vertexgraph.node[v1added][face1] = True
            for v2added in new_verts2.difference(orig_verts2):
                self.vertexgraph.node[v2added][face2] = True
                
            self.facegraph.add_node(face1, tris=tris1, edges=new_edges1)
            self.facegraph.add_node(face2, tris=tris2, edges=new_edges2)
            
        self.end_operation()

    def create_initial_parameterizations(self):
        self.begin_operation('(Step 2 of 7) Forming initial chart parameterizations...')
        new_uv_indices = numpy.zeros(shape=(len(self.all_vert_indices), 3), dtype=numpy.int32)
        new_uvs = []
        new_uvs_offset = 0
        
        for (face, facedata) in self.facegraph.nodes_iter(data=True):
            border_edges = facedata['edges']
            chart_tris = self.all_vert_indices[facedata['tris']]
    
            unique_verts = set(chain.from_iterable(chart_tris))
            border_verts = set(chain.from_iterable(border_edges))
            interior_verts = list(unique_verts.difference(border_verts))
                    
            bordergraph = nx.from_edgelist(border_edges)
            bigcycle = list(super_cycle(bordergraph))
            boundary_path = []
            for i in range(len(bigcycle)-1):
                boundary_path.append((bigcycle[i], bigcycle[i+1]))
            boundary_path.append((bigcycle[len(bigcycle)-1], bigcycle[0]))
            assert(len(boundary_path) == len(border_edges))
    
            total_dist = 0
            for (v1, v2) in boundary_path:
                total_dist += v3dist(self.all_vertices[v1], self.all_vertices[v2])
            
            vert2uv = {}
            curangle = 0
            for edge in boundary_path:
                angle = v3dist(self.all_vertices[edge[0]], self.all_vertices[edge[1]]) / total_dist
                curangle += angle * 2 * math.pi
                x, y = (math.sin(curangle) + 1) / 2.0, (math.cos(curangle) + 1.0) / 2.0
                vert2uv[edge[0]] = (x,y)
            
            if len(interior_verts) > 0:
            
                vert2idx = {}
                for i, v in enumerate(interior_verts):
                    vert2idx[v] = i
                
                A = numpy.zeros(shape=(len(interior_verts), len(interior_verts)), dtype=numpy.float32)
                Bu = numpy.zeros(len(interior_verts))
                Bv = numpy.zeros(len(interior_verts))
                sumu = numpy.zeros(len(interior_verts))
                
                for edge in self.vertexgraph.subgraph(unique_verts).edges_iter():
                    v1, v2 = edge
                    if v1 in border_verts and v2 in border_verts:
                        continue
                    
                    edgelen = v3dist(self.all_vertices[v1], self.all_vertices[v2])
                    if v1 in border_verts:
                        Bu[vert2idx[v2]] += edgelen * vert2uv[v1][0]
                        Bv[vert2idx[v2]] += edgelen * vert2uv[v1][1]
                        sumu[vert2idx[v2]] += edgelen
                    elif v2 in border_verts:
                        Bu[vert2idx[v1]] += edgelen * vert2uv[v2][0]
                        Bv[vert2idx[v1]] += edgelen * vert2uv[v2][1]
                        sumu[vert2idx[v1]] += edgelen
                    else:
                        A[vert2idx[v1]][vert2idx[v2]] = -1 * edgelen
                        A[vert2idx[v2]][vert2idx[v1]] = -1 * edgelen
                        sumu[vert2idx[v1]] += edgelen
                        sumu[vert2idx[v2]] += edgelen
                
                Bu.shape = (len(Bu), 1)
                Bv.shape = (len(Bv), 1)
                sumu.shape = (len(sumu), 1)
                
                A /= sumu
                Bu /= sumu
                Bv /= sumu
                try: numpy.fill_diagonal(A, 1)
                except AttributeError:
                    for i in xrange(len(A)):
                        A[i][i] = 1
                
                interior_us = numpy.linalg.solve(A, Bu)
                interior_vs = numpy.linalg.solve(A, Bv)
                for (i, (u, v)) in enumerate(zip(interior_us, interior_vs)):
                    vert2uv[interior_verts[i]] = (u[0], v[0])
                    
            new_uvs.append(vert2uv.values())
            newvert2idx = dict(zip(vert2uv.keys(), range(new_uvs_offset, new_uvs_offset + len(vert2uv))))
            for tri in facedata['tris']:
                for i, v in enumerate(self.all_vert_indices[tri]):
                    new_uv_indices[tri][i] = newvert2idx[v]
            new_uvs_offset += len(vert2uv)
            self.facegraph.node[face]['vert2uvidx'] = newvert2idx
            
        self.new_uvs = numpy.concatenate(new_uvs)
        self.new_uv_indices = new_uv_indices
        
        self.end_operation()

    def optimize_chart_parameterizations(self):

        self.begin_operation('(Step 2 of 7) Optimizing chart parameterizations...')
        total_L2 = 0
        for (face, facedata) in self.facegraph.nodes_iter(data=True):
            border_edges = facedata['edges']
            newvert2idx = self.facegraph.node[face]['vert2uvidx']
            chart_tris = self.all_vert_indices[facedata['tris']]
            tri_3d = self.all_vertices[chart_tris]
            tri_2d = self.new_uvs[self.new_uv_indices[facedata['tris']]]
            
            unique_verts, index_map = numpy.unique(chart_tris, return_inverse=True)
            index_map.shape = chart_tris.shape
            border_verts = set(chain.from_iterable(border_edges))
            
            for iteration in range(1, 11):
            
                L2 = stretch_metric(tri_3d, tri_2d)
                neighborhood_stretch = numpy.zeros(unique_verts.shape, dtype=numpy.float32)
                neighborhood_stretch[index_map[:,0]] += L2
                neighborhood_stretch[index_map[:,1]] += L2
                neighborhood_stretch[index_map[:,2]] += L2
                vert_stretch_heap = zip(-1 * neighborhood_stretch, unique_verts)
                heapq.heapify(vert_stretch_heap)
            
                while len(vert_stretch_heap) > 0:
                    stretch, vert = heapq.heappop(vert_stretch_heap)
                    if vert in border_verts:
                        continue
                    
                    ucoord, vcoord = self.new_uvs[newvert2idx[vert]]
                    
                    vert_tri1 = chart_tris[:,0] == vert
                    vert_tri2 = chart_tris[:,1] == vert
                    vert_tri3 = chart_tris[:,2] == vert
                    neighborhood_selector = numpy.logical_or(numpy.logical_or(vert_tri1, vert_tri2), vert_tri3)
                    
                    neighborhood_tri3d = tri_3d[neighborhood_selector]
                    neighborhood_tri_2d = tri_2d[neighborhood_selector]
                    neighborhood_L2, origA2d = stretch_metric(neighborhood_tri3d, neighborhood_tri_2d, return_A2d=True, normalize=True)
                    
                    randangle = random.uniform(0, 2 * math.pi)
                    randslope = math.tan(randangle)
        
                    # y - y1 = m(x - x1)
                    def yfromx(x):
                        return randslope * (x - ucoord) + vcoord
                    def xfromy(y):
                        return ((y - vcoord) / randslope) + ucoord
                    
                    xintercept0 = yfromx(0)
                    minx = 0.0 if 0 < xintercept0 < 1 else min(xfromy(0), xfromy(1))
                    xintercept1 = yfromx(1)
                    maxx = 1.0 if 0 < xintercept1 < 1 else max(xfromy(0), xfromy(1))
                    minx, maxx = tuple(sorted([minx, maxx]))
                    
                    assert(0 <= minx <= maxx <= 1)
                    
                    rangesize = (maxx-minx) / iteration
                    rangemin = ucoord - rangesize / 2.0
                    rangemax = ucoord + rangesize / 2.0
                    if rangemax > maxx:
                        rangemin -= (maxx-rangemax)
                        rangemax -= (maxx-rangemax)
                    if rangemin < minx:
                        rangemax += (minx-rangemin)
                        rangemin += (minx-rangemin)
                    if rangemax > maxx: rangemax = maxx
                    if rangemin < minx: rangemin = minx
        
                    assert(0 <= rangemin <= rangemax <= 1)
                    
                    samples = 10.0
                    step = (rangemax - rangemin) / samples
                    bestu, bestv = ucoord, vcoord
                    bestL2 = neighborhood_L2
                    for xval in numpy.arange(rangemin, rangemax, step):
                        yval = yfromx(xval)
        
                        tri_2d[vert_tri1,0,0] = xval
                        tri_2d[vert_tri2,1,0] = xval
                        tri_2d[vert_tri3,2,0] = xval
                        tri_2d[vert_tri1,0,1] = yval
                        tri_2d[vert_tri2,1,1] = yval
                        tri_2d[vert_tri3,2,1] = yval
                        
                        neighborhood_tri_2d = tri_2d[neighborhood_selector]
                        neighborhood_L2 = stretch_metric(neighborhood_tri3d, neighborhood_tri_2d, flippedCheck=origA2d, normalize=True)
                        if neighborhood_L2 < bestL2:
                            bestL2 = neighborhood_L2
                            bestu, bestv = xval, yval
        
                    tri_2d[vert_tri1,0,0] = bestu
                    tri_2d[vert_tri2,1,0] = bestu
                    tri_2d[vert_tri3,2,0] = bestu
                    tri_2d[vert_tri1,0,1] = bestv
                    tri_2d[vert_tri2,1,1] = bestv
                    tri_2d[vert_tri3,2,1] = bestv
                    
                    ucoord, vcoord = bestu, bestv
        
                    self.new_uvs[newvert2idx[vert],0] = ucoord
                    self.new_uvs[newvert2idx[vert],1] = vcoord
            
            chart_L2 = stretch_metric(tri_3d, tri_2d, normalize=True)
            total_L2 += chart_L2
            self.facegraph.node[face]['L2'] = chart_L2
            self.total_L2 = total_L2
            
        self.end_operation()

    def resize_charts(self):
        self.begin_operation('(Step 3 of 7) Creating and resizing charts...')

        #first create the PIL charts
        assert(len(self.mesh.images) == 1)
        origtexture = self.mesh.images[0].pilimage
        TEXTURE_DIMENSION = 1024
        TEXTURE_SIZE = TEXTURE_DIMENSION * TEXTURE_DIMENSION
        rp = RectPack(TEXTURE_DIMENSION*2, TEXTURE_DIMENSION*2)
        self.chart_ims = {}
        for face, facedata in self.facegraph.nodes_iter(data=True):
            relsize = int(math.sqrt((facedata['L2'] / self.total_L2) * TEXTURE_SIZE))
            
            #round to power of 2
            relsize = int(math.pow(2, round(math.log(relsize, 2))))
            self.facegraph.node[face]['chartsize'] = relsize
            
            chartim = Image.new('RGB', (relsize, relsize))
            for tri in facedata['tris']:
                prevuvs = self.all_orig_uvs[self.all_orig_uv_indices[tri]]
                newuvs = self.new_uvs[self.new_uv_indices[tri]]
                prevu = prevuvs[:,0] * origtexture.size[0]
                prevv = (1.0-prevuvs[:,1]) * origtexture.size[1]
                newu = (newuvs[:,0] * (relsize-0.5))
                newv = ((1.0-newuvs[:,1]) * (relsize-0.5))
                prevtri = [(prevu[0], prevv[0]), (prevu[1], prevv[1]), (prevu[2], prevv[2])]
                newtri = [(newu[0], newv[0]), (newu[1], newv[1]), (newu[2], newv[2])]
                transformblit(prevtri, newtri, origtexture, chartim)
            self.chart_ims[face] = chartim
            rp.addRectangle(face, relsize, relsize)
        assert(rp.pack())
        self.chart_packing = rp
        
        #now resize the uvs according to chart size
        for face, facedata in self.facegraph.nodes_iter(data=True):
            relsize = self.facegraph.node[face]['chartsize']
            toupdate = self.new_uv_indices[facedata['tris']]
            self.new_uvs[toupdate, 0] *= relsize-0.5
            self.new_uvs[toupdate, 1] *= relsize-0.5
        
        self.end_operation()

    def normalize_uvs(self):
        self.begin_operation('Normalizing texture coordinates...')

        for face, facedata in self.facegraph.nodes_iter(data=True):
            relsize = self.facegraph.node[face]['chartsize']
            
            toupdate = self.new_uv_indices[facedata['tris']]

            self.new_uvs[toupdate, 0] /= relsize-0.5
            self.new_uvs[toupdate, 1] /= relsize-0.5
        
        self.end_operation()

    def initialize_simplification_errors(self):
        self.begin_operation('Calculationg priority queue for initial edge contractions...')
        
        self.all_corners = set()
        self.all_edge_verts = set()
        self.tri2face = {}
        for face, facedata in self.facegraph.nodes_iter(data=True):
            self.all_corners = self.all_corners.union(facedata['corners'])
            self.all_edge_verts = self.all_edge_verts.union(set(chain.from_iterable(facedata['edges'])))
            for tri in facedata['tris']:
                self.tri2face[tri] = face

        self.vertexgraph.add_nodes_from(( (i, {'tris':set()}) for i in xrange(len(self.all_vertices))))
        for i, (v1,v2,v3) in enumerate(self.all_vert_indices):
            self.vertexgraph.node[v1]['tris'].add(i)
            self.vertexgraph.node[v2]['tris'].add(i)
            self.vertexgraph.node[v3]['tris'].add(i)

        self.contraction_priorities = []

        for (vv1, vv2) in self.vertexgraph.edges_iter():
            for (v1, v2) in ((vv1,vv2),(vv2,vv1)):
                #considering (v1,v2) -> v1
                
                #can't remove corners
                if v2 in self.all_corners:
                    continue
                
                #need to preserve boundary straightness
                if v2 in self.all_edge_verts and v1 not in self.all_edge_verts:
                    continue

                v2tris = list(self.vertexgraph.node[v2]['tris'])
                v2tri_idx = self.all_vert_indices[v2tris]
                
                moved = set()
                degenerate = set()
                for t2, t2_idx in izip(v2tris, v2tri_idx):
                    if v1 in t2_idx:
                        degenerate.add(t2)
                    else:
                        moved.add(t2)
                
                invalid_contraction = False
                total_texture_diff = 0
                for moved_tri_idx in moved:
                    facefrom = self.tri2face[moved_tri_idx]
                    face_vert2uvidx = self.facegraph.node[facefrom]['vert2uvidx']
                    
                    moved_vert_idx = self.all_vert_indices[moved_tri_idx]
                    other_pts = self.new_uvs[[ face_vert2uvidx[m]
                                 for m in moved_vert_idx if m != v2 ]]
                    
                    #m = (y2-y1)/(x2-x1)
                    slope = (other_pts[1][1] - other_pts[0][1]) / (other_pts[1][0] - other_pts[0][0])
                    #y = mx + b
                    #b = y-mx
                    yint = other_pts[0][1] - slope * other_pts[0][0]
                
                    #don't want to cross chart boundaries
                    if v1 not in face_vert2uvidx:
                        invalid_contraction = True
                        break
                    
                    v2_pt = self.new_uvs[face_vert2uvidx[v2]]
                    v1_pt = self.new_uvs[face_vert2uvidx[v1]]
                    v2_atline = slope * v2_pt[0] + yint
                    v1_atline = slope * v1_pt[0] + yint
                    
                    #don't want to flip the triangle in the parametric domain
                    if v2_atline > 0 and not(v1_atline > 0) or \
                        v2_atline < 0 and not(v1_atline < 0):
                        invalid_contraction = True
                        break
                    
                    texture_diff = math.sqrt((v1_pt[0]-v2_pt[0])**2 + (v1_pt[1]-v2_pt[1])**2)
                    total_texture_diff += texture_diff
                
                if invalid_contraction:
                    continue
                
                self.contraction_priorities.append((total_texture_diff, (v1, v2)))
                

        heapq.heapify(self.contraction_priorities)

        self.end_operation()

    def simplify_mesh(self):
        self.begin_operation('(Step 4 of 7) Simplifying...')
        
        self.tris_left = set(xrange(len(self.all_vert_indices)))
        
        while len(self.contraction_priorities) > 0:
            (texture_diff, (v1, v2)) = heapq.heappop(self.contraction_priorities)
            
            #considering (v1,v2) -> v1
            
            #check of one of these vertices was already contracted
            if v1 not in self.vertexgraph or v2 not in self.vertexgraph:
                continue

            #print 'texture_diff', texture_diff, 'v1', v1, 'v2', v2, 'numverts', len(self.vertexgraph), 'numfaces', len(self.tris_left), 'contractions left', len(self.contraction_priorities)
            
            v2tris = list(self.vertexgraph.node[v2]['tris'])
            v1tris = self.vertexgraph.node[v1]['tris']
            v2tri_idx = self.all_vert_indices[v2tris]
            
            degenerate = set()
            new_contractions = set()
            for t2, t2_idx in izip(v2tris, v2tri_idx):
                if v1 in t2_idx:
                    degenerate.add(t2)
                else:
                    #these are triangles that have a vertex moving from v2 to v1
                    where_v2 = numpy.where(t2_idx == v2)[0][0]
                    where_not_v2 = numpy.where(t2_idx != v2)
                    other1 = where_not_v2[0][0]
                    other2 = where_not_v2[0][1]
                    
                    #update vert and uv index values from v2 to v1
                    self.all_vert_indices[t2][where_v2] = v1
                    facefrom = self.tri2face[t2]
                    face_vert2uvidx = self.facegraph.node[facefrom]['vert2uvidx']
                    self.new_uv_indices[t2][where_v2] = face_vert2uvidx[v1]
                    
                    #add tri to v1's list now that we moved it
                    self.vertexgraph.node[v1]['tris'].add(t2)
                    
                    #try to find a triangle in the same chart as v2 that contains v1
                    # so we can copy its normal value
                    copy_tri_v1 = None
                    where_v1 = None
                    for facetri in self.facegraph.node[facefrom]['tris']:
                        if facetri in v1tris:
                            facetri_idx = self.all_vert_indices[facetri]
                            where_v1 = numpy.where(facetri_idx == v1)[0][0]
                            copy_tri_v1 = facetri
                            break
                    assert(copy_tri_v1 is not None)
                    self.all_normal_indices[t2][where_v2] = self.all_normal_indices[copy_tri_v1][where_v1]
                    
                    #add new candidate merges
                    new_contractions.add((t2_idx[other1], v1))
                    new_contractions.add((t2_idx[other2], v1))
                    new_contractions.add((v1, t2_idx[other1]))
                    new_contractions.add((v1, t2_idx[other2]))
            
            #remove the degenerate triangle from the triangle list of other vertices in the triangle
            for tri in degenerate:
                for v in self.all_vert_indices[tri]:
                    self.vertexgraph.node[v]['tris'].discard(tri)
            
            #discard the degenerate triangles from the total list of tris
            self.tris_left.difference_update(degenerate)
            
            #remove vertex from graph
            self.vertexgraph.remove_node(v2)
            
            #now update priority list with new valid contractions
            for (v1, v2) in new_contractions:
                #considering (v1,v2) -> v1
                
                #can't remove corners
                if v2 in self.all_corners:
                    continue
                
                #need to preserve boundary straightness
                if v2 in self.all_edge_verts and v1 not in self.all_edge_verts:
                    continue

                v2tris = list(self.vertexgraph.node[v2]['tris'])
                v2tri_idx = self.all_vert_indices[v2tris]
                
                moved = set()
                degenerate = set()
                for t2, t2_idx in izip(v2tris, v2tri_idx):
                    if v1 in t2_idx:
                        degenerate.add(t2)
                    else:
                        moved.add(t2)
                
                invalid_contraction = False
                total_texture_diff = 0
                for moved_tri_idx in moved:
                    facefrom = self.tri2face[moved_tri_idx]
                    face_vert2uvidx = self.facegraph.node[facefrom]['vert2uvidx']
                    
                    moved_vert_idx = self.all_vert_indices[moved_tri_idx]
                    other_pts = self.new_uvs[[ face_vert2uvidx[m]
                                 for m in moved_vert_idx if m != v2 ]]
                    
                    #m = (y2-y1)/(x2-x1)
                    slope = (other_pts[1][1] - other_pts[0][1]) / (other_pts[1][0] - other_pts[0][0])
                    #y = mx + b
                    #b = y-mx
                    yint = other_pts[0][1] - slope * other_pts[0][0]
                
                    #don't want to cross chart boundaries
                    if v1 not in face_vert2uvidx:
                        invalid_contraction = True
                        break
                    
                    v2_pt = self.new_uvs[face_vert2uvidx[v2]]
                    v1_pt = self.new_uvs[face_vert2uvidx[v1]]
                    v2_atline = slope * v2_pt[0] + yint
                    v1_atline = slope * v1_pt[0] + yint
                    
                    #don't want to flip the triangle in the parametric domain
                    if v2_atline > 0 and not(v1_atline > 0) or \
                        v2_atline < 0 and not(v1_atline < 0):
                        invalid_contraction = True
                        break
                    
                    texture_diff = math.sqrt((v1_pt[0]-v2_pt[0])**2 + (v1_pt[1]-v2_pt[1])**2)
                    total_texture_diff += texture_diff
                
                if invalid_contraction:
                    continue
                
                heapq.heappush(self.contraction_priorities, (total_texture_diff, (v1, v2)))
            
        self.end_operation()

    def pack_charts(self):
        self.begin_operation('(Step 6 of 7) Creating and packing charts into atlas...')
        self.atlasimg = Image.new('RGB', (self.chart_packing.width, self.chart_packing.height))
        for face, chartim in self.chart_ims.iteritems():
            
            x,y,w,h = self.chart_packing.getPlacement(face)
            self.atlasimg.paste(chartim, (x,y,x+w,y+h))
            x,y,w,h,width,height = (float(i) for i in (x,y,w,h,self.chart_packing.width,self.chart_packing.height))
    
            chart_tris = [f for f in self.facegraph.node[face]['tris'] if f in self.tris_left]
            chart_idx = numpy.unique(self.new_uv_indices[chart_tris])
    
            #this rescales the texcoords to map to the new atlas location
            self.new_uvs[chart_idx,0] = (self.new_uvs[chart_idx,0] * (w-0.5) + x) / width
            self.new_uvs[chart_idx,1] = 1.0 - (( (1.0-self.new_uvs[chart_idx,1]) * (h-0.5) + y ) / height)
        self.end_operation()

    def save_mesh(self):
        self.begin_operation('(Step 7 of 7) Saving mesh...')
        newmesh = collada.Collada()
        
        cimg = collada.material.CImage("sander-simplify-packed-atlas", "./atlas.jpg")
        imgout = StringIO()
        self.atlasimg.save(imgout, format="JPEG", quality=95, optimize=True)
        cimg.setData(imgout.getvalue())
        newmesh.images.append(cimg)
        
        surface = collada.material.Surface("sander-simplify-surface", cimg)
        sampler = collada.material.Sampler2D("sander-simplify-sampler", surface)
        map = collada.material.Map(sampler, "TEX0")
        effect = collada.material.Effect("sander-simplify-effect", [surface, sampler], "blinn", diffuse=map)
        newmesh.effects.append(effect)
        material = collada.material.Material("sander-simplify-material0", "sander-simplify-material", effect)
        newmesh.materials.append(material)
        
        vert_src = collada.source.FloatSource("sander-verts-array", self.all_vertices, ('X', 'Y', 'Z'))
        normal_src = collada.source.FloatSource("sander-normals-array", self.all_normals, ('X', 'Y', 'Z'))
        uv_src = collada.source.FloatSource("sander-uv-array", self.new_uvs, ('S', 'T'))
        geom = collada.geometry.Geometry(newmesh, "sander-geometry-0", "sander-mesh", [vert_src, normal_src, uv_src])
        
        input_list = collada.source.InputList()
        input_list.addInput(0, 'VERTEX', '#sander-verts-array')
        input_list.addInput(1, 'NORMAL', '#sander-normals-array')
        input_list.addInput(2, 'TEXCOORD', '#sander-uv-array')
        
        tris_left = list(self.tris_left)
        new_index = numpy.dstack((self.all_vert_indices[tris_left], self.all_normal_indices[tris_left], self.new_uv_indices[tris_left])).flatten()
        triset = geom.createTriangleSet(new_index, input_list, "materialref")
        geom.primitives.append(triset)
        newmesh.geometries.append(geom)
        
        matnode = collada.scene.MaterialNode("materialref", material, inputs=[('TEX0', 'TEXCOORD', '0')])
        geomnode = collada.scene.GeometryNode(geom, [matnode])
        node = collada.scene.Node("node0", children=[geomnode])
        
        myscene = collada.scene.Scene("myscene", [node])
        newmesh.scenes.append(myscene)
        newmesh.scene = myscene
        savezip = meshtool.filters.factory.getInstance('save_collada_zip')
        savezip.apply(newmesh, '/tmp/test.zip')
        self.end_operation()

    def simplify(self):
        self.uniqify_list()
        self.build_vertex_graph()
        self.build_face_graph()
        
        #renderCharts(self.facegraph, self.all_vertices, self.all_vert_indices)
        print 'number of faces =', len(self.facegraph)
        print 'connected components =', nx.algorithms.components.connected.number_connected_components(self.vertexgraph)
        
        self.initialize_chart_merge_errors()
        self.merge_charts()
        
        print 'number of charts =', len(self.facegraph)
        #renderCharts(self.facegraph, self.all_vertices, self.all_vert_indices)
        
        self.update_corners()
        self.calc_edge_length()
        self.straighten_chart_boundaries()
        
        #renderCharts(self.facegraph, self.all_vertices, self.all_vert_indices)
        
        self.create_initial_parameterizations()
        self.optimize_chart_parameterizations()
        self.resize_charts()
        self.update_corners(enforce=True)
        self.initialize_simplification_errors()
        self.simplify_mesh()
        self.normalize_uvs()
        self.pack_charts()
        self.save_mesh()

def FilterGenerator():
    class SandlerSimplificationFilter(OpFilter):
        def __init__(self):
            super(SandlerSimplificationFilter, self).__init__('sander_simplify', 'Simplifies the mesh based on sandler, et al. method.')
        def apply(self, mesh):
            s = SanderSimplify(mesh)
            s.simplify()
            return mesh
    return SandlerSimplificationFilter()
from meshtool.filters import factory
factory.register(FilterGenerator().name, FilterGenerator)
