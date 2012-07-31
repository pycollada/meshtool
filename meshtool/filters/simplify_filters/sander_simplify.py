from meshtool.args import FileArgument
from meshtool.filters.base_filters import SimplifyFilter
import inspect
import numpy
import networkx as nx
from itertools import chain, izip, combinations
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
try:
    from ipdb import launch_ipdb_on_exception
    USE_IPDB = True
except ImportError:
    USE_IPDB = False

from meshtool.util import Image, ImageDraw
from meshtool.filters.atlas_filters.rectpack import RectPack
from StringIO import StringIO
import meshtool.filters
import bisect

#after numpy 1.3, unique1d was renamed to unique
args, varargs, keywords, defaults = inspect.getargspec(numpy.unique)    
if 'return_inverse' not in args:
    numpy.unique = numpy.unique1d

# Import both cv (opencv's official python bindings)
# and pyopencv, a python binding to cv that makes it
# much easier to use. Mostly using pyopencv below, but
# in some cases the inpaint function fails using pyopencv
# so it gets used via the regular cv 
try: import pyopencv as pcv
except ImportError:
    pcv = None
try: import cv
except ImportError:
    cv = None

#Error threshold values, range 0-1
MERGE_ERROR_THRESHOLD = 0.91
SIMPLIFICATION_ERROR_THRESHOLD = 0.90

# if the mesh is less than this many triangles, don't make a progressive stream
TRIANGLE_MINIMUM = 10000
# keep trying to simplify if mesh is bigger than this, even if we think we should stop
TRIANGLE_MAXIMUM = 40000
# if the number of triangles in the progressive stream is less than this fraction of
# the total mesh, don't make a progressive stream
STREAM_THRESHOLD = 0.2

def timer():
    begintime = datetime.datetime.now()
    while True:
        curtime = datetime.datetime.now()
        yield (curtime-begintime)
        begintime = curtime

def seg_intersect(a1,a2, b1,b2):
    """line segment intersection using vectors
    see Computer Graphics by F.S. Hill
    
    line segment a given by endpoints a1, a2
    line segment b given by endpoints b1, b2
    
    returns point (c1,d1) where the two line
    segments intersect or None if lines don't intersect
    
    Original code from http://www.cs.mun.ca/~rod/2500/notes/numpy-arrays/numpy-arrays.html
    """
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    
    dap = numpy.copy(da)
    dap[0] = -da[1]
    dap[1] = da[0]
    
    denom = numpy.dot(dap, db)
    num = numpy.dot(dap, dp)
    intersect = (num / denom)*db + b1
    if intersect[0] > max(a1[0], a2[0]) or intersect[0] < min(a1[0], a2[0]) or \
        intersect[1] > max(a1[1], a2[1]) or intersect[1] < min(a1[1], a2[1]):
        return None
    return intersect
         
def calcPerimeter(pts):
    """Calculates the perimeter of an area by 
    summing the distance between the points in
    a set of edges"""
    dx = pts[:,0,0]-pts[:,1,0]
    dy = pts[:,0,1]-pts[:,1,1]
    dz = pts[:,0,2]-pts[:,1,2]
    return numpy.sum(numpy.sqrt(dx*dx + dy*dy + dz*dz))

def v2dist(pt1, pt2):
    """Calculates the distance between two 2d points element-wise
    along an array"""
    d = pt1 - pt2
    return math.sqrt(d[0]*d[0] + d[1]*d[1])
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
        A2d[A2d == numpy.inf] = 0
        sumA3d = numpy.sum(A3d)
        if flippedCheck is None and sumA3d == 0:
            L2 = 0
        else:
            A3d[L2 == numpy.inf] = numpy.inf
            L2 = numpy.sqrt(numpy.sum(L2*L2*A3d) / sumA3d) * numpy.sqrt(numpy.sum(numpy.abs(A2d)) / sumA3d)
        
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

def transformblit(src_tri, dst_tri, src_img, dst_img, alpha=255):
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

    try:
        A = numpy.linalg.solve(M, y)
    except numpy.linalg.LinAlgError:
        #can happen if the new triangle is a single point, so no need to paint it
        return

    transformed = src_img.transform((sizex, sizey), Image.AFFINE, A, Image.BICUBIC)
    
    mask = Image.new('1' if alpha == 255 else 'L', (sizex, sizey))
    maskdraw = ImageDraw.Draw(mask)
    maskdraw.polygon(((y11,y12), (y21,y22), (y31,y32)), outline=alpha, fill=alpha)

    dst_img.paste(transformed, (minx, miny), mask=mask)

def opencvblit(src_tri, dst_tri, src_cv_img, dst_pil_img):
    """Pastes a triangular region from one image
    into a triangular region in another image by
    using an affine transformation. Uses pyopencv
    for the source image and PIL for the destination
    image."""
    
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
    
    #if the destination size is 0, no point in performing the transformation
    # and opencv throws an exception that we can't catch anyway
    if sizex < 1 or sizey < 1:
        return
    
    src_vec = pcv.vector_Point2f([pcv.Point2f(float(x),float(y)) for (x,y) in src_tri])
    dst_vec = pcv.vector_Point2f([pcv.Point2f(float(x),float(y)) for (x,y) in [(y11,y12), (y21,y22), (y31,y32)]])
    
    A = pcv.getAffineTransform(src_vec, dst_vec)
    
    transformed_im = pcv.Mat()
    pcv.warpAffine(src_cv_img, transformed_im, A, pcv.Size(sizex, sizey), pcv.INTER_CUBIC, pcv.BORDER_WRAP)
    
    transformed_im = transformed_im.to_pil_image()
    mask = Image.new('1', (sizex, sizey))
    maskdraw = ImageDraw.Draw(mask)
    maskdraw.polygon(((y11,y12), (y21,y22), (y31,y32)), outline=255, fill=255)

    dst_pil_img.paste(transformed_im, (minx, miny), mask=mask)

def evalQuadric(A, b, c, pt):
    """Evaluates a quadric Q = (A,b,c) at the point pt"""
    return numpy.dot(pt,numpy.inner(A,pt)) + 2*numpy.dot(b,pt) + c

def quadricsForTriangles(tris):
    """Computes the quadric error matrix Q = (A,b,c)
    for each triangle. Also returns the area and normal
    for each triangle"""
    normal = numpy.cross( tris[::,1] - tris[::,0], tris[::,2] - tris[::,0] )
    collada.util.normalize_v3(normal)
    
    s1 = tris[::,1] - tris[::,0]
    s1 = array_dot(s1, s1)
    s2 = tris[::,2] - tris[::,0]
    s2 = array_dot(s2, s2)
    s3 = tris[::,2] - tris[::,1]
    s3 = array_dot(s3, s3)
    
    sp = (s1 + s2 + s3) / 2.0
    area = sp*(sp-s1)*(sp-s2)*(sp-s3)
    
    area = numpy.sqrt(numpy.abs(area))
    
    d = -array_mult(normal, tris[:,0])

    b = normal * (area*d)[:,numpy.newaxis]
    c = area*d*d
    
    A = numpy.dstack((normal[:,0][:,numpy.newaxis] * normal,
                       normal[:,1][:,numpy.newaxis] * normal,
                       normal[:,2][:,numpy.newaxis] * normal))
    A = area[:,numpy.newaxis,numpy.newaxis] * A
   
    return (A, b, c, area, normal)

def uniqify_multidim_indexes(sourcedata, indices, return_map=False):
    unique_data, index_map = numpy.unique(sourcedata.view([('',sourcedata.dtype)]*sourcedata.shape[1]), return_inverse=True)
    index_map = numpy.cast['int32'](index_map)
    if return_map:
        return unique_data.view(sourcedata.dtype).reshape(-1,sourcedata.shape[1]), index_map[indices], index_map
    return unique_data.view(sourcedata.dtype).reshape(-1,sourcedata.shape[1]), index_map[indices]

class STREAM_OP:
    OPERATION_BOUNDARY = 0
    INDEX_UPDATE = 1
    TRIANGLE_ADDITION = 2

class SanderSimplify(object):

    def __init__(self, mesh, pmbuf):
        self.mesh = mesh
        self.pmbuf = pmbuf
        
        self.all_vertices = []
        self.all_normals = []
        self.all_orig_uvs = []
        
        self.all_vert_indices = []
        self.all_normal_indices = []
        self.all_orig_uv_indices = []
        
        self.index_offset = 0
        self.vertex_offset = 0
        self.normal_offset = 0
        self.uv_offset = 0
        
        self.tri2material = []
        
        self.timer = timer()
        
        self.begin_operation('Building aggregated vertex and triangle list...')
        for boundgeom in chain(mesh.scene.objects('geometry'), mesh.scene.objects('controller')):
            if isinstance(boundgeom, collada.controller.BoundController):
                boundgeom = boundgeom.geometry
            for boundprim in boundgeom.primitives():
                if boundprim.vertex_index is None or len(boundprim.vertex_index) == 0:
                    continue

                #any triangles that have two identical vertices are useless
                bad_tris = (boundprim.vertex_index[:,0] == boundprim.vertex_index[:,1]) | \
                           (boundprim.vertex_index[:,1] == boundprim.vertex_index[:,2]) | \
                           (boundprim.vertex_index[:,0] == boundprim.vertex_index[:,2])

                self.all_vertices.append(boundprim.vertex)
                self.all_normals.append(boundprim.normal)
                self.all_vert_indices.append(numpy.delete(boundprim.vertex_index, numpy.where(bad_tris), axis=0) + self.vertex_offset)
                self.all_normal_indices.append(numpy.delete(boundprim.normal_index, numpy.where(bad_tris), axis=0) + self.normal_offset)
                self.vertex_offset += len(boundprim.vertex)
                self.normal_offset += len(boundprim.normal)
                
                if boundprim.texcoordset and len(boundprim.texcoordset) > 0:
                    
                    texsource = boundprim.texcoordset[0]
                    texindex = boundprim.texcoord_indexset[0]
                    texarray = texsource[texindex]
                    
                    if numpy.min(texarray) < 0.0 or numpy.max(texarray) > 1.0:
                    
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
                        
                        texarray.shape = (-1, 2)
                        texsource = texarray
                        texindex = numpy.arange(len(texindex)*3)
                        texindex.shape = (-1, 3)
                        
                        texsource, texindex = uniqify_multidim_indexes(texsource, texindex)
                    
                    self.all_orig_uvs.append(texsource)
                    self.all_orig_uv_indices.append(numpy.delete(texindex, numpy.where(bad_tris), axis=0) + self.uv_offset)
                    self.uv_offset += len(texsource)
                else:
                    self.all_orig_uv_indices.append(numpy.zeros(shape=(len(boundprim.index)-numpy.sum(bad_tris), 3), dtype=numpy.int32))
                
                self.tri2material.append((self.index_offset, boundprim.material))
                self.index_offset += len(boundprim.index)-numpy.sum(bad_tris)
                
        self.all_vertices = numpy.concatenate(self.all_vertices)
        self.all_normals = numpy.concatenate(self.all_normals)
        self.all_orig_uvs = numpy.concatenate(self.all_orig_uvs) if len(self.all_orig_uvs) > 0 else numpy.array([], dtype=numpy.float32)
        self.all_vert_indices = numpy.concatenate(self.all_vert_indices)
        self.all_normal_indices = numpy.concatenate(self.all_normal_indices)
        self.all_orig_uv_indices = numpy.concatenate(self.all_orig_uv_indices) if len(self.all_orig_uv_indices) > 0 else numpy.array([], dtype=numpy.int32)
    
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
        
        self.all_vertices, self.all_vert_indices = uniqify_multidim_indexes(self.all_vertices, self.all_vert_indices)
        self.all_normals, self.all_normal_indices = uniqify_multidim_indexes(self.all_normals, self.all_normal_indices)
        
        #scale to known range so error values are normalized
        self.all_vertices[:,0] -= numpy.min(self.all_vertices[:,0])
        self.all_vertices[:,1] -= numpy.min(self.all_vertices[:,1])
        self.all_vertices[:,2] -= numpy.min(self.all_vertices[:,2])
        self.all_vertices *= 1000.0 / numpy.max(self.all_vertices)
        
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
        facegraph.add_nodes_from(( (i, {'tris': [i],
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
        
        #store diffuse color of each face, or None for textures
        # this will be used to constrain the chart merging so that color-only
        # faces don't get merged with textured faces
        for matnum, (tri_index, mat) in enumerate(self.tri2material):
            if matnum < len(self.tri2material) - 1:
                end_range = self.tri2material[matnum+1][0]
            else:
                end_range = len(self.all_vert_indices)
            
            if mat is None or mat.effect is None or mat.effect.diffuse is None:
                diffuse = (0.5,0.5,0.5,1)
            elif isinstance(mat.effect.diffuse, tuple):
                diffuse = mat.effect.diffuse
            else:
                diffuse = None
            for i in xrange(tri_index, end_range):
                facegraph.node[i]['diffuse'] = diffuse
                
        self.facegraph = facegraph
        
        self.end_operation()

    def initialize_chart_merge_errors(self):
        self.merge_priorities = []
        self.maxerror = 0
        
        self.begin_operation('(Step 1 of 7) Creating priority queue for initial merges...')
        for v1, v2 in self.facegraph.edges_iter():
            if self.facegraph.node[v1]['diffuse'] != self.facegraph.node[v2]['diffuse']:
                continue
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
            if nx.number_connected_components(connected_components_graph) > 1:
                continue
            #check if boundary has more than one cycle, which can happen when one chart is
            # connected to another by an interior edge
            if len(nx.cycle_basis(connected_components_graph)) != 1:
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
            if logrel > MERGE_ERROR_THRESHOLD:
                break
            #print 'error', error, 'maxerror', self.maxerror, 'logrel', logrel, 'merged left', len(self.merge_priorities), 'numfaces', len(self.facegraph)
            
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
                if nx.number_connected_components(connected_components_graph) != 1:
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
            diffuse = self.facegraph.node[face1]['diffuse']
            self.facegraph.add_node(newface, tris=combined_tris, edges=combined_edges, diffuse=diffuse)        
            self.facegraph.add_edges_from(edges_to_add)
            
            adj_faces = set(self.facegraph.neighbors(face1))
            adj_faces = adj_faces.union(set(self.facegraph.neighbors(face2)))
            adj_faces.remove(face1)
            adj_faces.remove(face2)
            for otherface in adj_faces:
                if self.facegraph.node[otherface]['diffuse'] != diffuse: continue
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
            
            #can't straigten if differing diffuse source (color vs texture)
            if self.facegraph.node[face1]['diffuse'] != self.facegraph.node[face2]['diffuse']:
                continue
            
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
            except nx.exception.NetworkXError:
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
                continue
            
            # This can happen if the shortest path actually encompasses
            # the smaller face, but this would be equivalent to merging the
            # two faces. If we didn't merge these two in the previous step,
            # it was because the cost was too high or it would violate one of
            # the constraints, so just ignore this 
            if len(tris1) == 0 or len(tris2) == 0:
                continue
            
            #this can happen if the straightened path cuts off another face's edges
            if len(tris1) + len(tris2) != len(combined_tris):
                continue
    
            new_edges1 = new_edges1.union(new_combined_edges)
            new_edges2 = new_edges2.union(new_combined_edges)
    
            bordergraph1 = nx.from_edgelist(new_edges1)
            supcycle1 = super_cycle(bordergraph1)
            if supcycle1 is None or len(list(supcycle1)) < 1:
                continue
            bordergraph2 = nx.from_edgelist(new_edges2)
            supcycle2 = super_cycle(bordergraph2)
            if supcycle2 is None or len(list(supcycle2)) < 1:
                continue
    
            #if we stole edges from one face to the other, fix it
            edges_to_add = []
            edges_to_remove = []
            for otherface in self.facegraph.neighbors_iter(face1):
                if otherface == face2:
                    continue
                otheredges = self.facegraph.node[otherface]['edges']
                face1otheredges = otheredges.intersection(new_edges1)
                if len(face1otheredges) == 0:
                    edges_to_remove.append((otherface, face1))
                    edges_to_add.append((otherface, face2))
            for otherface in self.facegraph.neighbors_iter(face2):
                if otherface == face1:
                    continue
                otheredges = self.facegraph.node[otherface]['edges']
                face2otheredges = otheredges.intersection(new_edges2)
                if len(face2otheredges) == 0:
                    edges_to_remove.append((otherface, face2))
                    edges_to_add.append((otherface, face1))
                
            #check if boundary is more than one connected component
            connected_components_graph = nx.from_edgelist(new_combined_edges)
            if nx.number_connected_components(connected_components_graph) > 1:
                continue
            if len(nx.cycle_basis(connected_components_graph)) > 1:
                continue
            
            # check if either new set of edges would be more than one connected component
            graph1 = nx.from_edgelist(new_edges1)
            if nx.number_connected_components(graph1) > 1:
                continue
            graph2 = nx.from_edgelist(new_edges2)
            if nx.number_connected_components(graph2) > 1:
                continue
                
            #ideally we would swap these edges, but this would require revisiting these faces
            # in the loop above, and so we can't do that easily. instead, just disallow
            # a straigtening that would cause the two faces to have to swap neighbors
            #self.facegraph.remove_edges_from(edges_to_remove)
            #self.facegraph.add_edges_from(edges_to_add)
            if len(edges_to_remove) > 0 or len(edges_to_add) > 0:
                continue
                
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
                Bu = numpy.zeros(len(interior_verts), dtype=numpy.float32)
                Bv = numpy.zeros(len(interior_verts), dtype=numpy.float32)
                sumu = numpy.zeros(len(interior_verts), dtype=numpy.float32)
                
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
            
            if facedata['diffuse'] is not None:
                self.facegraph.node[face]['L2'] = 0
                self.new_uvs[self.new_uv_indices[facedata['tris']]] = 0.5
                continue
            
            border_edges = facedata['edges']
            newvert2idx = self.facegraph.node[face]['vert2uvidx']
            chart_tris = self.all_vert_indices[facedata['tris']]
            tri_3d = self.all_vertices[chart_tris]
            tri_2d = self.new_uvs[self.new_uv_indices[facedata['tris']]]
            
            unique_verts, index_map = numpy.unique(chart_tris, return_inverse=True)
            index_map.shape = chart_tris.shape
            border_verts = set(chain.from_iterable(border_edges))
            
            for iteration in range(1, 5):
            
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
                    minx = 0.0 if 0 <= xintercept0 <= 1 else min(xfromy(0), xfromy(1))
                    xintercept1 = yfromx(1)
                    maxx = 1.0 if 0 <= xintercept1 <= 1 else max(xfromy(0), xfromy(1))
                    minx, maxx = tuple(sorted([minx, maxx]))
                    
                    if not(0 <= minx <= maxx <= 1):
                        print minx, maxx, xfromy(0), xfromy(1), yfromx(0), yfromx(1)
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
                    if rangemax < rangemin:
                        rangemin, rangemax = rangemax, rangemin
                        
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

        self.material2color = {}
        tri_areas = []
        for matnum, (i, mat) in enumerate(self.tri2material):
            if matnum < len(self.tri2material) - 1:
                end_range = self.tri2material[matnum+1][0]
            else:
                end_range = len(self.all_orig_uv_indices)
            
            if mat not in self.material2color:
                if mat is None or mat.effect is None or mat.effect.diffuse is None:
                    self.material2color[mat] = (0.5,0.5,0.5,1)
                elif isinstance(mat.effect.diffuse, tuple):
                    self.material2color[mat] = mat.effect.diffuse
                else:
                    self.material2color[mat] = mat.effect.diffuse.sampler.surface.image.pilimage
                    
            if mat is not None and mat.effect is not None and mat.effect.diffuse is not None and not isinstance(mat.effect.diffuse, tuple):
                tri_uvs = self.all_orig_uvs[self.all_orig_uv_indices[i:end_range]]
                tri_uvs[:,:,0] *= self.material2color[mat].size[0]
                tri_uvs[:,:,1] *= self.material2color[mat].size[1]
                areas_2d = numpy.abs(tri_areas_2d(tri_uvs))
                tri_areas.append(areas_2d)
            else:
                tri_areas.append(numpy.zeros(end_range-i+1, dtype=numpy.int32))
        
        if len(tri_areas) > 0:
            tri_areas = numpy.concatenate(tri_areas)
        total_texture_area = numpy.sum(tri_areas)

        total_3d_area = 0
        for face, facedata in self.facegraph.nodes_iter(data=True):
            chart_3d_area = numpy.sum(tri_areas_3d(self.all_vertices[self.all_vert_indices[facedata['tris']]]))
            facedata['chart_3d_area'] = chart_3d_area
            total_3d_area += chart_3d_area

        TEXTURE_DIMENSION = min(math.sqrt(total_texture_area), 2048.0)
        TEXTURE_SIZE = TEXTURE_DIMENSION * TEXTURE_DIMENSION

        new_total_L2 = self.total_L2
        for face, facedata in self.facegraph.nodes_iter(data=True):
            if facedata['diffuse'] is not None:
                continue
            
            chart_area = numpy.sum(tri_areas[facedata['tris']])
            l2_frac = facedata['L2'] / self.total_L2
            area_frac = chart_area / total_texture_area
            area3d_frac = facedata['chart_3d_area'] / total_3d_area
            fair_share = ((l2_frac * area_frac * area3d_frac) ** (1.0/3.0)) * TEXTURE_SIZE
            
            if fair_share > chart_area and fair_share > 16:
                newface_L2 = (((chart_area / TEXTURE_SIZE) ** 2) / area_frac) * self.total_L2
                new_total_L2 -= facedata['L2'] - newface_L2
                facedata['L2'] = newface_L2

        self.total_L2 = new_total_L2
        
        rp = RectPack()
        self.chart_ims = {}
        self.chart_masks = {}
        self.pil_to_cv = {}
        for face, facedata in self.facegraph.nodes_iter(data=True):
            
            if facedata['diffuse'] is not None:
                continue
            
            #calculate fair share but multiplying the L2 fraction with
            # the fraction that the chart takes up in the total area
            # then take the square root so that it sums to 1
            chart_area = numpy.sum(tri_areas[facedata['tris']])
            l2_frac = facedata['L2'] / self.total_L2
            area_frac = chart_area / total_texture_area
            area3d_frac = facedata['chart_3d_area'] / total_3d_area
            fair_share = ((l2_frac + area_frac + area3d_frac) / 3.0) * TEXTURE_SIZE
            
            #get the x range and y range of the chart circle
            chart_uv_locs = numpy.unique(self.new_uv_indices[facedata['tris']])
            minx = numpy.min(self.new_uvs[chart_uv_locs, 0])
            maxx = numpy.max(self.new_uvs[chart_uv_locs, 0])
            miny = numpy.min(self.new_uvs[chart_uv_locs, 1])
            maxy = numpy.max(self.new_uvs[chart_uv_locs, 1])
            #then scale the chart uvs to be in the range 0,1
            self.new_uvs[chart_uv_locs, 0] -= minx
            chart_width_frac = (maxx - minx)
            self.new_uvs[chart_uv_locs, 0] *= 1.0 / chart_width_frac
            self.new_uvs[chart_uv_locs, 1] -= miny
            chart_height_frac = (maxy - miny)
            self.new_uvs[chart_uv_locs, 1] *= 1.0 / chart_height_frac

            #now chart width and height can be calculated by the uv range and fair share fraction
            chart_width = math.pow(fair_share, chart_width_frac / (chart_width_frac + chart_height_frac))
            chart_height = math.pow(fair_share, chart_height_frac / (chart_width_frac + chart_height_frac))
            if chart_width > (TEXTURE_DIMENSION / 2.0):
                chart_height = fair_share / (TEXTURE_DIMENSION / 2.0)
                chart_width = (TEXTURE_DIMENSION / 2.0)
            if chart_height > (TEXTURE_DIMENSION / 2.0):
                chart_width += fair_share / (TEXTURE_DIMENSION / 2.0)
                chart_height = (TEXTURE_DIMENSION / 2.0)
            if chart_width < 8: chart_width = 8.0
            if chart_height < 8: chart_height = 8.0
            
            #round to power of 2 with 1 pixel border
            chart_width = int(math.pow(2, round(math.log(chart_width, 2)))) - 2
            chart_height = int(math.pow(2, round(math.log(chart_height, 2)))) - 2
            
            self.facegraph.node[face]['chart_size'] = (chart_width, chart_height)

            chartim = Image.new('RGB', (chart_width, chart_height))
            chartmask = Image.new('L', (chart_width, chart_height), 255)
            maskdraw = ImageDraw.Draw(chartmask)
                
            for tri in facedata['tris']:
                
                newuvs = self.new_uvs[self.new_uv_indices[tri]]
                newu = (newuvs[:,0] * (chart_width-0.5))
                newv = ((1.0-newuvs[:,1]) * (chart_height-0.5))
                newtri = [(newu[0], newv[0]), (newu[1], newv[1]), (newu[2], newv[2])]
                maskdraw.polygon(newtri, fill=0, outline=0)
                
                bisect_loc = bisect.bisect(self.tri2material, (tri,0))
                if bisect_loc >= len(self.tri2material) or tri != self.tri2material[bisect_loc][0]:
                    bisect_loc -= 1
                diffuse_source = self.material2color[self.tri2material[bisect_loc][1]]
                if pcv is not None:
                    cv_diffuse_source = self.pil_to_cv.get(diffuse_source)
                    if cv_diffuse_source is None:
                        cv_diffuse_source = pcv.Mat.from_pil_image(diffuse_source)
                        self.pil_to_cv[diffuse_source] = cv_diffuse_source
                
                prevuvs = self.all_orig_uvs[self.all_orig_uv_indices[tri]]
                prevu = prevuvs[:,0] * diffuse_source.size[0]
                prevv = (1.0-prevuvs[:,1]) * diffuse_source.size[1]
                prevtri = [(prevu[0], prevv[0]), (prevu[1], prevv[1]), (prevu[2], prevv[2])]
                
                if pcv is None:
                    transformblit(prevtri, newtri, diffuse_source, chartim)
                else:
                    #we prefer opencv because it allows us to wrap the texcoords
                    opencvblit(prevtri, newtri, cv_diffuse_source, chartim)
                    
            self.chart_ims[face] = chartim
            self.chart_masks[face] = chartmask

            rp.addRectangle(face, chart_width+2, chart_height+2)
        
        del self.pil_to_cv
        assert(rp.pack())
        
        #find size for color charts so that they are still visible at 128x128 mipmap
        mintexsize = min(rp.width, rp.height)
        colorsize = max((mintexsize / 128.0), 8.0)
        #round up to power of 2, with 2 pixel border
        colorsize = int(math.pow(2, math.ceil(math.log(colorsize, 2)))) - 2
        
        self.color2chart = {}
        self.color2faces = {}
        for face, facedata in self.facegraph.nodes_iter(data=True):
            if facedata['diffuse'] is None: continue
            
            if facedata['diffuse'] not in self.color2chart:
                color = facedata['diffuse']
                color = (int(color[0]*255), int(color[1]*255), int(color[2]*255))
                chartim = Image.new('RGB', (colorsize, colorsize), color)
                chartmask = Image.new('L', (colorsize, colorsize), 0)
                self.color2chart[facedata['diffuse']] = (chartim, chartmask)
                self.color2faces[facedata['diffuse']] = [face]
                rp.addRectangle(facedata['diffuse'], colorsize+2, colorsize+2)
                self.chart_ims[facedata['diffuse']] = chartim
                self.chart_masks[facedata['diffuse']] = chartmask
            else:
                self.color2faces[facedata['diffuse']].append(face)
            
        assert(rp.pack())

        self.chart_packing = rp
        
        #now resize the uvs according to chart size
        for face, facedata in self.facegraph.nodes_iter(data=True):
            chart_uvs = numpy.unique(self.new_uv_indices[facedata['tris']])
            self.facegraph.node[face]['chart_uvs'] = chart_uvs
            
            if facedata['diffuse'] is not None: continue
            
            (chart_width, chart_height) = self.facegraph.node[face]['chart_size']
            self.new_uvs[chart_uvs, 0] *= chart_width-0.5
            self.new_uvs[chart_uvs, 1] *= chart_height-0.5
        
        self.end_operation()

    def normalize_uvs(self):
        self.begin_operation('Normalizing texture coordinates...')

        for face, facedata in self.facegraph.nodes_iter(data=True):
            if facedata['diffuse'] is not None: continue
            
            (chart_width, chart_height) = self.facegraph.node[face]['chart_size']
            
            chart_uvs = self.facegraph.node[face]['chart_uvs']

            self.new_uvs[chart_uvs, 0] /= chart_width-0.5
            self.new_uvs[chart_uvs, 1] /= chart_height-0.5
        
        self.end_operation()

    def evaluate_edge_collapse(self, v1, v2):
        #considering (v1,v2) -> v1
        
        #can't remove corners
        if v2 in self.all_corners:
            return
        
        #need to preserve boundary straightness
        if v2 in self.all_edge_verts and v1 not in self.all_edge_verts:
            return

        v2tris = list(self.vertexgraph.node[v2]['tris'])
        v2tri_idx = self.all_vert_indices[v2tris]
        
        moved = set()
        degenerate = set()
        degenerate_segments = []
        for t2, t2_idx in izip(v2tris, v2tri_idx):
            if v1 in t2_idx:
                degenerate.add(t2)
                facefrom = self.tri2face[t2]
                
                #texture stretch for color-only face is 0
                if self.facegraph.node[facefrom]['diffuse'] is not None:
                    continue
                
                face_vert2uvidx = self.facegraph.node[facefrom]['vert2uvidx']
                other_vert = [v for v in t2_idx if v != v1 and v != v2][0]
                degen_3d = (self.all_vertices[other_vert], self.all_vertices[v2])
                degen_uv = (self.new_uvs[face_vert2uvidx[other_vert]], self.new_uvs[face_vert2uvidx[v2]])
                degenerate_segments.append((degen_3d, degen_uv))
            else:
                moved.add(t2)

        texture_stretches = []
        for moved_tri_idx in moved:
            facefrom = self.tri2face[moved_tri_idx]
            
            #texture stretch for color-only face is 0
            if self.facegraph.node[facefrom]['diffuse'] is not None:
                continue
            
            face_vert2uvidx = self.facegraph.node[facefrom]['vert2uvidx']
            
            #don't want to cross chart boundaries
            if v1 not in face_vert2uvidx:
                return
            
            moved_vert_idx = self.all_vert_indices[moved_tri_idx]
            other_pts = [ m for m in moved_vert_idx if m != v2 ]
            other_uv_pts = [ self.new_uvs[face_vert2uvidx[m]] for m in other_pts ]
            
            v2_pt = self.new_uvs[face_vert2uvidx[v2]]
            v1_pt = self.new_uvs[face_vert2uvidx[v1]]
            
            #this checks if the triangle has flipped
            # see: http://stackoverflow.com/questions/7365531/detecting-if-a-triangle-flips-when-changing-a-point
            vec1 = numpy.array([other_uv_pts[1][0] - other_uv_pts[0][0], other_uv_pts[1][1] - other_uv_pts[0][1], 0], dtype=numpy.float32)
            vec2_v1 = numpy.array([v1_pt[0] - other_uv_pts[0][0], v1_pt[1] - other_uv_pts[0][1], 0], dtype=numpy.float32)
            vec2_v2 = numpy.array([v2_pt[0] - other_uv_pts[0][0], v2_pt[1] - other_uv_pts[0][1], 0], dtype=numpy.float32)
            v1_direction = numpy.cross(vec1, vec2_v1)[2]
            v2_direction = numpy.cross(vec1, vec2_v2)[2]
            
            #don't want to flip the triangle in the parametric domain
            if v2_direction > 0 and not(v1_direction > 0) or \
                v2_direction < 0 and not(v1_direction < 0):
                return

            other_v3_pts = [self.all_vertices[m] for m in other_pts]
            v1_3d = self.all_vertices[v1]
            
            #the maximum texture deviation could lie at the edge,edge intersection
            # between the moving edge and a degenerate edge
            for moving_3d, moving_uv in zip(other_v3_pts, other_uv_pts):
                for (degen_3d, degen_uv) in degenerate_segments:
                    #find where the uv coords intersect
                    intersect_uv = seg_intersect(moving_uv, v1_pt, degen_uv[0], degen_uv[1])
                    if intersect_uv is None: continue

                    #convert the u,v intersection to 3d for the degenerate edge
                    degen_uv_dist = v2dist(degen_uv[0], degen_uv[1])
                    if degen_uv_dist <= 0: continue
                    degen_intersect_relative = v2dist(degen_uv[0], intersect_uv) / degen_uv_dist
                    degen_v3_diff = degen_3d[1] - degen_3d[0]
                    degen_v3_intersect = degen_3d[0] + degen_v3_diff * degen_intersect_relative
                    
                    #and then to 3d for the moving edge
                    moving_uv_dist = v2dist(moving_uv, v1_pt)
                    if moving_uv_dist <= 0: continue
                    moving_intersect_relative = v2dist(moving_uv, intersect_uv) / moving_uv_dist
                    moving_v3_diff = v1_3d - moving_3d
                    moving_v3_intersect = moving_3d + moving_v3_diff * moving_intersect_relative

                    v3_texture_stretch = v3dist(degen_v3_intersect, moving_v3_intersect)
                    texture_stretches.append(v3_texture_stretch)
        
        dist3d_v1_v2 = v3dist(self.all_vertices[v1], self.all_vertices[v2])
        texture_stretches.append(dist3d_v1_v2)
        texture_diff = max(texture_stretches)
        
        combined_A = self.vert_quadric_A[v1] + self.vert_quadric_A[v2]
        combined_b = self.vert_quadric_b[v1] + self.vert_quadric_b[v2]
        combined_c = self.vert_quadric_c[v1] + self.vert_quadric_c[v2]
        quadric_error = evalQuadric(combined_A, combined_b, combined_c, self.all_vertices[v1])
        combined_error = texture_diff + quadric_error
        return combined_error

    def initialize_simplification_errors(self):
        self.begin_operation('Calculationg priority queue for initial edge contractions...')
        
        self.all_corners = set()
        self.all_edge_verts = set()
        self.tri2face = {}
        for face, facedata in self.facegraph.nodes_iter(data=True):
            self.all_corners = self.all_corners.union(facedata['corners'])
            self.all_edge_verts = self.all_edge_verts.union(set(chain.from_iterable(facedata['edges'])))
            facev2uv = self.facegraph.node[face]['vert2uvidx']
            for tri in facedata['tris']:
                self.tri2face[tri] = face
            self.facegraph.node[face]['orig_tris'] = facedata['tris']

        self.vertexgraph.add_nodes_from(( (i, {'tris':set()}) for i in xrange(len(self.all_vertices))))
        for i, (v1,v2,v3) in enumerate(self.all_vert_indices):
            self.vertexgraph.node[v1]['tris'].add(i)
            self.vertexgraph.node[v2]['tris'].add(i)
            self.vertexgraph.node[v3]['tris'].add(i)

        self.contraction_priorities = []

        (A, b, c, area, normal) = quadricsForTriangles(self.all_vertices[self.all_vert_indices])
        
        self.vert_quadric_A = numpy.zeros(shape=(len(self.all_vertices), 3, 3), dtype=numpy.float32)
        self.vert_quadric_b = numpy.zeros(shape=(len(self.all_vertices), 3), dtype=numpy.float32)
        self.vert_quadric_c = numpy.zeros(shape=(len(self.all_vertices)), dtype=numpy.float32)
        
        self.vert_quadric_A[self.all_vert_indices[:,0]] += A / 3.0
        self.vert_quadric_A[self.all_vert_indices[:,1]] += A / 3.0
        self.vert_quadric_A[self.all_vert_indices[:,2]] += A / 3.0
        
        self.vert_quadric_b[self.all_vert_indices[:,0]] += b / 3.0
        self.vert_quadric_b[self.all_vert_indices[:,1]] += b / 3.0
        self.vert_quadric_b[self.all_vert_indices[:,2]] += b / 3.0
        
        self.vert_quadric_c[self.all_vert_indices[:,0]] += c / 3.0
        self.vert_quadric_c[self.all_vert_indices[:,1]] += c / 3.0
        self.vert_quadric_c[self.all_vert_indices[:,2]] += c / 3.0

        #to preserve borders, we inflate the quadric error for edges that
        # only have one incident triangle
        for v1,v2 in self.vertexgraph.edges_iter():
            tris1 = self.vertexgraph.node[v1]['tris']
            tris2 = self.vertexgraph.node[v2]['tris']
            tris_both = tris1.intersection(tris2)
            if len(tris_both) > 1: continue
            t = list(tris_both)[0]
            
            v = self.all_vertices[v1] - self.all_vertices[v2]
            normal2 = numpy.cross(v, normal[t])
            normal2 = normal2 / numpy.linalg.norm(normal2)
            d = -numpy.dot(normal[t], self.all_vertices[v1])
            A3 = area[t] * numpy.outer(normal2, normal2)
            b3 = area[t] * d * normal2
            c3 = area[t] * d * d

            self.vert_quadric_A[v1] += A3
            self.vert_quadric_b[v1] += b3
            self.vert_quadric_c[v1] += c3

        self.maxerror = 0
        
        for (vv1, vv2) in self.vertexgraph.edges_iter():
            for (v1, v2) in ((vv1,vv2),(vv2,vv1)):
                combined_error = self.evaluate_edge_collapse(v1,v2)
                if combined_error is None:
                    continue
                if combined_error > self.maxerror:
                    self.maxerror = combined_error
                
                self.contraction_priorities.append((combined_error, (v1, v2)))
                
        heapq.heapify(self.contraction_priorities)

        self.end_operation()

    def simplify_mesh(self):
        self.begin_operation('(Step 4 of 7) Simplifying...')
        
        self.tris_left = set(xrange(len(self.all_vert_indices)))
        self.simplify_operations = []
        
        while len(self.contraction_priorities) > 0:
            (error, (v1, v2)) = heapq.heappop(self.contraction_priorities)
            
            #considering (v1,v2) -> v1
            
            #check of one of these vertices was already contracted
            if v1 not in self.vertexgraph or v2 not in self.vertexgraph:
                continue

            #cutoff value was chosen which seems to work well for most models
            if error > 0:
                logrel = math.log(1 + error) / math.log(1 + self.maxerror)
                #print 'error', error, 'maxerror', self.maxerror, 'logrel', logrel, 'v1', v1, 'v2', v2, 'numverts', len(self.vertexgraph), 'numfaces', len(self.tris_left), 'contractions left', len(self.contraction_priorities)
            else: logrel = 0
            if logrel > SIMPLIFICATION_ERROR_THRESHOLD and len(self.tris_left) < TRIANGLE_MAXIMUM:
                break
            
            v2tris = list(self.vertexgraph.node[v2]['tris'])
            v1tris = self.vertexgraph.node[v1]['tris']
            v2tri_idx = self.all_vert_indices[v2tris]
            
            invalid_contraction = False
            for t2 in v2tris:
                facefrom = self.tri2face[t2]
                if v1 not in self.facegraph.node[facefrom]['vert2uvidx']:
                    #this can happen if this triangle was created by a different
                    # merge and so we didn't check this constraint when considering the edge before
                    invalid_contraction = True
                    break
            if invalid_contraction:
                continue
            
            self.simplify_operations.append(STREAM_OP.OPERATION_BOUNDARY)
            
            #do degenerate first so we can record values for progressive stream
            degenerate = set()
            for t2, t2_idx in izip(v2tris, v2tri_idx):
                if v1 in t2_idx:
                    degenerate.add(t2)
                    self.simplify_operations.append((STREAM_OP.TRIANGLE_ADDITION, t2, t2_idx, self.all_normal_indices[t2], self.new_uv_indices[t2]))
            
            new_contractions = set()
            for t2, t2_idx in izip(v2tris, v2tri_idx):
                if v1 not in t2_idx:
                    #these are triangles that have a vertex moving from v2 to v1
                    where_v2 = numpy.where(t2_idx == v2)[0][0]
                    where_not_v2 = numpy.where(t2_idx != v2)
                    other1 = where_not_v2[0][0]
                    other2 = where_not_v2[0][1]
                    
                    #update vert and uv index values from v2 to v1
                    facefrom = self.tri2face[t2]
                    face_vert2uvidx = self.facegraph.node[facefrom]['vert2uvidx']
                    prev_vertex_value = self.all_vert_indices[t2][where_v2]
                    self.all_vert_indices[t2][where_v2] = v1
                    prev_uv_value = self.new_uv_indices[t2][where_v2]
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
                    prev_normal_value = self.all_normal_indices[t2][where_v2]
                    self.all_normal_indices[t2][where_v2] = self.all_normal_indices[copy_tri_v1][where_v1]
                    
                    self.simplify_operations.append((STREAM_OP.INDEX_UPDATE, t2, where_v2, prev_vertex_value, prev_normal_value, prev_uv_value))
                    
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
            
            #update quadric
            self.vert_quadric_A[v1] += self.vert_quadric_A[v2]
            self.vert_quadric_b[v1] += self.vert_quadric_b[v2]
            self.vert_quadric_c[v1] += self.vert_quadric_c[v2]       
            
            #now update priority list with new valid contractions
            for (v1, v2) in new_contractions:
                combined_error = self.evaluate_edge_collapse(v1,v2)
                if combined_error is None:
                    continue
                if combined_error > self.maxerror:
                    self.maxerror = combined_error
                
                heapq.heappush(self.contraction_priorities, (combined_error, (v1, v2)))
            
        self.end_operation()

    def enforce_simplification(self):
        if len(self.tris_left) < TRIANGLE_MAXIMUM:
            return
        self.begin_operation('Enforcing simplification...')
        
        list_tris_left = numpy.array(list(self.tris_left), dtype=numpy.int32)
        areas3d = tri_areas_3d(self.all_vertices[self.all_vert_indices[list_tris_left]])
        sorted_indices = numpy.argsort(areas3d)
        list_tris_left = list_tris_left[sorted_indices]
        
        i = 0
        while len(self.tris_left) > TRIANGLE_MAXIMUM:
            tri = list_tris_left[i]
            i += 1
            tri_idx = self.all_vert_indices[tri]
            self.simplify_operations.append((STREAM_OP.TRIANGLE_ADDITION,
                                             tri,
                                             tri_idx,
                                             self.all_normal_indices[tri],
                                             self.new_uv_indices[tri]))
            
            for v in tri_idx:
                self.vertexgraph.node[v]['tris'].discard(tri)
                
            self.tris_left.remove(tri)
        
        self.end_operation()

    def pack_charts(self):
        self.begin_operation('(Step 6 of 7) Creating and packing charts into atlas...')
        self.atlasimg = Image.new('RGB', (self.chart_packing.width, self.chart_packing.height))
        atlasmask = Image.new('L', (self.chart_packing.width, self.chart_packing.height), 255)
        
        for face_or_color, chartim in self.chart_ims.iteritems():

            x,y,w,h = self.chart_packing.getPlacement(face_or_color)
            #adjust for 1 pixel border
            x += 1
            y += 1
            w -= 2
            h -= 2
            
            self.atlasimg.paste(chartim, (x,y,x+w,y+h))
            atlasmask.paste(self.chart_masks[face_or_color], (x,y,x+w,y+h))
            x,y,w,h,width,height = (float(i) for i in (x,y,w,h,self.chart_packing.width,self.chart_packing.height))
            
            faces_to_get = []
            if isinstance(face_or_color, tuple):
                faces_to_get = self.color2faces[face_or_color]
            else:
                faces_to_get.append(face_or_color)
            
            chart_tris = []
            chart_uvs = []
            for face in faces_to_get:
                chart_tris.extend(f for f in self.facegraph.node[face]['orig_tris'] if f in self.tris_left)
                chart_uvs.extend(self.facegraph.node[face]['chart_uvs'])
            chart_uvs = numpy.unique(chart_uvs)
    
            #this rescales the texcoords to map to the new atlas location
            self.new_uvs[chart_uvs,0] = (self.new_uvs[chart_uvs,0] * (w-0.5) + x) / width
            self.new_uvs[chart_uvs,1] = 1.0 - (( (1.0-self.new_uvs[chart_uvs,1]) * (h-0.5) + y ) / height)

        if cv is not None:
            #convert image and mask to opencv format
            cv_im = cv.CreateImageHeader(self.atlasimg.size, cv.IPL_DEPTH_8U, 3)
            cv.SetData(cv_im, self.atlasimg.tostring())
            cv_mask = cv.CreateImageHeader(atlasmask.size, cv.IPL_DEPTH_8U, 1)
            cv.SetData(cv_mask, atlasmask.tostring())

            #do the inpainting
            cv_painted_im = cv.CloneImage(cv_im)
            cv.Inpaint(cv_im, cv_mask, cv_painted_im, 3, cv.CV_INPAINT_TELEA)
            
            #convert back to PIL
            self.atlasimg = Image.fromstring("RGB", cv.GetSize(cv_painted_im), cv_painted_im.tostring())

            # -- using pyopencv which has strange bugs
            #cv_im = cv.Mat.from_pil_image(self.atlasimg)
            #cv_mask = cv.Mat.from_pil_image(atlasmask)
            #cv_mask = cv_mask.reshape(1, cv_im.rows)
            #cv_painted_im = cv_im.clone()
            #cv.inpaint(cv_im, cv_mask, cv_painted_im, 3, cv.INPAINT_TELEA)
            #self.atlasimg = cv_painted_im.to_pil_image()

        self.end_operation()

    def split_base_and_pm(self):
        
        self.begin_operation('Creating progressive stream...')
        
        #first uniqify the uvs
        self.new_uvs, self.new_uv_indices, self.old2newuvmap = uniqify_multidim_indexes(self.new_uvs, self.new_uv_indices, return_map=True)
        
        base_tris = numpy.array(list(self.tris_left), dtype=numpy.int32)
        tri_mapping = numpy.zeros(shape=(len(self.all_vert_indices),), dtype=numpy.int32)
        tri_mapping[base_tris] = numpy.arange(len(self.tris_left), dtype=numpy.int32)
        
        #strip out unused indices
        self.all_vert_indices = self.all_vert_indices[base_tris]
        self.all_normal_indices = self.all_normal_indices[base_tris]
        self.new_uv_indices = self.new_uv_indices[base_tris]

        #have to flatten and reshape like this so that it's contiguous
        stacked_indices = numpy.hstack((self.all_vert_indices.reshape(-1, 1),
                                        self.all_normal_indices.reshape(-1, 1),
                                        self.new_uv_indices.reshape(-1, 1))).flatten().reshape((-1, 3))
        
        #index_map - maps each unique value back to a location in the original array it came from
        #   eg. stacked_indices[index_map] == unique_stacked_indices
        #new_tris - maps original array locations to their location in the unique array
        #   e.g. unique_stacked_indices[new_tris] == stacked_indices
        unique_stacked_indices, index_map, new_tris = numpy.unique(stacked_indices.view([('',stacked_indices.dtype)]*stacked_indices.shape[1]), return_index=True, return_inverse=True)
        unique_stacked_indices = unique_stacked_indices.view(stacked_indices.dtype).reshape(-1,stacked_indices.shape[1])
        
        #unique returns as int64, so cast back
        index_map = numpy.cast['int32'](index_map)
        new_tris = numpy.cast['int32'](new_tris)
        
        #sort the index map to get a list of the index of the first time each value was encountered
        sorted_map = numpy.cast['int32'](numpy.argsort(index_map))
        
        #since we're sorting the unique values, we have to map the new_tris to the new index locations
        backwards_map = numpy.zeros_like(sorted_map)
        backwards_map[sorted_map] = numpy.arange(len(sorted_map), dtype=numpy.int32)
        
        #now this is the new unique values and their indices
        unique_stacked_indices = unique_stacked_indices[sorted_map]
        new_tris = backwards_map[new_tris]
        new_tris.shape = (-1, 3)
        
        oldindex2newindex = {}
        for i, oldset in enumerate(unique_stacked_indices):
            oldindex2newindex[tuple(oldset)] = i 
        
        print 'num unique vert data locs in base mesh', len(unique_stacked_indices)
        print 'num triangles in base mesh', len(self.tris_left)
        cur_triangle = len(self.tris_left)
        
        vertex_buffer = []
        update_buffer = []
        triangle_buffer = []
        operations_buffer = []
        
        for operation in reversed(self.simplify_operations):
            if operation == STREAM_OP.OPERATION_BOUNDARY:
                num_ops = len(vertex_buffer) + len(update_buffer) + len(triangle_buffer)
                if num_ops > 0:
                    entire_op = "%d\n" % num_ops
                    for s in chain(vertex_buffer, triangle_buffer, update_buffer):
                        entire_op += s
                    operations_buffer.append(entire_op)
                    vertex_buffer = []
                    update_buffer = []
                    triangle_buffer = []
            
            elif operation[0] == STREAM_OP.INDEX_UPDATE:
                op, tri_index, vert_index, changed_vert, changed_normal, changed_uv = operation
                changed_uv = self.old2newuvmap[changed_uv]
                #print 'index update', tri_index, vert_index, changed_vert, changed_normal, changed_uv
                
                new_tri_index = tri_mapping[tri_index]
                unique_old_index = (changed_vert, changed_normal, changed_uv)
                if unique_old_index not in oldindex2newindex:
                    v = self.all_vertices[changed_vert]
                    n = self.all_normals[changed_normal]
                    u = self.new_uvs[changed_uv]
                    vertex_buffer.append("v %.7g %.7g %.7g %.7g %.7g %.7g %.7g %.7g\n" % (v[0], v[1], v[2], n[0], n[1], n[2], u[0], u[1]))
                    oldindex2newindex[unique_old_index] = len(oldindex2newindex)
                update_buffer.append("u %d %d\n" % (new_tri_index*3 + vert_index, oldindex2newindex[unique_old_index]))
                
            elif operation[0] == STREAM_OP.TRIANGLE_ADDITION:
                op, oldtri, vert_idx, norm_idx, uv_idx = operation
                uv_idx = [self.old2newuvmap[uv_idx[0]], self.old2newuvmap[uv_idx[1]], self.old2newuvmap[uv_idx[2]]]
                #print 'newtri', oldtri, vert_idx, norm_idx, uv_idx
                
                newtri = []
                for pt in ((vert_idx[0], norm_idx[0], uv_idx[0]),
                           (vert_idx[1], norm_idx[1], uv_idx[1]),
                           (vert_idx[2], norm_idx[2], uv_idx[2])):
                    if pt in oldindex2newindex:
                        newtri.append(oldindex2newindex[pt])
                    else:
                        v = self.all_vertices[pt[0]]
                        n = self.all_normals[pt[1]]
                        u = self.new_uvs[pt[2]]
                        vertex_buffer.append("v %.7g %.7g %.7g %.7g %.7g %.7g %.7g %.7g\n" % (v[0], v[1], v[2], n[0], n[1], n[2], u[0], u[1]))
                        newtri.append(len(oldindex2newindex))
                        oldindex2newindex[pt] = len(oldindex2newindex)
                triangle_buffer.append("t %d %d %d\n" % (newtri[0], newtri[1], newtri[2]))
                tri_mapping[oldtri] = cur_triangle
                cur_triangle += 1
            else:
                assert(False)
        
        #check for last operation
        num_ops = len(vertex_buffer) + len(update_buffer) + len(triangle_buffer)
        if num_ops > 0:
            entire_op = "%d\n" % num_ops
            for s in chain(vertex_buffer, triangle_buffer, update_buffer):
                entire_op += s
            operations_buffer.append(entire_op)
        
        self.pmbuf.write("PDAE\n")
        self.pmbuf.write("%d\n" % len(operations_buffer))
        for op in operations_buffer:
            self.pmbuf.write(op)
        operations_buffer = None
        
        self.end_operation()
        
        self.begin_operation('Compressing base mesh...')

        #compress verts
        self.all_vertices, self.all_vert_indices = uniqify_multidim_indexes(self.all_vertices, self.all_vert_indices)

        #compress normals
        self.all_normals, self.all_normal_indices = uniqify_multidim_indexes(self.all_normals, self.all_normal_indices)
        
        #compress uvs
        self.new_uvs, self.new_uv_indices = uniqify_multidim_indexes(self.new_uvs, self.new_uv_indices)
        
        self.end_operation()

    def add_back_pm(self):
        self.begin_operation('Reconstructing full mesh because progressive stream is too small...')
        
        for operation in reversed(self.simplify_operations):
            if operation == STREAM_OP.OPERATION_BOUNDARY:
                pass
            
            elif operation[0] == STREAM_OP.INDEX_UPDATE:
                op, tri_index, vert_index, changed_vert, changed_normal, changed_uv = operation
                self.all_vert_indices[tri_index][vert_index] = changed_vert
                self.all_normal_indices[tri_index][vert_index] = changed_normal
                self.new_uv_indices[tri_index][vert_index] = changed_uv
                
            elif operation[0] == STREAM_OP.TRIANGLE_ADDITION:
                pass
        
        self.end_operation()
        
        self.begin_operation('Compressing base mesh...')

        #compress verts
        self.all_vertices, self.all_vert_indices = uniqify_multidim_indexes(self.all_vertices, self.all_vert_indices)

        #compress normals
        self.all_normals, self.all_normal_indices = uniqify_multidim_indexes(self.all_normals, self.all_normal_indices)
        
        #compress uvs
        self.new_uvs, self.new_uv_indices = uniqify_multidim_indexes(self.new_uvs, self.new_uv_indices)
        
        self.end_operation()

    def save_mesh(self):
        self.begin_operation('(Step 7 of 7) Saving mesh...')
        newmesh = collada.Collada()
        newmesh.assetInfo.title = self.mesh.assetInfo.title
        newmesh.assetInfo.subject = self.mesh.assetInfo.subject
        newmesh.assetInfo.revision = self.mesh.assetInfo.revision
        newmesh.assetInfo.keywords = self.mesh.assetInfo.keywords
        newmesh.assetInfo.upaxis = self.mesh.assetInfo.upaxis
        
        sander_contributor = collada.asset.Contributor(authoring_tool='meshtool',
                                                       comments='Retextured and simplified base mesh using Texture Mapping Progressive Meshes, Sander et al.')
        newmesh.assetInfo.contributors.append(sander_contributor)
        
        cimg = collada.material.CImage("sander-simplify-packed-atlas", "./atlas.jpg")
        imgout = StringIO()
        self.atlasimg.save(imgout, format="JPEG", quality=95, optimize=True)
        cimg.setData(imgout.getvalue())
        newmesh.images.append(cimg)
        
        surface = collada.material.Surface("sander-simplify-surface", cimg)
        sampler = collada.material.Sampler2D("sander-simplify-sampler", surface)
        mapper = collada.material.Map(sampler, "TEX0")
        effect = collada.material.Effect("sander-simplify-effect", [surface, sampler], "blinn", diffuse=mapper)
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
        
        new_index = numpy.dstack((self.all_vert_indices, self.all_normal_indices, self.new_uv_indices)).flatten()
        triset = geom.createTriangleSet(new_index, input_list, "materialref")
        geom.primitives.append(triset)
        newmesh.geometries.append(geom)
        
        matnode = collada.scene.MaterialNode("materialref", material, inputs=[('TEX0', 'TEXCOORD', '0')])
        geomnode = collada.scene.GeometryNode(geom, [matnode])
        node = collada.scene.Node("node0", children=[geomnode])
        
        myscene = collada.scene.Scene("myscene", [node])
        newmesh.scenes.append(myscene)
        newmesh.scene = myscene
        self.end_operation()
        return newmesh

    def simplify(self):
        self.uniqify_list()
        self.build_vertex_graph()
        self.build_face_graph()
        
        #renderCharts(self.facegraph, self.all_vertices, self.all_vert_indices)
        print 'number of vertices =', len(self.vertexgraph)
        print 'number of faces =', len(self.facegraph)
        print 'connected vertex components =', nx.number_connected_components(self.vertexgraph)
        print 'connected face components =', nx.number_connected_components(self.facegraph)
        
        self.initialize_chart_merge_errors()
        self.merge_charts()
        
        print 'number of charts =', len(self.facegraph)
        print 'connected face components =', nx.number_connected_components(self.facegraph)
        
        #renderCharts(self.facegraph, self.all_vertices, self.all_vert_indices)
        
        self.update_corners()
        
        self.calc_edge_length()
        self.straighten_chart_boundaries()
        self.update_corners(enforce=True)
        
        #renderCharts(self.facegraph, self.all_vertices, self.all_vert_indices)
        
        self.create_initial_parameterizations()
        self.optimize_chart_parameterizations()
        self.resize_charts()
        print 'texture size = (%dx%d)' % (self.chart_packing.width, self.chart_packing.height)
        
        self.initialize_simplification_errors()
        self.simplify_mesh()
        print 'number of faces in base mesh after simplification =', len(self.tris_left)
        self.enforce_simplification()
        print 'number of faces in base mesh after enforced simplification =', len(self.tris_left)
        
        self.normalize_uvs()
        self.pack_charts()
        
        self.orig_tri_count = len(self.all_vert_indices)
        self.base_tri_count = len(self.tris_left)
        
        #if the full resolution mesh is less than 10k tris
        # or the stream is less than 20% of the total
        # then don't bother with the stream at all
        if self.orig_tri_count < TRIANGLE_MINIMUM or float(self.orig_tri_count - self.base_tri_count) / self.orig_tri_count < STREAM_THRESHOLD:
            self.base_tri_count = self.orig_tri_count
            self.add_back_pm()
        else:
            self.split_base_and_pm()
        
        return self.save_mesh()

def FilterGenerator():
    class SandlerSimplificationFilter(SimplifyFilter):
        def __init__(self):
            super(SandlerSimplificationFilter, self).__init__('sander_simplify', 'Simplifies the mesh based on sandler, et al. method.')
            self.arguments.append(FileArgument('pm_file', 'Where to save the progressive mesh stream'))
        def apply(self, mesh, pm_file):
            try:
                pmout = open(pm_file, 'w')
            except TypeError:
                pmout = pm_file
            
            s = SanderSimplify(mesh, pmout)
            if USE_IPDB:
                with launch_ipdb_on_exception():
                    mesh = s.simplify()
            else:
                mesh = s.simplify()
            return mesh
    return SandlerSimplificationFilter()
from meshtool.filters import factory
factory.register(FilterGenerator().name, FilterGenerator)
