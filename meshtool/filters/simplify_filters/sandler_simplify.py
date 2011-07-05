from meshtool.args import *
from meshtool.filters.base_filters import *
import inspect
import numpy
import networkx as nx
from itertools import izip, chain, repeat
import datetime

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

def getAdjacentFaces(vgraph, verts, facenum):
    faces = {}
    for i in range(len(verts)):
        for j in range(i+1,len(verts)):
            faces = dict(faces.items() + vgraph.edge[verts[i]][verts[j]].items())
    if facenum in faces:
        del faces[facenum]
    return faces.keys()

def getAdjacentEdges(vgraph, idx):
    for i, tri in enumerate(idx):
        for adjacent_face in getAdjacentFaces(vgraph, tri, i):
            yield (adjacent_face, i)
                            
def sandler_simplify(mesh):
    all_vertices = []
    all_indices = []
    vertex_offset = 0
    t = timer()
    
    print 'building aggregated vertex and triangle list...',
    for boundgeom in mesh.scene.objects('geometry'):
        for boundprim in boundgeom.primitives():
            all_vertices.append(boundprim.vertex)
            all_indices.append(boundprim.vertex_index + vertex_offset)
            vertex_offset += len(boundprim.vertex)
            
    all_vertices = numpy.concatenate(all_vertices)
    all_indices = numpy.concatenate(all_indices)
    
    print next(t)
    print 'uniqifying the list...',
    unique_data, index_map = numpy.unique(all_vertices.view([('',all_vertices.dtype)]*all_vertices.shape[1]), return_inverse=True)
    all_vertices = unique_data.view(all_vertices.dtype).reshape(-1,all_vertices.shape[1])
    all_indices = index_map[all_indices]
    
    print next(t)
    print 'building vertex vertices...',
    vertexgraph = nx.Graph()
    vertexgraph.add_nodes_from(xrange(len(all_vertices)))
    print next(t)
    print 'building vertex edges...',
    vertexgraph.add_edges_from(( (edge[0], edge[1], {facenum:True})
                                 for edge, facenum in
                                 izip(all_indices[:,(0,1)], xrange(len(all_indices))) ))
    vertexgraph.add_edges_from(( (edge[0], edge[1], {facenum:True})
                                 for edge, facenum in
                                 izip(all_indices[:,(0,2)], xrange(len(all_indices))) ))
    vertexgraph.add_edges_from(( (edge[0], edge[1], {facenum:True})
                                 for edge, facenum in
                                 izip(all_indices[:,(1,2)], xrange(len(all_indices))) ))

    print next(t)
    print 'building face vertices...',
    facegraph = nx.Graph()
    facegraph.add_nodes_from(( (i, {'vertices':[tri[0], tri[1], tri[2]]})
                               for i, tri in
                               enumerate(all_indices) ))
    print next(t)
    print 'building face edges...',
    facegraph.add_edges_from(getAdjacentEdges(vertexgraph, all_indices))
    
    print next(t)

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