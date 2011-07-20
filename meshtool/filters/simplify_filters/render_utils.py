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

def gen_color2(N):
    import colorsys
    HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in range(N)]
    for x in HSV_tuples:
        yield map(lambda x:int(x * 256), colorsys.hsv_to_rgb(*x))

def gen_color():
    """generator for getting n of evenly distributed colors using
    hsv color and golden ratio. It always return same order of colors
 
     Taken from: http://www.python-blog.com/2011/05/17/evenly-distributed-random-color-generator-in-python/
 
    :returns: RGB tuple
    """
    import colorsys
    golden_ratio = 0.618033988749895
    h = 0.22717784590367374
 
    while 1:
        h += golden_ratio
        h %= 1
        HSV_tuple = [h, 0.95, 0.95]  # this defines how "deep" are the colors
        RGB_tuple = colorsys.hsv_to_rgb(*HSV_tuple)
        yield map(lambda x:int(x * 256), RGB_tuple)

def renderCharts(facegraph, verts):
    
    from meshtool.filters.panda_filters.pandacore import getVertexData, attachLights, ensureCameraAt
    from meshtool.filters.panda_filters.pandacontrols import KeyboardMovement, MouseDrag, MouseScaleZoom
    from panda3d.core import GeomTriangles, Geom, GeomNode, GeomVertexFormat, GeomVertexData, GeomVertexWriter
    from direct.showbase.ShowBase import ShowBase
       
    format=GeomVertexFormat.getV3c4()
    vdata=GeomVertexData('square', format, Geom.UHDynamic)

    vertex=GeomVertexWriter(vdata, 'vertex')
    color=GeomVertexWriter(vdata, 'color')
    
    colors = gen_color()
    numtris = 0
    for chart, data in facegraph.nodes_iter(data=True):
        curcolor = next(colors)
        for tri in data['tris']:
            triv = verts[tri]
            vertex.addData3f(triv[0][0], triv[0][1], triv[0][2])
            vertex.addData3f(triv[1][0], triv[1][1], triv[1][2])
            vertex.addData3f(triv[2][0], triv[2][1], triv[2][2])
            color.addData4f(curcolor[0],curcolor[1], curcolor[2], 1)
            color.addData4f(curcolor[0],curcolor[1], curcolor[2], 1)
            color.addData4f(curcolor[0],curcolor[1], curcolor[2], 1)
            numtris += 1

    tris=GeomTriangles(Geom.UHDynamic)
    tris.addConsecutiveVertices(0, 3*numtris)
    tris.closePrimitive()

    pgeom = Geom(vdata)
    pgeom.addPrimitive(tris)
    node = GeomNode("primitive")
    node.addGeom(pgeom)
    p3dApp = ShowBase()
    #attachLights(render)
    geomPath = render.attachNewNode(node)
    #geomPath.setRenderModeWireframe()
    ensureCameraAt(geomPath, base.camera)
    KeyboardMovement()
    #render.setShaderAuto()
    p3dApp.run()