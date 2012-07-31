def renderVerts(verts, idx):
    
    from meshtool.filters.panda_filters.pandacore import getVertexData, attachLights, ensureCameraAt
    from meshtool.filters.panda_filters.pandacontrols import KeyboardMovement, MouseDrag, MouseScaleZoom
    from panda3d.core import GeomTriangles, Geom, GeomNode
    from direct.showbase.ShowBase import ShowBase
    
    vdata, indexdata = getVertexData(verts, idx)
    gprim = GeomTriangles(Geom.UHStatic)
    gprim.setIndexType(Geom.NTUint32)
    gprim.setVertices(indexdata)
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

def gen_color3(N):
    import matplotlib.cm
    import random
    
    cm = matplotlib.cm.get_cmap('Accent')
    N = float(N)
    colors = [map(lambda x:int(x * 256), cm(i/N)) for i in range(int(N))]
    random.shuffle(colors)
    for i in range(int(N)):
        yield colors[i]

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

def renderCharts(facegraph, verts, vert_indices, lineset=None):
    
    from meshtool.filters.panda_filters.pandacore import getVertexData, attachLights, ensureCameraAt
    from meshtool.filters.panda_filters.pandacontrols import KeyboardMovement, MouseDrag, MouseScaleZoom, ButtonUtils
    from panda3d.core import GeomTriangles, Geom, GeomNode, GeomVertexFormat, GeomVertexData, GeomVertexWriter, LineSegs
    from direct.showbase.ShowBase import ShowBase
       
    vformat = GeomVertexFormat.getV3c4()
    vdata=GeomVertexData('tris', vformat, Geom.UHDynamic)

    vertex=GeomVertexWriter(vdata, 'vertex')
    color=GeomVertexWriter(vdata, 'color')
    
    colors = gen_color3(len(facegraph))
    numtris = 0
    for chart, data in facegraph.nodes_iter(data=True):
        curcolor = next(colors)
        for tri in data['tris']:
            triv = verts[vert_indices[tri]]
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
        
    linenodes = []
    if lineset:
        for lines in lineset:
            ls = LineSegs()
            ls.setThickness(4)
            curcolor = next(colors)
            ls.setColor(curcolor[0]/256.0, curcolor[1]/256.0, curcolor[2]/256.0, 1)
    
            tuples = False
            for blah in lines:
                if isinstance(blah, tuple):
                    tuples = True
                break
            if tuples:
                for i, j in lines:
                    frompt = verts[i]
                    topt = verts[j]
                    ls.moveTo(frompt[0], frompt[1], frompt[2])
                    ls.drawTo(topt[0], topt[1], topt[2])
            else:
                for i in range(len(lines)-1):
                    frompt = verts[lines[i]]
                    topt = verts[lines[i+1]]
                    ls.moveTo(frompt[0], frompt[1], frompt[2])
                    ls.drawTo(topt[0], topt[1], topt[2])
            
            linenodes.append(ls.create())
        

    pgeom = Geom(vdata)
    pgeom.addPrimitive(tris)

    node = GeomNode("primitive")
    node.addGeom(pgeom)

    p3dApp = ShowBase()
    #attachLights(render)
    geomPath = render.attachNewNode(node)

    for linenode in linenodes:
        geomPath.attachNewNode(linenode)
    
    #geomPath.setRenderModeWireframe()
    
    ensureCameraAt(geomPath, base.cam)
    
    boundingSphere = geomPath.getBounds()
    base.cam.setPos(boundingSphere.getCenter() + boundingSphere.getRadius())

    base.cam.lookAt(boundingSphere.getCenter())
    
    KeyboardMovement()
    ButtonUtils(geomPath)
    MouseDrag(geomPath)
    MouseScaleZoom(geomPath)
    #render.setShaderAuto()
    p3dApp.run()