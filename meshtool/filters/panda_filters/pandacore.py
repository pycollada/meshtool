import numpy
import collada
import posixpath
import struct
from math import pi, sin, cos
import Image
import ImageOps
from StringIO import StringIO

from direct.task import Task
from direct.showbase.ShowBase import ShowBase
from panda3d.core import GeomVertexFormat, GeomVertexArrayData
from panda3d.core import GeomVertexData, GeomEnums, GeomVertexWriter
from panda3d.core import GeomLines, GeomTriangles, Geom, GeomNode, NodePath
from panda3d.core import PNMImage, Texture, StringStream
from panda3d.core import RenderState, TextureAttrib, MaterialAttrib, Material
from panda3d.core import TextureStage, RigidBodyCombiner, CullFaceAttrib, TransparencyAttrib
from panda3d.core import VBase4, Vec4, Mat4, SparseArray, Vec3
from panda3d.core import AmbientLight, DirectionalLight, PointLight, Spotlight
from panda3d.core import Character, PartGroup, CharacterJoint
from panda3d.core import TransformBlend, TransformBlendTable, JointVertexTransform
from panda3d.core import GeomVertexAnimationSpec, GeomVertexArrayFormat, InternalName
from panda3d.core import AnimBundle, AnimGroup, AnimChannelMatrixXfmTable
from panda3d.core import PTAFloat, CPTAFloat, AnimBundleNode, DepthOffsetAttrib
from direct.actor.Actor import Actor
from panda3d.core import loadPrcFileData

def getNodeFromController(controller, controlled_prim):
    if type(controlled_prim) is collada.controller.BoundSkinPrimitive:
        ch = Character('simplechar')
        bundle = ch.getBundle(0)
        skeleton = PartGroup(bundle, '<skeleton>')

        character_joints = {}
        for (name, joint_matrix) in controller.joint_matrices.iteritems():
            joint_matrix.shape = (-1)
            character_joints[name] = CharacterJoint(ch, bundle, skeleton, name, Mat4(*joint_matrix)) 
        
        tbtable = TransformBlendTable()
        
        for influence in controller.index:
            blend = TransformBlend()
            for (joint_index, weight_index) in influence:
                char_joint = character_joints[controller.getJoint(joint_index)]
                weight = controller.getWeight(weight_index)[0]
                blend.addTransform(JointVertexTransform(char_joint), weight)
            tbtable.addBlend(blend)
            
        array = GeomVertexArrayFormat()
        array.addColumn(InternalName.make('vertex'), 3, Geom.NTFloat32, Geom.CPoint)
        array.addColumn(InternalName.make('normal'), 3, Geom.NTFloat32, Geom.CPoint)
        array.addColumn(InternalName.make('texcoord'), 2, Geom.NTFloat32, Geom.CTexcoord)
        blendarr = GeomVertexArrayFormat()
        blendarr.addColumn(InternalName.make('transform_blend'), 1, Geom.NTUint16, Geom.CIndex)
        
        format = GeomVertexFormat()
        format.addArray(array)
        format.addArray(blendarr)
        aspec = GeomVertexAnimationSpec()
        aspec.setPanda()
        format.setAnimation(aspec)
        format = GeomVertexFormat.registerFormat(format)
        
        dataname = controller.id + '-' + controlled_prim.primitive.material.id
        vdata = GeomVertexData(dataname, format, Geom.UHStatic)
        vertex = GeomVertexWriter(vdata, 'vertex')
        normal = GeomVertexWriter(vdata, 'normal')
        texcoord = GeomVertexWriter(vdata, 'texcoord')
        transform = GeomVertexWriter(vdata, 'transform_blend') 
        
        numtris = 0
        if type(controlled_prim.primitive) is collada.polylist.BoundPolylist:
            for poly in controlled_prim.primitive.polygons():
                for tri in poly.triangles():
                    for tri_pt in range(3):
                        vertex.addData3f(tri.vertices[tri_pt][0], tri.vertices[tri_pt][1], tri.vertices[tri_pt][2])
                        normal.addData3f(tri.normals[tri_pt][0], tri.normals[tri_pt][1], tri.normals[tri_pt][2])
                        if len(controlled_prim.primitive._texcoordset) > 0:
                            texcoord.addData2f(tri.texcoords[0][tri_pt][0], tri.texcoords[0][tri_pt][1])
                        transform.addData1i(tri.indices[tri_pt])
                    numtris+=1
        elif type(controlled_prim.primitive) is collada.triangleset.BoundTriangleSet:
            for tri in controlled_prim.primitive.triangles():
                for tri_pt in range(3):
                    vertex.addData3f(tri.vertices[tri_pt][0], tri.vertices[tri_pt][1], tri.vertices[tri_pt][2])
                    normal.addData3f(tri.normals[tri_pt][0], tri.normals[tri_pt][1], tri.normals[tri_pt][2])
                    if len(controlled_prim.primitive._texcoordset) > 0:
                        texcoord.addData2f(tri.texcoords[0][tri_pt][0], tri.texcoords[0][tri_pt][1])
                    transform.addData1i(tri.indices[tri_pt])
                numtris+=1
                    
        tbtable.setRows(SparseArray.lowerOn(vdata.getNumRows())) 
        
        gprim = GeomTriangles(Geom.UHStatic)
        for i in range(numtris):
            gprim.addVertices(i*3, i*3+1, i*3+2)
            gprim.closePrimitive()
            
        pgeom = Geom(vdata)
        pgeom.addPrimitive(gprim)
        
        render_state = getStateFromMaterial(controlled_prim.primitive.material)
        control_node = GeomNode("ctrlnode")
        control_node.addGeom(pgeom, render_state)
        ch.addChild(control_node)
    
        bundle = AnimBundle('simplechar', 5.0, 2)
        skeleton = AnimGroup(bundle, '<skeleton>')
        root = AnimChannelMatrixXfmTable(skeleton, 'root')
        
        #hjoint = AnimChannelMatrixXfmTable(root, 'joint1') 
        #table = [10, 11, 12, 13, 14, 15, 14, 13, 12, 11] 
        #data = PTAFloat.emptyArray(len(table)) 
        #for i in range(len(table)): 
        #    data.setElement(i, table[i]) 
        #hjoint.setTable(ord('i'), CPTAFloat(data)) 
        
        #vjoint = AnimChannelMatrixXfmTable(hjoint, 'joint2') 
        #table = [10, 9, 8, 7, 6, 5, 6, 7, 8, 9] 
        #data = PTAFloat.emptyArray(len(table)) 
        #for i in range(len(table)): 
        #    data.setElement(i, table[i]) 
        #vjoint.setTable(ord('j'), CPTAFloat(data)) 

        wiggle = AnimBundleNode('wiggle', bundle)

        np = NodePath(ch) 
        anim = NodePath(wiggle) 
        a = Actor(np, {'simplechar' : anim})
        a.loop('simplechar') 
        return a
        #a.setPos(0, 0, 0)
    
    else:
        raise Exception("Error: unsupported controller type")

def getVertexData(vertex, vertex_index, normal=None, normal_index=None,
                  texcoordset=(), texcoord_indexset=(),
                  textangentset=(), textangent_indexset=(),
                  texbinormalset=(), texbinormal_indexset=()):
    
    format = GeomVertexFormat()
    formatArray = GeomVertexArrayFormat() 
    
    vertex_data = vertex[vertex_index]
    vertex_data.shape = (-1, 3)
    stacked = vertex_data
    vertex_data = None
    formatArray.addColumn(InternalName.make("vertex"), 3, Geom.NTFloat32, Geom.CPoint) 
    
    if normal is not None:
        normal_data = normal[normal_index]
        normal_data.shape = (-1, 3)
        collada.util.normalize_v3(normal_data)
        stacked = numpy.hstack((stacked, normal_data))
        normal_data = None
        formatArray.addColumn(InternalName.make("normal"), 3, Geom.NTFloat32, Geom.CVector)
    if len(texcoordset) > 0:
        texcoord_data = texcoordset[0][texcoord_indexset[0]]
        texcoord_data.shape = (-1, 2)
        stacked = numpy.hstack((stacked, texcoord_data))
        texcoord_data = None
        formatArray.addColumn(InternalName.make("texcoord"), 2, Geom.NTFloat32, Geom.CTexcoord)
    if len(textangentset) > 0:
        textangent_data = textangentset[0][textangent_indexset[0]]
        textangent_data.shape = (-1, 3)
        stacked = numpy.hstack((stacked, textangent_data))
        textangent_data = None
        formatArray.addColumn(InternalName.make("tangent"), 3, Geom.NTFloat32, Geom.CVector)
    if len(texbinormalset) > 0:
        texbinormal_data = texbinormalset[0][texbinormal_indexset[0]]
        texbinormal_data.shape = (-1, 3)
        stacked = numpy.hstack((stacked, texbinormal_data))
        texbinormal_data = None
        formatArray.addColumn(InternalName.make("binormal"), 3, Geom.NTFloat32, Geom.CVector)

    stacked = stacked.flatten()
    stacked.shape = (-1)
    all_data = stacked.tostring()
    stacked = None

    format.addArray(formatArray)
    format = GeomVertexFormat.registerFormat(format)
    
    vdata = GeomVertexData("dataname", format, Geom.UHStatic)
    arr = GeomVertexArrayData(format.getArray(0), GeomEnums.UHStream)
    datahandle = arr.modifyHandle()
    datahandle.setData(all_data)
    all_data = None
    vdata.setArray(0, arr)
    datahandle = None
    arr = None
    
    return vdata

def getPrimAndDataFromTri(triset, matstate):
    if triset.normal is None:
        triset.generateNormals()

    needsTanAndBin = False
    if matstate is not None and matstate.hasAttrib(TextureAttrib):
        texattr = matstate.getAttrib(TextureAttrib)
        for i in range(texattr.getNumOnStages()):
            if texattr.getOnStage(i).getMode() == TextureStage.MNormal:
                needsTanAndBin = True
    
    if needsTanAndBin and isinstance(triset, collada.triangleset.TriangleSet) and \
            len(triset.texcoordset) > 0 and len(triset.textangentset) == 0:
        triset.generateTexTangentsAndBinormals()

    vdata = getVertexData(triset.vertex, triset.vertex_index,
                          triset.normal, triset.normal_index,
                          triset.texcoordset, triset.texcoord_indexset,
                          triset.textangentset, triset.textangent_indexset,
                          triset.texbinormalset, triset.texbinormal_indexset)

    gprim = GeomTriangles(Geom.UHStatic)
    gprim.addConsecutiveVertices(0, 3*triset.ntriangles)
    gprim.closePrimitive()
    
    return (vdata, gprim)

def textureFromData(image_data, filename=""):
    tex = None
    
    if image_data:
        myTexture = Texture()
        
        myImage = PNMImage()
        success = myImage.read(StringStream(image_data), filename)
        
        if success == 1:
            #PNMImage can handle most texture formats
            myTexture.load(myImage)
        else:
            #Except for DDS, which PNMImage.read will return 0, so try to load as DDS
            success = myTexture.readDds(StringStream(image_data))
            
        if success != 0:
            tex = myTexture
            
    return tex

def pilFromData(image_data):
    try:
        im = Image.open(StringIO(image_data))
        im.load()
    except IOError:
        #PIL couldn't open, so try to read with panda3d which supports DDS:
        im = None
        tex = textureFromData(image_data)
        if tex is not None:
            outdata = tex.getRamImageAs('RGB').getData()
            try:
                im = Image.fromstring('RGB', (tex.getXSize(), tex.getYSize()), outdata)
                im.load()
            except IOError:
                #Any problem with panda3d might generate an invalid image buffer, so don't convert this
                im = None
    return im

def getTexture(color=None, alpha=None, texture_cache=None):

    unique_id = ""
    if color:
        unique_id += str(id(color.sampler.surface.image))
    if alpha:
        unique_id += '_' + str(id(alpha.sampler.surface.image))

    if texture_cache is not None:
        if unique_id in texture_cache:
            return texture_cache[unique_id]
    
    image_file = ""
    if alpha:
        im = pilFromData(alpha.sampler.surface.image.data)
        
        gray = None
        for i, band in enumerate(im.getbands()):
            if band == 'A':
                gray = im.split()[i]
        if gray is None:
            gray = ImageOps.grayscale(im)
            
        if color:
            newim = pilFromData(color.sampler.surface.image.data)
            if 'A' not in newim.getbands():
                newim = newim.convert('RGBA')
        else:
            newim = Image.new('RGBA', im.size)
        newim.putalpha(gray)
        newbuf = StringIO()
        newim.save(newbuf, 'PNG')
        image_data = newbuf.getvalue()
        image_file = 'whatever.png'
    else:
        image_file = posixpath.basename(color.sampler.surface.image.path)
        im = pilFromData(color.sampler.surface.image.data)
        if im:
            if 'A' in im.getbands():
                im = im.convert('RGBA')
            else:
                im = im.convert('RGB')
            newbuf = StringIO()
            im.save(newbuf, 'PNG')
            image_data = newbuf.getvalue()
        else:
            image_data = color.sampler.surface.image.data
    
    tex = textureFromData(image_data, image_file)

    if texture_cache is not None:
        texture_cache[unique_id] = tex
    return tex

def v4fromtuple(value):
    val4 = value[3] if len(value) > 3 else 1.0
    return VBase4(value[0], value[1], value[2], val4)

def addTextureStage(texId, texMode, texAttr, tex):
    if tex:
        ts = TextureStage(texId)
        ts.setMode(texMode)
        texAttr = texAttr.addOnStage(ts, tex)
    return texAttr

def getStateFromMaterial(prim_material, texture_cache):
    state = RenderState.makeFullDefault()
    
    mat = Material()
    texattr = TextureAttrib.makeAllOff()
    
    hasDiffuse = False
    if prim_material and prim_material.effect:
        
        diffuse = getattr(prim_material.effect, 'diffuse')
        transparent = getattr(prim_material.effect, 'transparent')
        if isinstance(diffuse, collada.material.Map) and isinstance(transparent, collada.material.Map):
            if diffuse.sampler.surface.image == transparent.sampler.surface.image:
                #some exporters put the same map in the diffuse channel
                # and the transparent channel when they don't really mean to
                transparent = None
        
        if isinstance(diffuse, collada.material.Map) or isinstance(transparent, collada.material.Map):
            diffuseMap = None
            transparentMap = None
            if isinstance(diffuse, collada.material.Map):
                diffuseMap = diffuse
            if isinstance(transparent, collada.material.Map):
                transparentMap = transparent
            if diffuseMap == transparentMap:
                transparentMap = None
                
            diffuseTexture = getTexture(color=diffuseMap, alpha=transparentMap, texture_cache=texture_cache)
            texattr = addTextureStage('tsDiff', TextureStage.MModulate, texattr, diffuseTexture)
            hasDiffuse = True

        if type(diffuse) is tuple:
            mat.setDiffuse(v4fromtuple(diffuse))
        
        emission = getattr(prim_material.effect, 'emission')
        if isinstance(emission, collada.material.Map):
            emissionTexture = getTexture(alpha=emission, texture_cache=texture_cache)
            texattr = addTextureStage('tsEmiss', TextureStage.MGlow, texattr, emissionTexture)
        elif type(emission) is tuple:
            mat.setEmission(v4fromtuple(emission))
        
        ambient = getattr(prim_material.effect, 'ambient')
        if type(ambient) is tuple:
            mat.setAmbient(v4fromtuple(ambient))
        
        specular = getattr(prim_material.effect, 'specular')
        if isinstance(specular, collada.material.Map):
            specularTexture = getTexture(color=specular, texture_cache=texture_cache)
            texattr = addTextureStage('tsSpec', TextureStage.MGloss, texattr, specularTexture)
            mat.setSpecular(VBase4(0.1, 0.1, 0.1, 1.0))
        elif type(specular) is tuple:
            mat.setSpecular(v4fromtuple(specular))

        shininess = getattr(prim_material.effect, 'shininess')
        #this sets a sane value for blinn shading
        if shininess <= 1.0:
            if shininess < 0.01:
                shininess = 1.0
            shininess = shininess * 128.0
        mat.setShininess(shininess)

        bumpmap = getattr(prim_material.effect, 'bumpmap')
        if isinstance(bumpmap, collada.material.Map):
            bumpTexture = getTexture(color=bumpmap, texture_cache=texture_cache)
            texattr = addTextureStage('tsBump', TextureStage.MNormal, texattr, bumpTexture)

        if prim_material.effect.double_sided:
            state = state.addAttrib(CullFaceAttrib.make(CullFaceAttrib.MCullNone))

    if hasDiffuse:
        state = state.addAttrib(DepthOffsetAttrib.make(1))

    state = state.addAttrib(MaterialAttrib.make(mat))
    state = state.addAttrib(texattr)
    
    return state

def setCameraAngle(ang):
    base.camera.setPos(2000.0 * sin(ang), -2000.0 * cos(ang), 0)
    base.camera.lookAt(0.0, 0.0, 0.0)

def spinCameraTask(task):
    speed = 5.0
    curSpot = task.time % speed
    angleDegrees = (curSpot / speed) * 360
    angleRadians = angleDegrees * (pi / 180.0)
    setCameraAngle(angleRadians)
    return Task.cont

def ColorToVec4(colorstring):
    r, g, b = colorstring[:2], colorstring[2:4], colorstring[4:]
    r, g, b = [int(n, 16) for n in (r, g, b)]
    r, g, b = [n / 255.0 for n in (r, g, b)]
    return Vec4(r, g, b, 1)

def rotsToMat4(x, y, z):
    rotx = Mat4(1,0,0,0,0,cos(x),-sin(x),0,0,sin(x),cos(x),0,0,0,0,1);
    roty = Mat4(cos(y),0,sin(y),0,0,1,0,0,-sin(y),0,cos(y),0,0,0,0,1);
    rotz = Mat4(cos(z),-sin(z),0,0,sin(z),cos(z),0,0,0,0,1,0,0,0,0,1);
    swapyz = Mat4.rotateMat(90, Vec3(-1,0,0))
    return rotx * roty * rotz * swapyz

def attachLights(render):
    dl = DirectionalLight('dirLight')
    dl.setColor(ColorToVec4('666060'))
    dlNP = render.attachNewNode(dl)
    dlNP.setHpr(0, -45, 0)
    render.setLight(dlNP)
    
    dl = DirectionalLight('dirLight')
    dl.setColor(ColorToVec4('606666'))
    dlNP = render.attachNewNode(dl)
    dlNP.setHpr(180, 45, 0)
    render.setLight(dlNP)
    
    dl = DirectionalLight('dirLight')
    dl.setColor(ColorToVec4('606060'))
    dlNP = render.attachNewNode(dl)
    dlNP.setHpr(90, -45, 0)
    render.setLight(dlNP)
    
    dl = DirectionalLight('dirLight')
    dl.setColor(ColorToVec4('626262'))
    dlNP = render.attachNewNode(dl)
    dlNP.setHpr(-90, 45, 0)
    render.setLight(dlNP)
    
    ambientLight = AmbientLight('ambientLight')
    ambientLight.setColor(Vec4(0.2, 0.2, 0.2, 1))
    ambientLightNP = render.attachNewNode(ambientLight)
    render.setLight(ambientLightNP)

def getBaseNodePath(render):
    globNode = render.find("collada")
    if globNode.isEmpty():
        globNode = GeomNode("collada")
        globNode = render.attachNewNode(globNode)
    return globNode

def destroyScene(render):
    base.taskMgr.stop()
    for child in render.getChildren():
        if child != base.camera:
            child.removeNode()
    base.graphicsEngine.removeAllWindows()

def ensureCameraAt(nodePath, cam):
    if nodePath.getNumChildren() > 0:
        boundingSphere = nodePath.getBounds()
        if not boundingSphere.isEmpty():
            scale = 500 / boundingSphere.getRadius()
            
            nodePath.setScale(scale, scale, scale)
            boundingSphere = nodePath.getBounds()
            nodePath.setPos(-1 * boundingSphere.getCenter().getX(),
                            -1 * boundingSphere.getCenter().getY(),
                            -1 * boundingSphere.getCenter().getZ())
            nodePath.setHpr(0,0,0)
       
    cam.setPos(1500, -1500, 1)
    cam.lookAt(0.0, 0.0, 0.0)

def getGeomFromPrim(prim, matstate):
    if type(prim) is collada.triangleset.TriangleSet:
        (vdata, gprim) = getPrimAndDataFromTri(prim, matstate)
    elif type(prim) is collada.polylist.Polylist or type(prim) is collada.polygons.Polygons:
        triset = prim.triangleset()
        (vdata, gprim) = getPrimAndDataFromTri(triset, matstate)
    elif type(prim) is collada.lineset.LineSet:
        vdata = getVertexData(prim.vertex, prim.vertex_index)           
        gprim = GeomLines(Geom.UHStatic)
        gprim.addConsecutiveVertices(0, 2*prim.nlines)
        gprim.closePrimitive()
    else:
        raise Exception("Error: Unsupported primitive type. Exiting.")

    pgeom = Geom(vdata)
    pgeom.addPrimitive(gprim)
    return pgeom

def recurseScene(curnode, scene_members, data_cache, M, texture_cache):
    M = numpy.dot(M, curnode.matrix)
    for node in curnode.children:
        if isinstance(node, collada.scene.Node):
            recurseScene(node, scene_members, data_cache, M, texture_cache)
        elif isinstance(node, collada.scene.GeometryNode) or isinstance(node, collada.scene.ControllerNode):
            if isinstance(node, collada.scene.GeometryNode):
                geom = node.geometry
            else:
                geom = node.controller.geometry
            
            materialnodesbysymbol = {}
            for mat in node.materials:
                materialnodesbysymbol[mat.symbol] = mat

            for prim in geom.primitives:
                if len(prim) > 0:
                    mat = materialnodesbysymbol.get(prim.material)
                    matstate = None
                    if mat is not None:
                        matstate = data_cache['material2state'].get(mat.target)
                        if matstate is None:
                            matstate = getStateFromMaterial(mat.target, texture_cache)
                            if geom.double_sided:
                                matstate = matstate.addAttrib(CullFaceAttrib.make(CullFaceAttrib.MCullNone))
                            data_cache['material2state'][mat.target] = matstate
                    
                    mat4 = Mat4(*M.T.flatten().tolist())
                    
                    primgeom = data_cache['prim2geom'].get(prim)
                    if primgeom is None:
                        primgeom = getGeomFromPrim(prim, matstate)
                        data_cache['prim2geom'][prim] = primgeom
                    
                    scene_members.append((primgeom, matstate, mat4))

def getSceneMembers(col):
    #caches materials and geoms so they can be reused
    data_cache = {
      'material2state' : {},
      #maps material.Material to RenderState
      
      'prim2geom' : {}
      #maps primitive.Primitive to Geom
    }
    
    #stores tuples of Geom, RenderState, and Mat4 transform matrix
    scene_members = []
    
    m = numpy.identity(4)
    texture_cache = {}
    if col.scene is not None:
        for node in col.scene.nodes:
            recurseScene(node, scene_members, data_cache, m, texture_cache)
    
    return scene_members

def setupPandaApp(mesh):
    scene_members = getSceneMembers(mesh)
    
    p3dApp = ShowBase()
    nodePath = getBaseNodePath(render)
    
    rotateNode = GeomNode("rotater")
    rotatePath = nodePath.attachNewNode(rotateNode)
    matrix = numpy.identity(4)
    if mesh.assetInfo['up_axis'] == 'X_UP':
        r = collada.scene.RotateTransform(0,1,0,90)
        matrix = r.matrix
    elif mesh.assetInfo['up_axis'] == 'Y_UP':
        r = collada.scene.RotateTransform(1,0,0,90)
        matrix = r.matrix
    rotatePath.setMat(Mat4(*matrix.T.flatten().tolist()))
    
    rbc = RigidBodyCombiner('combiner')
    rbcPath = rotatePath.attachNewNode(rbc)
    
    for geom, renderstate, mat4 in scene_members:
        node = GeomNode("primitive")
        node.addGeom(geom)
        if renderstate is not None:
            node.setGeomState(0, renderstate)
        geomPath = rbcPath.attachNewNode(node)
        geomPath.setMat(mat4)
        
    rbc.collect()
        
    ensureCameraAt(nodePath, base.camera)
    base.disableMouse()
    attachLights(render)
    render.setShaderAuto()
    render.setTransparency(TransparencyAttrib.MDual, 1)

    return p3dApp

def getScreenshot(p3dApp):

    p3dApp.taskMgr.step()
    p3dApp.taskMgr.step()
    pnmss = PNMImage()
    p3dApp.win.getScreenshot(pnmss)
    resulting_ss = StringStream()
    pnmss.write(resulting_ss, "screenshot.png")
    screenshot_buffer = resulting_ss.getData()
    pilimage = Image.open(StringIO(screenshot_buffer))
    pilimage.load()
    
    #pnmimage will sometimes output as palette mode for 8-bit png so convert
    pilimage = pilimage.convert('RGBA')
    return pilimage
