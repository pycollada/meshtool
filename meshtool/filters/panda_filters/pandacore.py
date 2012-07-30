import numpy
import collada
import posixpath
import struct
from math import pi, sin, cos
from meshtool.util import Image, ImageOps
from StringIO import StringIO
import inspect
import math

from direct.task import Task
from direct.showbase.ShowBase import ShowBase
from panda3d.core import GeomVertexFormat, GeomVertexArrayData
from panda3d.core import GeomVertexData, GeomEnums, GeomVertexWriter
from panda3d.core import GeomLines, GeomTriangles, Geom, GeomNode, NodePath
from panda3d.core import PNMImage, Texture, StringStream
from panda3d.core import RenderState, TextureAttrib, MaterialAttrib, Material
from panda3d.core import TextureStage, RigidBodyCombiner, CullFaceAttrib, TransparencyAttrib, ColorScaleAttrib
from panda3d.core import VBase4, Vec4, Mat4, SparseArray, Vec3
from panda3d.core import AmbientLight, DirectionalLight, PointLight, Spotlight
from panda3d.core import Character, PartGroup, CharacterJoint
from panda3d.core import TransformBlend, TransformBlendTable, JointVertexTransform
from panda3d.core import GeomVertexAnimationSpec, GeomVertexArrayFormat, InternalName
from panda3d.core import AnimBundle, AnimGroup, AnimChannelMatrixXfmTable
from panda3d.core import PTAFloat, CPTAFloat, AnimBundleNode, DepthOffsetAttrib
from direct.actor.Actor import Actor
from panda3d.core import loadPrcFileData

#after numpy 1.3, unique1d was renamed to unique
args, varargs, keywords, defaults = inspect.getargspec(numpy.unique)    
if 'return_inverse' not in args:
    numpy.unique = numpy.unique1d

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
    
    indices2stack = [vertex_index.reshape(-1, 1)]
    alldata = [vertex]
    formatArray.addColumn(InternalName.make("vertex"), 3, Geom.NTFloat32, Geom.CPoint)
    if normal is not None:
        indices2stack.append(normal_index.reshape(-1, 1))
        alldata.append(collada.util.normalize_v3(numpy.copy(normal)))
        formatArray.addColumn(InternalName.make("normal"), 3, Geom.NTFloat32, Geom.CVector)
    if len(texcoordset) > 0:
        indices2stack.append(texcoord_indexset[0].reshape(-1, 1))
        alldata.append(texcoordset[0])
        formatArray.addColumn(InternalName.make("texcoord"), 2, Geom.NTFloat32, Geom.CTexcoord)
    if len(textangentset) > 0:
        indices2stack.append(textangent_indexset[0].reshape(-1, 1))
        alldata.append(textangentset[0])
        formatArray.addColumn(InternalName.make("tangent"), 3, Geom.NTFloat32, Geom.CVector)
    if len(texbinormalset) > 0:
        indices2stack.append(texbinormal_indexset[0].reshape(-1, 1))
        alldata.append(texbinormalset[0])
        formatArray.addColumn(InternalName.make("binormal"), 3, Geom.NTFloat32, Geom.CVector)
        
    #have to flatten and reshape like this so that it's contiguous
    stacked_indices = numpy.hstack(indices2stack).flatten().reshape((-1, len(indices2stack)))

    #index_map - maps each unique value back to a location in the original array it came from
    #   eg. stacked_indices[index_map] == unique_stacked_indices
    #inverse_map - maps original array locations to their location in the unique array
    #   e.g. unique_stacked_indices[inverse_map] == stacked_indices
    unique_stacked_indices, index_map, inverse_map = numpy.unique(stacked_indices.view([('',stacked_indices.dtype)]*stacked_indices.shape[1]), return_index=True, return_inverse=True)
    unique_stacked_indices = unique_stacked_indices.view(stacked_indices.dtype).reshape(-1,stacked_indices.shape[1])
    
    #unique returns as int64, so cast back
    index_map = numpy.cast['uint32'](index_map)
    inverse_map = numpy.cast['uint32'](inverse_map)
    
    #sort the index map to get a list of the index of the first time each value was encountered
    sorted_map = numpy.cast['uint32'](numpy.argsort(index_map))
    
    #since we're sorting the unique values, we have to map the inverse_map to the new index locations
    backwards_map = numpy.zeros_like(sorted_map)
    backwards_map[sorted_map] = numpy.arange(len(sorted_map), dtype=numpy.uint32)
    
    #now this is the new unique values and their indices
    unique_stacked_indices = unique_stacked_indices[sorted_map]
    inverse_map = backwards_map[inverse_map]
    
    #combine the unique stacked indices into unique stacked data
    data2stack = []
    for idx, data in enumerate(alldata):
        data2stack.append(data[unique_stacked_indices[:,idx]])
    unique_stacked_data = numpy.hstack(data2stack).flatten()
    unique_stacked_data.shape = (-1)
    all_data = unique_stacked_data.tostring()

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

    indexFormat = GeomVertexArrayFormat()
    indexFormat.addColumn(InternalName.make("index"), 1, Geom.NTUint32, Geom.CIndex)
    indexFormat = GeomVertexArrayFormat.registerFormat(indexFormat)
    indexArray = GeomVertexArrayData(indexFormat, GeomEnums.UHStream)
    indexHandle = indexArray.modifyHandle()
    indexData = inverse_map.tostring()
    indexHandle.setData(indexData)
    return vdata, indexArray

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

    vdata, indexdata = getVertexData(triset.vertex, triset.vertex_index,
                          triset.normal, triset.normal_index,
                          triset.texcoordset, triset.texcoord_indexset,
                          triset.textangentset, triset.textangent_indexset,
                          triset.texbinormalset, triset.texbinormal_indexset)

    gprim = GeomTriangles(Geom.UHStatic)
    gprim.setIndexType(Geom.NTUint32)
    gprim.setVertices(indexdata)
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
            tex.setMinfilter(Texture.FTLinearMipmapLinear)
            
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

def getTexture(color=None, alpha=None, texture_cache=None, diffuseinit=None):

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
            if diffuseinit is None:
                diffuseinit = (0,0,0,1)
            else:
                diffuseinit = tuple(int(v*255) for v in diffuseinit)
            newim = Image.new('RGBA', im.size, diffuseinit)
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

# luminance function, based on the ISO/CIE color standards
# see ITU-R Recommendation BT.709-4
def luminance(c):
    return c[0] * 0.212671 + c[1] * 0.715160 + c[2] * 0.072169

def getStateFromMaterial(prim_material, texture_cache, col_inst=None):
    state = RenderState.makeEmpty()
    
    mat = Material()
    texattr = TextureAttrib.makeAllOff()
    
    hasDiffuse = False
    if prim_material and prim_material.effect:
        
        diffuse = getattr(prim_material.effect, 'diffuse', None)
        transparent = getattr(prim_material.effect, 'transparent', None)
        if isinstance(diffuse, collada.material.Map) and isinstance(transparent, collada.material.Map):
            if diffuse.sampler.surface.image == transparent.sampler.surface.image:
                #some exporters put the same map in the diffuse channel
                # and the transparent channel when they don't really mean to
                transparent = None
        
        if isinstance(diffuse, collada.material.Map) or isinstance(transparent, collada.material.Map):
            diffuseMap = None
            transparentMap = None
            diffuseInitColor = None
            if isinstance(diffuse, collada.material.Map):
                diffuseMap = diffuse
            else:
                diffuseInitColor = v4fromtuple(diffuse)
            if isinstance(transparent, collada.material.Map):
                transparentMap = transparent
            if diffuseMap == transparentMap:
                transparentMap = None
                
            diffuseTexture = getTexture(color=diffuseMap, alpha=transparentMap, texture_cache=texture_cache, diffuseinit=diffuseInitColor)
            texattr = addTextureStage('tsDiff', TextureStage.MModulate, texattr, diffuseTexture)
            hasDiffuse = True

        if type(diffuse) is tuple:
            mat.setDiffuse(v4fromtuple(diffuse))
        
        # hack to look for sketchup version < 8 where transparency was exported flipped
        # also ColladaMaya v2.03b had this same issue
        flip_alpha = False
        if col_inst and col_inst.assetInfo:
            for contributor in col_inst.assetInfo.contributors:
                tool_name = contributor.authoring_tool
                split = tool_name.split()
                if len(split) == 3 and \
                      split[0].strip().lower() == 'google' and \
                      split[1].strip().lower() == 'sketchup':
                    version = split[2].strip().split('.')
                    try:
                        major_version = int(version[0])
                        if major_version < 8:
                            flip_alpha = True
                    except (ValueError, TypeError):
                        continue
                    
                try:
                    collada_maya_idx = split.index('ColladaMaya')
                    if split[collada_maya_idx + 1] == 'v2.03b':
                        flip_alpha = True
                except (ValueError, IndexError):
                    continue
        
        if type(transparent) is tuple:
            trR, trG, trB = transparent[0], transparent[1], transparent[2]
            trA = transparent[3] if len(transparent) > 3 else 1.0
        else:
            trR, trG, trB = 1.0, 1.0, 1.0
            trA = 1.0
        
        transparency = getattr(prim_material.effect, 'transparency', 1.0)
        if transparency is None:
            transparency = 1.0
        a_one = prim_material.effect.opaque_mode == collada.material.OPAQUE_MODE.A_ONE
        if a_one:
            alphaR = alphaG = alphaB = alphaA = transparency * trA
        else:
            alphaR = transparency * trR
            alphaG = transparency * trG
            alphaB = transparency * trB
            alphaA = luminance([trR, trG, trB])
            flip_alpha = not flip_alpha
        
        if flip_alpha:
            alphaR = 1.0 - alphaR
            alphaG = 1.0 - alphaG
            alphaB = 1.0 - alphaB
            alphaA = 1.0 - alphaA
        
        if alphaA < 1.0:
            state = state.addAttrib(ColorScaleAttrib.make(VBase4(alphaR, alphaG, alphaB, alphaA)))
        
        emission = getattr(prim_material.effect, 'emission', None)
        if isinstance(emission, collada.material.Map):
            emissionTexture = getTexture(alpha=emission, texture_cache=texture_cache)
            texattr = addTextureStage('tsEmiss', TextureStage.MGlow, texattr, emissionTexture)
        elif type(emission) is tuple:
            mat.setEmission(v4fromtuple(emission))
        
        ambient = getattr(prim_material.effect, 'ambient', None)
        if type(ambient) is tuple:
            mat.setAmbient(v4fromtuple(ambient))
        
        specular = getattr(prim_material.effect, 'specular', None)
        if isinstance(specular, collada.material.Map):
            specularTexture = getTexture(color=specular, texture_cache=texture_cache)
            texattr = addTextureStage('tsSpec', TextureStage.MGloss, texattr, specularTexture)
            mat.setSpecular(VBase4(0.1, 0.1, 0.1, 1.0))
        elif type(specular) is tuple:
            mat.setSpecular(v4fromtuple(specular))

        shininess = getattr(prim_material.effect, 'shininess', None)
        #this sets a sane value for blinn shading
        if shininess <= 1.0:
            if shininess < 0.01:
                shininess = 1.0
            shininess = shininess * 128.0
        mat.setShininess(shininess)

        bumpmap = getattr(prim_material.effect, 'bumpmap', None)
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

def centerAndScale(nodePath):
    wrapNode = None
    
    if not nodePath.isEmpty():
        
        parentNP = nodePath.getParent()
        nodePath.detachNode()
        nodePath.setName('wrapper-centering-collada')
        wrapNode = parentNP.attachNewNode('collada')
        nodePath.reparentTo(wrapNode)
        
        boundingSphere = nodePath.getBounds()
        if not boundingSphere.isEmpty():
            
            minPt, maxPt = nodePath.getTightBounds()
            scalex = math.fabs(maxPt.getX() - minPt.getX())
            scaley = math.fabs(maxPt.getY() - minPt.getY())
            scalez = math.fabs(maxPt.getZ() - minPt.getZ())
            
            scale = 1000.0 / max(scalex, scaley, scalez)
            
            nodePath.setScale(scale, scale, scale)

            minPt, maxPt = nodePath.getTightBounds()
            centerPt = (maxPt + minPt) * 0.5
            nodePath.setPos(-1 * centerPt.getX(),
                            -1 * centerPt.getY(),
                            -1 * centerPt.getZ())

            nodePath.setHpr(0,0,0)
    
    if wrapNode is not None:
        return wrapNode
    return nodePath

def ensureCameraAt(nodePath, cam):
    wrapNode = centerAndScale(nodePath)
    
    cam.setPos(1500, -1500, 1)
    cam.lookAt(0.0, 0.0, 0.0)
    
    return wrapNode

def getGeomFromPrim(prim, matstate):
    if type(prim) is collada.triangleset.TriangleSet:
        (vdata, gprim) = getPrimAndDataFromTri(prim, matstate)
    elif type(prim) is collada.polylist.Polylist or type(prim) is collada.polygons.Polygons:
        triset = prim.triangleset()
        (vdata, gprim) = getPrimAndDataFromTri(triset, matstate)
    elif type(prim) is collada.lineset.LineSet:
        vdata, indexdata = getVertexData(prim.vertex, prim.vertex_index)
        gprim = GeomLines(Geom.UHStatic)
        gprim.setIndexType(Geom.NTUint32)
        gprim.setVertices(indexdata)
        gprim.closePrimitive()
    else:
        raise Exception("Error: Unsupported primitive type. Exiting.")

    pgeom = Geom(vdata)
    pgeom.addPrimitive(gprim)
    return pgeom

def recurseScene(curnode, scene_members, data_cache, M, texture_cache, col_inst=None):
    M = numpy.dot(M, curnode.matrix)
    for node in curnode.children:
        if isinstance(node, collada.scene.Node):
            recurseScene(node, scene_members, data_cache, M, texture_cache, col_inst=col_inst)
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
                            matstate = getStateFromMaterial(mat.target, texture_cache, col_inst)
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
            recurseScene(node, scene_members, data_cache, m, texture_cache, col_inst=col)
    
    return scene_members

def setupPandaApp(mesh):
    scene_members = getSceneMembers(mesh)
    
    p3dApp = ShowBase()
    nodePath = getBaseNodePath(render)
    
    rotateNode = GeomNode("rotater")
    rotatePath = nodePath.attachNewNode(rotateNode)
    matrix = numpy.identity(4)
    if mesh.assetInfo.upaxis == collada.asset.UP_AXIS.X_UP:
        r = collada.scene.RotateTransform(0,1,0,90)
        matrix = r.matrix
    elif mesh.assetInfo.upaxis == collada.asset.UP_AXIS.Y_UP:
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
