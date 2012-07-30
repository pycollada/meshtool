from meshtool.filters.base_filters import VisualizationFilter

from pandacore import getSceneMembers
import numpy
from direct.showbase.ShowBase import ShowBase
from panda3d.core import GeomNode, TransparencyAttrib, OrthographicLens
from panda3d.core import AmbientLight, DirectionalLight, PointLight, Spotlight
from panda3d.core import Vec4, Vec3, Point3, Mat4
from panda3d.core import loadPrcFileData
import collada
import math
from pandacontrols import KeyboardMovement, MouseDrag, MouseScaleZoom

def runViewer(mesh):
    scene_members = getSceneMembers(mesh)
    
    loadPrcFileData('', 'win-size 300 300')
    base = ShowBase()
    globNode = GeomNode("collada")
    nodePath = base.render.attachNewNode(globNode)
    
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
    
    basecollada = GeomNode("basecollada")
    basecolladaNP = rotatePath.attachNewNode(basecollada)
    for geom, renderstate, mat4 in scene_members:
        node = GeomNode("primitive")
        node.addGeom(geom)
        if renderstate is not None:
            node.setGeomState(0, renderstate)
        geomPath = basecolladaNP.attachNewNode(node)
        geomPath.setMat(mat4)

    for boundlight in mesh.scene.objects('light'):
        
        if len(boundlight.color) == 3:
            color = (boundlight.color[0], boundlight.color[1], boundlight.color[2], 1)
        else:
            color = boundlight.color
        
        if isinstance(boundlight, collada.light.BoundDirectionalLight):
            dl = DirectionalLight('dirLight')
            dl.setColor(Vec4(color[0], color[1], color[2], color[3]))
            lightNP = rotatePath.attachNewNode(dl)
            lightNP.lookAt(Point3(boundlight.direction[0],boundlight.direction[1],boundlight.direction[2]))
        elif isinstance(boundlight, collada.light.BoundAmbientLight):
            ambientLight = AmbientLight('ambientLight')
            ambientLight.setColor(Vec4(color[0], color[1], color[2], color[3]))
            lightNP = rotatePath.attachNewNode(ambientLight)
        elif isinstance(boundlight, collada.light.BoundPointLight):
            pointLight = PointLight('pointLight')
            pointLight.setColor(Vec4(color[0], color[1], color[2], color[3]))
            pointLight.setAttenuation(Vec3(boundlight.constant_att, boundlight.linear_att, boundlight.quad_att))
            lightNP = rotatePath.attachNewNode(pointLight)
            lightNP.setPos(Vec3(boundlight.position[0], boundlight.position[1], boundlight.position[2]))
        elif isinstance(boundlight, collada.light.BoundSpotLight):
            spotLight = Spotlight('spotLight')
            spotLight.setColor(Vec4(color[0], color[1], color[2], color[3]))
            spotLight.setAttenuation(Vec3(boundlight.constant_att, boundlight.linear_att, boundlight.quad_att))
            spotLight.setExponent(boundlight.falloff_exp)
            lightNP = rotatePath.attachNewNode(spotLight)
            lightNP.setPos(Vec3(boundlight.position[0], boundlight.position[1], boundlight.position[2]))
            lightNP.lookAt(Point3(boundlight.direction[0], boundlight.direction[1], boundlight.direction[2]),
                               Vec3(boundlight.up[0], boundlight.up[1], boundlight.up[2]))
        else:
            print 'Unknown light type', boundlight
            continue
            
        base.render.setLight(lightNP)

    for boundcam in mesh.scene.objects('camera'):
        if isinstance(boundcam, collada.camera.BoundPerspectiveCamera):
            base.camera.reparentTo(rotatePath)
            base.camLens.setNear(boundcam.znear)
            base.camLens.setFar(boundcam.zfar)
            
            if boundcam.xfov is not None and boundcam.yfov is not None:
                #xfov + yfov
                base.camLens.setFov(boundcam.xfov, boundcam.yfov)
            elif boundcam.xfov is not None and boundcam.aspect_ratio is not None:
                #xfov + aspect_ratio
                base.camLens.setFov(boundcam.xfov)
                base.camLens.setAspectRatio(boundcam.aspect_ratio)
            elif boundcam.yfov is not None and boundcam.aspect_ratio is not None:
                #yfov + aspect_ratio
                #aspect_ratio = tan(0.5*xfov) / tan(0.5*yfov)
                xfov = math.degrees(2.0 * math.atan(boundcam.aspect_ratio * math.tan(math.radians(0.5 * boundcam.yfov))))
                base.camLens.setFov(xfov, boundcam.yfov)
            elif boundcam.yfov is not None:
                #yfov only
                #aspect_ratio = tan(0.5*xfov) / tan(0.5*yfov)
                xfov = math.degrees(2.0 * math.atan(base.camLens.getAspectRatio() * math.tan(math.radians(0.5 * boundcam.yfov))))
                base.camLens.setFov(xfov, boundcam.yfov)
            elif boundcam.xfov is not None:
                base.camLens.setFov(boundcam.xfov)
            
            base.camera.setPos(Vec3(boundcam.position[0], boundcam.position[1], boundcam.position[2]))
            base.camera.lookAt(Point3(boundcam.direction[0], boundcam.direction[1], boundcam.direction[2]),
                               Vec3(boundcam.up[0], boundcam.up[1], boundcam.up[2]))
        elif isinstance(boundcam, collada.camera.BoundOrthographicCamera):
            
            lens = OrthographicLens()
            base.cam.node().setLens(lens)
            base.camLens = lens
            base.camera.reparentTo(rotatePath)
            base.camLens.setNear(boundcam.znear)
            base.camLens.setFar(boundcam.zfar)
            
            if boundcam.xmag is not None and boundcam.ymag is not None:
                #xmag + ymag
                base.camLens.setFilmSize(boundcam.xmag, boundcam.ymag)
            elif boundcam.xmag is not None and boundcam.aspect_ratio is not None:
                #xmag + aspect_ratio
                base.camLens.setFilmSize(boundcam.xmag)
                base.camLens.setAspectRatio(boundcam.aspect_ratio)
            elif boundcam.ymag is not None and boundcam.aspect_ratio is not None:
                #ymag + aspect_ratio
                xmag = boundcam.aspect_ratio * boundcam.ymag
                base.camLens.setFilmSize(xmag, boundcam.ymag)
            elif boundcam.ymag is not None:
                #ymag only
                xmag = base.camLens.getAspectRatio() * boundcam.ymag
                base.camLens.setFilmSize(xmag, boundcam.ymag)
            elif boundcam.xmag is not None:
                base.camLens.setFilmSize(boundcam.xmag)
            
            base.camera.setPos(Vec3(boundcam.position[0], boundcam.position[1], boundcam.position[2]))
            base.camera.lookAt(Point3(boundcam.direction[0], boundcam.direction[1], boundcam.direction[2]),
                               Vec3(boundcam.up[0], boundcam.up[1], boundcam.up[2]))
            
        else:
            print 'Unknown camera type', boundcam
            continue

    base.disableMouse()
    base.render.setShaderAuto()
    base.render.setTransparency(TransparencyAttrib.MDual, 1)
    
    KeyboardMovement()
    MouseDrag(basecolladaNP)
    MouseScaleZoom(basecolladaNP)
    
    base.run()

def FilterGenerator():
    class ColladaViewer(VisualizationFilter):
        def __init__(self):
            super(ColladaViewer, self).__init__('collada_viewer', 'Uses panda3d to bring up a viewer with lights and camera from the collada file')
        def apply(self, mesh):
            runViewer(mesh)
            return mesh

    return ColladaViewer()
from meshtool.filters import factory
factory.register(FilterGenerator().name, FilterGenerator)