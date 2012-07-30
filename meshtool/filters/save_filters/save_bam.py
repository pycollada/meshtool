from meshtool.filters.base_filters import SaveFilter, FilterException
from meshtool.filters.panda_filters import pandacore
from panda3d.core import GeomNode, NodePath, Mat4
import collada
import os
import numpy
import tempfile

def getBam(mesh, filename):
    scene_members = pandacore.getSceneMembers(mesh)
    
    rotateNode = GeomNode("rotater")
    rotatePath = NodePath(rotateNode)
    matrix = numpy.identity(4)
    if mesh.assetInfo.upaxis == collada.asset.UP_AXIS.X_UP:
        r = collada.scene.RotateTransform(0,1,0,90)
        matrix = r.matrix
    elif mesh.assetInfo.upaxis == collada.asset.UP_AXIS.Y_UP:
        r = collada.scene.RotateTransform(1,0,0,90)
        matrix = r.matrix
    rotatePath.setMat(Mat4(*matrix.T.flatten().tolist()))
    
    for geom, renderstate, mat4 in scene_members:
        node = GeomNode("primitive")
        node.addGeom(geom)
        if renderstate is not None:
            node.setGeomState(0, renderstate)
        geomPath = rotatePath.attachNewNode(node)
        geomPath.setMat(mat4)

    rotatePath.flattenStrong()
    wrappedNode = pandacore.centerAndScale(rotatePath)
    
    model_name = filename.replace('/', '_')
    wrappedNode.setName(model_name)
    
    bam_temp = tempfile.mktemp(suffix = model_name + '.bam')
    wrappedNode.writeBamFile(bam_temp)
    
    bam_f = open(bam_temp, 'rb')
    bam_data = bam_f.read()
    bam_f.close()
    
    os.remove(bam_temp)
    
    return bam_data

def FilterGenerator():
    class BamSaveFilter(SaveFilter):
        def __init__(self):
            super(BamSaveFilter, self).__init__('save_bam', 'Saves to Panda3D BAM file format')
        def apply(self, mesh, filename):
            if os.path.exists(filename):
                raise FilterException("specified filename already exists")
            bam_data = getBam(mesh, os.path.basename(filename))
            f = open(filename, 'wb')
            f.write(bam_data)
            f.close()
            return mesh
    return BamSaveFilter()
from meshtool.filters import factory
factory.register(FilterGenerator().name, FilterGenerator)