import tempfile
import tarfile
import shutil
import os
import subprocess

import numpy
import collada
from direct.showbase.ShowBase import ShowBase
from panda3d.core import GeomNode, Mat4, TransparencyAttrib, PNMImage, StringStream, Texture
from panda3d.core import MultiplexStream, Notify, Filename

from meshtool.filters.base_filters import PrintFilter, FileArgument, FilterException
from meshtool.filters.panda_filters import pdae_utils
from meshtool.filters.panda_filters.pdae_utils import panda as pdae_panda
from meshtool.filters.panda_filters import pandacontrols as controls
from meshtool.filters.panda_filters.pandacore import getSceneMembers, ensureCameraAt, attachLights, getScreenshot
from meshtool.util import which

PM_CHUNK_SIZE = 1 * 1024 * 1024 # 1 megabyte

def getNumTriangles(geomPath):
    geom = geomPath.node().getGeom(0)
    prim = geom.getPrimitive(0)
    indexdata = prim.getVertices()
    return indexdata.getNumRows()

def takeScreenshot(tempdir, base, geomPath, texim, angle):
    numTriangles = getNumTriangles(geomPath)
    ss_string = '%d_%d_%d_%d.png' % (numTriangles, texim.getXSize(), texim.getYSize(), angle)
    pilimage = getScreenshot(base)
    pilimage.save(os.path.join(tempdir, ss_string))

def getPmPerceptualError(mesh, pm_filebuf, mipmap_tarfilebuf):
    perceptualdiff = which('perceptualdiff')
    if perceptualdiff is None:
        raise Exception("perceptualdiff exectuable not found on path")
    
    pm_chunks = []
    
    if pm_filebuf is not None:
        data = pm_filebuf.read(PM_CHUNK_SIZE)
        refinements_read = 0
        num_refinements = None
        while len(data) > 0:
            (refinements_read, num_refinements, pm_refinements, data_left) = pdae_utils.readPDAEPartial(data, refinements_read, num_refinements)
            pm_chunks.append(pm_refinements)
            data = data_left + pm_filebuf.read(PM_CHUNK_SIZE)
    
    tar = tarfile.TarFile(fileobj=mipmap_tarfilebuf)
    texsizes = []
    largest_tarinfo = (0, None)
    for tarinfo in tar:
        tarinfo.xsize = int(tarinfo.name.split('x')[0])
        if tarinfo.xsize > largest_tarinfo[0]:
            largest_tarinfo = (tarinfo.xsize, tarinfo)
        if tarinfo.xsize >= 128:
            texsizes.append(tarinfo)
    if len(texsizes) == 0:
        texsizes.append(largest_tarinfo[1])
    
    texsizes = sorted(texsizes, key=lambda t: t.xsize)
    texims = []
    first_image_data = None
    for tarinfo in texsizes:
        f = tar.extractfile(tarinfo)
        texdata = f.read()
        if first_image_data is None:
            first_image_data = texdata
        
        texpnm = PNMImage()
        texpnm.read(StringStream(texdata), 'something.jpg')
        newtex = Texture()
        newtex.load(texpnm)
        texims.append(newtex)
    
    mesh.images[0].setData(first_image_data)
    
    scene_members = getSceneMembers(mesh)
    
    # turn off panda3d printing to stdout
    nout = MultiplexStream()
    Notify.ptr().setOstreamPtr(nout, 0)
    nout.addFile(Filename(os.devnull))
    
    base = ShowBase()
    
    rotateNode = GeomNode("rotater")
    rotatePath = base.render.attachNewNode(rotateNode)
    matrix = numpy.identity(4)
    if mesh.assetInfo.upaxis == collada.asset.UP_AXIS.X_UP:
        r = collada.scene.RotateTransform(0,1,0,90)
        matrix = r.matrix
    elif mesh.assetInfo.upaxis == collada.asset.UP_AXIS.Y_UP:
        r = collada.scene.RotateTransform(1,0,0,90)
        matrix = r.matrix
    rotatePath.setMat(Mat4(*matrix.T.flatten().tolist()))
    geom, renderstate, mat4 = scene_members[0]
    node = GeomNode("primitive")
    node.addGeom(geom)
    if renderstate is not None:
        node.setGeomState(0, renderstate)
    geomPath = rotatePath.attachNewNode(node)
    geomPath.setMat(mat4)
        
    wrappedNode = ensureCameraAt(geomPath, base.camera)
    base.disableMouse()
    attachLights(base.render)
    base.render.setShaderAuto()
    base.render.setTransparency(TransparencyAttrib.MNone)
    base.render.setColorScaleOff(9999)
    
    controls.KeyboardMovement()
    controls.MouseDrag(wrappedNode)
    controls.MouseScaleZoom(wrappedNode)
    controls.ButtonUtils(wrappedNode)
    controls.MouseCamera()
    
    error_data = []
    
    try:
        tempdir = tempfile.mkdtemp(prefix='meshtool-print-pm-perceptual-error')
        
        triangleCounts = []
        
        hprs = [(0, 0, 0),
                (0, 90, 0),
                (0, 180, 0),
                (0, 270, 0),
                (90, 0, 0),
                (-90, 0, 0)]
        
        for texim in texims:
            np = base.render.find("**/rotater/collada")
            np.setTextureOff(1)
            np.setTexture(texim, 1)
            for angle, hpr in enumerate(hprs):
                wrappedNode.setHpr(*hpr)
                takeScreenshot(tempdir, base, geomPath, texim, angle)
        triangleCounts.append(getNumTriangles(geomPath))
        
        for pm_chunk in pm_chunks:
            pdae_panda.add_refinements(geomPath, pm_chunk)
            
            for texim in texims:
                np = base.render.find("**/rotater/collada")
                np.setTextureOff(1)
                np.setTexture(texim, 1)
                for angle, hpr in enumerate(hprs):
                    wrappedNode.setHpr(*hpr)
                    takeScreenshot(tempdir, base, geomPath, texim, angle)
            triangleCounts.append(getNumTriangles(geomPath))
        
        full_tris = triangleCounts[-1]
        full_tex = texims[-1]
        
        for numtris in triangleCounts:
            for texim in texims:
                pixel_diff = 0
                for angle, hpr in enumerate(hprs):
                    curFile = '%d_%d_%d_%d.png' % (numtris, texim.getXSize(), texim.getYSize(), angle)
                    curFile = os.path.join(tempdir, curFile)
                    
                    fullFile = '%d_%d_%d_%d.png' % (full_tris, full_tex.getXSize(), full_tex.getYSize(), angle)
                    fullFile = os.path.join(tempdir, fullFile)
                    
                    try:
                        output = subprocess.check_output([perceptualdiff, '-threshold', '1', fullFile, curFile])
                    except subprocess.CalledProcessError, ex:
                        output = ex.output
                    
                    output = output.strip()
                    if len(output) > 0:
                        pixel_diff = max(pixel_diff, int(output.split('\n')[1].split()[0]))
                    
                error_data.append({'triangles': numtris,
                                   'width': texim.getXSize(),
                                   'height': texim.getYSize(),
                                   'pixel_error': pixel_diff})
    
    finally:
        shutil.rmtree(tempdir, ignore_errors=True)
        
    return error_data

def printPmPerceptualError(mesh, pm_filebuf, mipmap_tarfilebuf):
    error_data = getPmPerceptualError(mesh, pm_filebuf, mipmap_tarfilebuf)
    for level in error_data:
        print '%(triangles)d tris, %(width)dx%(height)d: %(pixel_error)d pixels' % level

def FilterGenerator():
    class PrintPmPerceptualError(PrintFilter):
        def __init__(self):
            super(PrintPmPerceptualError, self).__init__('print_pm_perceptual_error',
                    'Prints perceptual error at different levels of a progressive mesh compared to full resolution')
            
            self.arguments.append(FileArgument("pm_file", "Path of the progressive mesh file. Specify NONE if no pm file."))
            self.arguments.append(FileArgument("mipmap_tar_file", "Path of the tar file with mipmap levels in it"))
            
        def apply(self, mesh, pm_filename, mipmap_tarfilename):
            try:
                pm_filebuf = open(pm_filename, 'r') if pm_filename != 'NONE' else None
            except IOError, ex:
                raise FilterException("Error opening pm file: %s" % str(ex))
            
            try:
                mipmap_tarfilebuf = open(mipmap_tarfilename, 'rb')
            except IOError, ex:
                raise FilterException("Error opening mipmap tar: %s" % str(ex))
            
            perceptualdiff = which('perceptualdiff')
            if perceptualdiff is None:
                raise FilterException("perceptualdiff exectuable not found on path")
            
            printPmPerceptualError(mesh, pm_filebuf, mipmap_tarfilebuf)
            return mesh
    
    return PrintPmPerceptualError()

from meshtool.filters import factory
factory.register(FilterGenerator().name, FilterGenerator)
