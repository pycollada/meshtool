from meshtool.args import *
from meshtool.filters.base_filters import *

from pandacore import getSceneMembers, ensureCameraAt, attachLights
from pandacontrols import KeyboardMovement, MouseDrag, MouseScaleZoom
from direct.gui.DirectGui import DirectButton, DirectSlider, OnscreenText
from direct.showbase.ShowBase import ShowBase
from panda3d.core import GeomNode, TransparencyAttrib, GeomVertexWriter
from panda3d.core import GeomVertexReader, GeomVertexRewriter, TextNode
from panda3d.core import Mat4
import os
import sys
import numpy
import collada

uiArgs = { 'rolloverSound':None,
           'clickSound':None
        }

class PM_OP:
    INDEX_UPDATE = 1
    TRIANGLE_ADDITION = 2
    VERTEX_ADDITION = 3
    
def readPm(pm_filebuf):
    pdae_line = pm_filebuf.readline().strip()
    if pdae_line != 'PDAE':
        print >> sys.stderr, 'Progressive mesh file given does not have a valid PDAE header'
        sys.exit(1)
    
    num_refinements = int(pm_filebuf.readline().strip())

    pm_refinements = []
    for refinement_index in range(num_refinements):
        num_operations = int(pm_filebuf.readline().strip())
        refinement_ops = []
        for operation_index in range(num_operations):
            vals = pm_filebuf.readline().strip().split()
            op = vals.pop(0)
            if op == 't':
                v1, v2, v3 = map(int, vals)
                refinement_ops.append((PM_OP.TRIANGLE_ADDITION, v1, v2, v3))
            elif op == 'u':
                tindex, vindex = map(int, vals)
                refinement_ops.append((PM_OP.INDEX_UPDATE, tindex, vindex))
            elif op == 'v':
                vx, vy, vz, nx, ny, nz, s, t = map(float, vals)
                refinement_ops.append((PM_OP.VERTEX_ADDITION, vx, vy, vz, nx, ny, nz, s, t))
        pm_refinements.append(refinement_ops)

    return pm_refinements

class PandaPmViewer:
    
    def __init__(self, mesh, pm_filebuf):

        scene_members = getSceneMembers(mesh)
        
        base = ShowBase()
        
        if len(scene_members) > 1:
            print 'There is more than one geometry in the scene, so I think this is not a progressive base mesh.'
            sys.exit(1)
        
        rotateNode = GeomNode("rotater")
        rotatePath = render.attachNewNode(rotateNode)
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
        self.geomPath = rotatePath.attachNewNode(node)
        self.geomPath.setMat(mat4)
            
        ensureCameraAt(self.geomPath, base.camera)
        base.disableMouse()
        attachLights(render)
        render.setShaderAuto()
        render.setTransparency(TransparencyAttrib.MDual, 1)
    
        base.render.analyze()
        KeyboardMovement()
        MouseDrag(self.geomPath)
        MouseScaleZoom(self.geomPath)
        
        print 'Loading pm into memory... ',
        sys.stdout.flush()
        self.pm_refinements = readPm(pm_filebuf)
        self.pm_index = 0
        print 'Done'

        self.slider = DirectSlider(range=(0,len(self.pm_refinements)),
                                   value=0, pageSize=len(self.pm_refinements)/20,
                                   command=self.sliderMoved, pos=(0, 0, -.9), scale=1)
        for key, val in uiArgs.iteritems():
            self.slider.thumb[key] = val
        
        self.triText = OnscreenText(text="", pos=(-1,0.85), scale = 0.15,
                                    fg=(1, 0.5, 0.5, 1), align=TextNode.ALeft, mayChange=1)
        
        base.run()
        
    def sliderMoved(self):
        sliderVal = int(self.slider['value'])
        if self.pm_index != sliderVal:
            self.movePmTo(sliderVal)
        self.triText.setText('Triangles: ' + str(self.geomPath.node().getGeom(0).getPrimitive(0).getNumFaces()))

    def movePmTo(self, dest_index):
        geom = self.geomPath.node().modifyGeom(0)
        vertdata = geom.modifyVertexData()
        prim = geom.modifyPrimitive(0)
        indexdata = prim.modifyVertices()
        
        indexrewriter = GeomVertexRewriter(indexdata)
        indexrewriter.setColumn(0)
        nextTriangleIndex = indexdata.getNumRows()
        
        vertwriter = GeomVertexWriter(vertdata, 'vertex')
        numverts = vertdata.getNumRows()
        vertwriter.setRow(numverts)
        normalwriter = GeomVertexWriter(vertdata, 'normal')
        normalwriter.setRow(numverts)
        uvwriter = GeomVertexWriter(vertdata, 'texcoord')
        uvwriter.setRow(numverts)
        
        while self.pm_index < dest_index:
            for op_index in range(len(self.pm_refinements[self.pm_index])):
                vals = self.pm_refinements[self.pm_index][op_index]
                op = vals[0]
                if op == PM_OP.TRIANGLE_ADDITION:
                    indexrewriter.setRow(nextTriangleIndex)
                    nextTriangleIndex += 3
                    indexrewriter.addData1i(vals[1])
                    indexrewriter.addData1i(vals[2])
                    indexrewriter.addData1i(vals[3])
                elif op == PM_OP.INDEX_UPDATE:
                    #TODO: ugly workaround for p3d 1.7 bug, change to below for 1.8
                    indexreader = GeomVertexReader(indexdata)
                    indexreader.setColumn(0)
                    indexreader.setRow(vals[1])
                    oldval = indexreader.getData1i()
                    del indexreader
                    
                    #indexrewriter.setRow(vals[1])
                    #oldval = indexrewriter.getData1i()
                    
                    indexrewriter.setRow(vals[1])
                    indexrewriter.setData1i(vals[2])
                    self.pm_refinements[self.pm_index][op_index] = (op, vals[1], oldval)
                elif op == PM_OP.VERTEX_ADDITION:
                    numverts += 1
                    vertwriter.addData3f(vals[1], vals[2], vals[3])
                    normalwriter.addData3f(vals[4], vals[5], vals[6])
                    uvwriter.addData2f(vals[7], vals[8])
                
            self.pm_index += 1

        while self.pm_index > dest_index:
            self.pm_index -= 1
            for op_index in range(len(self.pm_refinements[self.pm_index])):
                vals = self.pm_refinements[self.pm_index][op_index]
                op = vals[0]
                if op == PM_OP.TRIANGLE_ADDITION:
                    nextTriangleIndex -= 3
                elif op == PM_OP.INDEX_UPDATE:
                    #TODO: ugly workaround for p3d 1.7 bug, change to below for 1.8
                    indexreader = GeomVertexReader(indexdata)
                    indexreader.setColumn(0)
                    indexreader.setRow(vals[1])
                    oldval = indexreader.getData1i()
                    del indexreader
                    
                    #indexrewriter.setRow(vals[1])
                    #oldval = indexrewriter.getData1i()
                    
                    indexrewriter.setRow(vals[1])
                    indexrewriter.setData1i(vals[2])
                    self.pm_refinements[self.pm_index][op_index] = (op, vals[1], oldval)
                elif op == PM_OP.VERTEX_ADDITION:
                    numverts -= 1

        if nextTriangleIndex < indexdata.getNumRows():
            indexdata.setNumRows(nextTriangleIndex)
        if numverts < vertdata.getNumRows():
            vertdata.setNumRows(numverts)


def FilterGenerator():
    class PmViewer(OpFilter):
        def __init__(self):
            super(PmViewer, self).__init__('pm_viewer', 'Uses panda3d to bring up a viewer of a base mesh and progressive stream')
            self.arguments.append(FileArgument("pm_file", "Path of the progressive mesh file"))
        def apply(self, mesh, pm_filename):
            try:
                pm_filebuf = open(pm_filename, 'r')
            except IOError, ex:
                print "Error opening pm file:", str(ex)
                sys.exit(1)
            pmview = PandaPmViewer(mesh, pm_filebuf)
            return mesh

    return PmViewer()
from meshtool.filters import factory
factory.register(FilterGenerator().name, FilterGenerator)