from meshtool.args import *
from meshtool.filters.base_filters import *

from pandacore import getSceneMembers, ensureCameraAt, attachLights
from pandacontrols import KeyboardMovement, MouseDrag, MouseScaleZoom
from direct.gui.DirectGui import DirectButton
from direct.showbase.ShowBase import ShowBase
from panda3d.core import GeomNode, TransparencyAttrib, GeomVertexWriter, GeomVertexReader
import os
import sys

uiArgs = { 'rolloverSound':None,
           'clickSound':None
        }

def readPm(pm_filebuf, geomPath):
    
    geom = geomPath.node().modifyGeom(0)
    vertdata = geom.modifyVertexData()
    prim = geom.modifyPrimitive(0)
    indexdata = prim.modifyVertices()
    
    indexwriter = GeomVertexWriter(indexdata)
    indexwriter.setColumn(0)
    nextTriangleIndex = indexdata.getNumRows()
    
    vertwriter = GeomVertexWriter(vertdata, 'vertex')
    vertwriter.setRow(vertdata.getNumRows())
    normalwriter = GeomVertexWriter(vertdata, 'normal')
    normalwriter.setRow(vertdata.getNumRows())
    uvwriter = GeomVertexWriter(vertdata, 'texcoord')
    uvwriter.setRow(vertdata.getNumRows())
    
    #gotV = False
    for line in pm_filebuf:
        vals = line.strip().split()
        op = vals.pop(0)
        if op == 't':
            v1, v2, v3 = map(int, vals)
            indexwriter.setRow(nextTriangleIndex)
            nextTriangleIndex += 3
            indexwriter.addData1i(v1)
            indexwriter.addData1i(v2)
            indexwriter.addData1i(v3)
        elif op == 'u':
            tindex, vindex = map(int, vals)
            indexwriter.setRow(tindex)
            indexwriter.setData1i(vindex)
        elif op == 'v':
            #if gotV:
            #    return
            #gotV = True
            vx, vy, vz, nx, ny, nz, s, t = map(float, vals)
            vertwriter.addData3f(vx,vy,vz)
            normalwriter.addData3f(nx,ny,nz)
            uvwriter.addData2f(s,t)
        else:
            assert(False)

def runPmViewer(mesh, pm_filebuf):

    scene_members = getSceneMembers(mesh)
    
    base = ShowBase()
    
    if len(scene_members) > 1:
        print 'There is more than one geometry in the scene, so I think this is not a progressive base mesh.'
        sys.exit(1)
    
    geom, renderstate, mat4 = scene_members[0]
    node = GeomNode("primitive")
    node.addGeom(geom)
    if renderstate is not None:
        node.setGeomState(0, renderstate)
    geomPath = render.attachNewNode(node)
    geomPath.setMat(mat4)
        
    ensureCameraAt(geomPath, base.camera)
    base.disableMouse()
    attachLights(render)
    render.setShaderAuto()
    render.setTransparency(TransparencyAttrib.MDual, 1)

    base.render.analyze()
    KeyboardMovement()
    MouseDrag(geomPath)
    MouseScaleZoom(geomPath)
    
    DirectButton(text="Load Progressive", scale=.05, pos=(-1.11, 0, 0.94), command=readPm, extraArgs=[pm_filebuf, geomPath], **uiArgs)
    
    base.run()

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
            runPmViewer(mesh, pm_filebuf)
            return mesh

    return PmViewer()
from meshtool.filters import factory
factory.register(FilterGenerator().name, FilterGenerator)