from meshtool.args import *
from meshtool.filters.base_filters import *

from pandacore import setupPandaApp, getBaseNodePath
from pandacontrols import KeyboardMovement, MouseDrag

def runViewer(mesh):
    p3dApp = setupPandaApp(mesh)
    p3dApp.render.analyze()
    KeyboardMovement()
    MouseDrag(getBaseNodePath(p3dApp.render))
    p3dApp.run()

def FilterGenerator():
    class PandaViewer(OpFilter):
        def __init__(self):
            super(PandaViewer, self).__init__('viewer', 'Uses panda3d to bring up a viewer')
        def apply(self, mesh):
            runViewer(mesh)
            return mesh

    return PandaViewer()
