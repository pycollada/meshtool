from meshtool.args import *
from meshtool.filters.base_filters import *

from pandacore import setupPandaApp, spinCameraTask
from pandacontrols import KeyboardMovement

def runViewer(mesh):
    p3dApp = setupPandaApp(mesh)
    p3dApp.render.analyze()
    KeyboardMovement()
    p3dApp.run()

def FilterGenerator():
    class PandaViewer(OpFilter):
        def __init__(self):
            super(PandaViewer, self).__init__('viewer', 'Uses panda3d to bring up a viewer')
        def apply(self, mesh):
            runViewer(mesh)
            return mesh

    return PandaViewer()
