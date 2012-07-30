from meshtool.filters.base_filters import VisualizationFilter

from pandacore import setupPandaApp, getBaseNodePath
from pandacontrols import KeyboardMovement, MouseDrag, MouseScaleZoom, MouseCamera, ButtonUtils

def runViewer(mesh):
    p3dApp = setupPandaApp(mesh)
    p3dApp.render.analyze()
    KeyboardMovement()
    ButtonUtils(getBaseNodePath(p3dApp.render))
    MouseDrag(getBaseNodePath(p3dApp.render))
    MouseScaleZoom(getBaseNodePath(p3dApp.render))
    MouseCamera()
    p3dApp.run()

def FilterGenerator():
    class PandaViewer(VisualizationFilter):
        def __init__(self):
            super(PandaViewer, self).__init__('viewer', 'Uses panda3d to bring up a viewer')
        def apply(self, mesh):
            runViewer(mesh)
            return mesh

    return PandaViewer()
from meshtool.filters import factory
factory.register(FilterGenerator().name, FilterGenerator)