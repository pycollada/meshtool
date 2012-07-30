from meshtool.filters.base_filters import SaveFilter, FilterException
import os.path

from meshtool.filters.panda_filters.pandacore import setupPandaApp, getScreenshot

def saveScreenshot(p3dapp, filename):
    pilimage = getScreenshot(p3dapp)
    pilimage.save(filename, optimize=1)

def FilterGenerator():
    class SaveScreenshotFilter(SaveFilter):
        def __init__(self):
            super(SaveScreenshotFilter, self).__init__('save_screenshot', 'Saves a screenshot of the rendered collada file')
        def apply(self, mesh, filename):
            if os.path.exists(filename):
                raise FilterException("specified filename already exists")
            p3dApp = setupPandaApp(mesh)
            saveScreenshot(p3dApp, filename)
            return mesh
        
    return SaveScreenshotFilter()
from meshtool.filters import factory
factory.register(FilterGenerator().name, FilterGenerator)