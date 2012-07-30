from meshtool.args import FilterArgument
from meshtool.filters.base_filters import SaveFilter, FilterException
import os.path
from math import pi
from meshtool.util import Image

from meshtool.filters.panda_filters.pandacore import setupPandaApp, getScreenshot, setCameraAngle

def saveRotateScreenshots(p3dapp, basename, N, W, H):
    min_angle = 0.0
    max_angle = 2.0 * pi
    incr_angle = (max_angle-min_angle) / N

    for i in range(N):
        cur_angle = min_angle + incr_angle * i
        setCameraAngle(cur_angle)
        cur_angle += incr_angle
        p3dapp.taskMgr.step()
        pilimage = getScreenshot(p3dapp)
        pilimage.thumbnail((W,H), Image.ANTIALIAS)
        pilimage.save("%s.%d.png" % (basename, i), optimize=1)

def FilterGenerator():
    class SaveScreenshotFilter(SaveFilter):
        def __init__(self):
            super(SaveScreenshotFilter, self).__init__('save_rotate_screenshots',
                            'Saves N screenshots of size WxH, rotating evenly spaced around the object ' +
                            'between shots. Each screenshot file will be file.n.png')
            self.arguments.append(FilterArgument("N", "Number of screenshots to save"))
            self.arguments.append(FilterArgument("W", "Width of thumbnail"))
            self.arguments.append(FilterArgument("H", "Height of screenshot to save"))
        def apply(self, mesh, filename, N, W, H):
            
            try: N = int(N)
            except ValueError: raise FilterException("Given value for N not a valid integer")
            if N < 1:
                raise FilterException("Given value for N not a valid integer")
            
            try: W = int(W)
            except ValueError: raise FilterException("Given value for W not a valid integer")
            if W < 1:
                raise FilterException("Given value for W not a valid integer")
            
            try: H = int(H)
            except ValueError: raise FilterException("Given value for H not a valid integer")
            if H < 1:
                raise FilterException("Given value for H not a valid integer")
            
            if os.path.exists(filename):
                raise FilterException("specified filename already exists")
            
            p3dApp = setupPandaApp(mesh)
            saveRotateScreenshots(p3dApp, filename, N, W, H)
            
            return mesh
        
    return SaveScreenshotFilter()
from meshtool.filters import factory
factory.register(FilterGenerator().name, FilterGenerator)