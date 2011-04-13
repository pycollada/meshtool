from args import *
from ..base_filters import *
from .rectpack import RectPack
from PIL import Image
import os

def FilterGenerator():
    class MakeAtlasFilter(SaveFilter):
        def __init__(self):
            super(MakeAtlasFilter, self).__init__('make_atlas', 'Makes a texture atlas with the textures referenced in the given file')
        def apply(self, mesh, filename):
            rp = RectPack()
            paths = []
            basepath = os.path.dirname(mesh.filename)
            for t in mesh.images:
                path = os.path.join(basepath, t.path)
                im = Image.open(path)
                width, height = im.size
                rp.addRectangle(path, width, height)
                paths.append(path)
            rp.pack()
            img = Image.new("RGBA",(rp.width, rp.height))
            for path in paths:
                im = Image.open(path)
                x,y,w,h = rp.getPlacement(path)
                img.paste(im, (x,y,x+w,y+h))
                # for pt in vertices_that_use_t:
                #    (x,y) = pt.textcoords
                #    pt.textcoords = (px + x, py + y)
                #    pt.texture = img
            img.save(filename)
            return mesh
    return MakeAtlasFilter()
