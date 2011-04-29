from meshtool.args import *
from meshtool.filters.base_filters import *
from meshtool.filters.atlas_filters.rectpack import RectPack
from PIL import Image
import math
import collada

def makeAtlas(mesh):
    unique_images = {}
    for cimg in mesh.images:
        path = cimg.path
        if path not in unique_images:
            unique_images[path] = cimg.pilimage
    
    print unique_images
    
    rp = RectPack()
    for path, pilimg in unique_images.iteritems():
        width, height = pilimg.size
        print path, width, height
        rp.addRectangle(path, width, height)
    rp.pack()
    
    print "recsize", rp.width, rp.height
    
    #round up to power of 2
    width = int(math.pow(2, math.ceil(math.log(rp.width, 2))))
    height = int(math.pow(2, math.ceil(math.log(rp.height, 2))))
    
    print "newsize", width, height
    
    atlasimg = Image.new('RGBA', (width, height))
    for path, pilimg in unique_images.iteritems():
        x,y,w,h = rp.getPlacement(path)
        atlasimg.paste(pilimg, (x,y,x+w,y+h))
    
    for geom in mesh.scene.objects('geometry'):
        print geom

def FilterGenerator():
    class MakeAtlasFilter(OpFilter):
        def __init__(self):
            super(MakeAtlasFilter, self).__init__('make_atlas', 'Makes a texture atlas with the textures referenced in the given file')
        def apply(self, mesh):
                      
            makeAtlas(mesh)
             
#            rp = RectPack()
#            paths = []
#            basepath = os.path.dirname(mesh.filename)
#            for t in mesh.images:
#                path = os.path.join(basepath, t.path)
#                im = Image.open(path)
#                width, height = im.size
#                rp.addRectangle(path, width, height)
#                paths.append(path)
#            rp.pack()
#            img = Image.new("RGBA",(rp.width, rp.height))
#            for path in paths:
#                im = Image.open(path)
#                x,y,w,h = rp.getPlacement(path)
#                img.paste(im, (x,y,x+w,y+h))
#                # for pt in vertices_that_use_t:
#                #    (x,y) = pt.textcoords
#                #    pt.textcoords = (px + x, py + y)
#                #    pt.texture = img
#            img.save(filename)
            
            return mesh
    return MakeAtlasFilter()
