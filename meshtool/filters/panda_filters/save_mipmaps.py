from meshtool.args import *
from meshtool.filters.base_filters import *
import os.path

from meshtool.filters.panda_filters.pandacore import getTexture
import collada
import Image
from StringIO import StringIO

def getMipMaps(mesh):
    mipmaps = {}
    for effect in mesh.effects:
        for prop in effect.supported:
            propval = getattr(effect, prop)
            if isinstance(propval, collada.material.Map):
                tex = getTexture(propval)
                
                tex.generateRamMipmapImages()

                mips = []
                for i in range(tex.getExpectedNumMipmapLevels()):
                    width = tex.getExpectedMipmapXSize(i)
                    height = tex.getExpectedMipmapYSize(i)
                    outdata = tex.getRamMipmapImage(i).getData()
                    im = Image.fromstring('RGB', (width, height), outdata, 'raw', 'BGR', 0, -1)
                    im.load()
                    mips.append(im)
                mips.reverse()
                mipmaps[propval.sampler.surface.image.path] = mips
    return mipmaps

def saveMipMaps(mesh):
    mipmaps = getMipMaps(mesh)
    
    if not mesh.filename:
        return False
    reldir = os.path.dirname(mesh.filename)
    if not os.path.isdir(reldir):
        return False
    
    for imgpath, mips in mipmaps.iteritems():
        saveto = os.path.normpath(os.path.join(reldir, imgpath))
        saveto += '.mipmap'
        out_strs = []
        for img in mips:
            buf = StringIO()
            img.save(buf, format='PNG', optimize=1)
            out_strs.append(buf.getvalue())
        big_str = ''.join(out_strs)
        f = open(saveto, 'wb')
        f.write(big_str)
        f.close()
    return True

def FilterGenerator():
    class SaveMipMapsFilter(OpFilter):
        def __init__(self):
            super(SaveMipMapsFilter, self).__init__('save_mipmaps', 'Saves mipmaps to disk in concatenated PNG format in the same location as textures but with an added .mipmap')
        def apply(self, mesh):
            succ = saveMipMaps(mesh)
            if not succ:
                raise FilterException('Failed to save mipmaps')
            return mesh
        
    return SaveMipMapsFilter()