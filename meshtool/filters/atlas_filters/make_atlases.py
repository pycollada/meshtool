from meshtool.args import *
from meshtool.filters.base_filters import *
from meshtool.filters.atlas_filters.rectpack import RectPack
from PIL import Image
import math
import collada
import numpy
import posixpath
from StringIO import StringIO

#Following is a workaround for setting quality=95, optimize=1 when encoding JPEG
#Otherwise, an error is output when trying to save
#Taken from http://mail.python.org/pipermail/image-sig/1999-August/000816.html
import ImageFile
ImageFile.MAXBLOCK = 1000000 # default is 64k

class TexcoordSet(object):
    """Container class holding all the information needed to indentify and locate a
    single set of texture coordinates"""
    def __init__(self, geom_id, prim_index, texcoordset_index):
        self.geom_id = geom_id
        self.prim_index = prim_index
        self.texcoordset_index = texcoordset_index
    def __eq__(self, other):
        return self.geom_id == other.geom_id and \
                self.prim_index == other.prim_index and \
                self.texcoordset_index == other.texcoordset_index
    def __repr__(self):
        return "<Texcoordset %s:%d:%d>" % (self.geom_id, self.prim_index, self.texcoordset_index)
    def __str__(self): return self.__repr__()
    def __hash__(self):
        return hash("%s_%s_%s" % (self.geom_id, str(self.prim_index), str(self.texcoordset_index)))

def getTexcoordToImgMapping(mesh):

    #get a list of all texture coordinate sets
    all_texcoords = {}
    for geom in mesh.geometries:
        for prim_index, prim in enumerate(geom.primitives):
            for texcoordset_index in range(len(prim.texcoordset)):
                texset = TexcoordSet(geom.id, prim_index, texcoordset_index)
                all_texcoords[texset] = []
    
    #create a mapping between each texcoordset and the images they get bound to by traversing scenes
    for scene in mesh.scenes:
        for boundgeom in scene.objects('geometry'):
            geom_id = boundgeom.original.id
            for prim_index, boundprim in enumerate(boundgeom.primitives()):
                if boundprim.material is not None:
                    effect = boundprim.material.effect
                    inputmap = boundprim.inputmap
                    for prop in effect.supported:
                        propval = getattr(effect, prop)
                        if type(propval) is collada.material.Map:
                            if propval.texcoord in inputmap:
                                cimg = propval.sampler.surface.image
                                semantic, set = inputmap[propval.texcoord]
                                if not set: set = 0
                                else:
                                    try: set = int(set)
                                    except ValueError: set = 0
                                if semantic == 'TEXCOORD':
                                    texset = TexcoordSet(geom_id, prim_index, set)
                                    if texset in all_texcoords:
                                        if cimg.path not in all_texcoords[texset]:
                                            all_texcoords[texset].append(cimg.path)
    
    #remove any texture coordinates that dont get mapped to textures
    all_texcoords = dict( (texset, imglist)
                          for texset, imglist in all_texcoords.iteritems()
                          if len(imglist) > 0 )
    
    return all_texcoords

def packImages(mesh, img2texs, unique_images):
    #if there aren't at least two images left, nothing to do
    if len(unique_images) < 2:
        return
    
    #okay, now we can start packing!
    rp = RectPack()
    for path, pilimg in unique_images.iteritems():
        width, height = pilimg.size
        rp.addRectangle(path, width, height)
    rp.pack()
    
    #round up to power of 2
    width = int(math.pow(2, math.ceil(math.log(rp.width, 2))))
    height = int(math.pow(2, math.ceil(math.log(rp.height, 2))))
    
    # don't want to create gigantic atlases
    # if this happens, split into two groups instead
    if width > 2048 or height > 2048:
        groups = {}
        groups[0] = {}
        groups[1] = {}
        curgroup = 0
        for (imgpath, pilimg) in sorted(unique_images.items(),
                                        key=lambda tup: max(tup[1].size),
                                        reverse=True):
            groups[curgroup][imgpath] = pilimg
            curgroup+=1
            if curgroup == 2: curgroup = 0
        packImages(mesh, img2texs, groups[0])
        packImages(mesh, img2texs, groups[1])
        return
    
    print "actually making atlas of size %dx%d with %d subimages referenced by %d texcoords" % \
        (width, height, len(unique_images), sum([len(img2texs[imgpath]) for imgpath in unique_images]))
    atlasimg = Image.new('RGBA', (width, height))
    for path, pilimg in unique_images.iteritems():
        x,y,w,h = rp.getPlacement(path)
        atlasimg.paste(pilimg, (x,y,x+w,y+h))
        
        for texset in img2texs[path]:
            geom = mesh.geometries[texset.geom_id]
            prim = geom.primitives[texset.prim_index]
            texarray = prim.texcoordset[texset.texcoordset_index]
            texarray = texarray - numpy.floor(texarray)
            
            x_scale = float(w) / width
            y_scale = float(h) / height
            x_offset = float(x) / width
            y_offset = float(y) / height

            texarray[:,0] = texarray[:,0] * x_scale + x_offset
            texarray[:,1] = texarray[:,1] * y_scale + y_offset
            
            oldsources = prim.getInputList().getList()
            newsources = collada.source.InputList()
            for (offset, semantic, source, set) in oldsources:
                if semantic == 'TEXCOORD' and int(set) == texset.texcoordset_index:
                    while source[1:] in geom.sourceById:
                        source += '-atlas'
                    new_tex_src = collada.source.FloatSource(source[1:], texarray, ('S', 'T'))
                    geom.sourceById[source[1:]] = new_tex_src
                newsources.addInput(offset, semantic, source, set)
            
            del geom.primitives[texset.prim_index]
            
            if type(prim) is collada.triangleset.TriangleSet:
                prim.index.shape = -1
                newprim = geom.createTriangleSet(prim.index, newsources, prim.material)
            elif type(prim) is collada.polylist.Polylist:
                prim.index.shape = -1
                prim.vcounts.shape = -1
                newprim = geom.createPolylist(prim.index, prim.vcounts, newsources, prim.material)
            elif type(prim) is collada.polygons.Polygons:
                prim.index.shape = -1
                newprim = geom.createPolygons(prim.index, newsources, prim.material)
            elif type(prim) is collada.lineset.LineSet:
                prim.index.shape = -1
                newprim = geom.createLineSet(prim.index, newsources, prim.material)
            else:
                raise Exception("Unknown primitive type")
            
            geom.primitives.append(newprim)
        
    imgs_deleted = [cimg for cimg in mesh.images if cimg.path in unique_images]
    mesh.images = [cimg for cimg in mesh.images if cimg.path not in unique_images]
    newimgid = imgs_deleted[0].id + '-atlas'
    newimgpath = './atlas'
    
    while newimgid in mesh.images:
        newimgid = newimgid + '-atlas'
    while newimgpath + '.png' in [cimg.path for cimg in mesh.images]:
        newimgpath = newimgpath + '-x'
    newimgpath = newimgpath + '.png'
    
    newcimage = collada.material.CImage(newimgid, newimgpath, mesh)
    
    strbuf = StringIO()
    atlasimg.save(strbuf, 'PNG', optimize=True)
    newcimage._data = strbuf.getvalue()
    mesh.images.append(newcimage)
    
    for effect in mesh.effects:
        for prop in effect.supported:
            propval = getattr(effect, prop)
            if type(propval) is collada.material.Map:
                propval.sampler.surface.image = newcimage

def makeAtlases(mesh):
    # get a mapping from path to actual image, since theoretically you could have
    # the same image file in multiple image nodes
    unique_images = {}
    for cimg in mesh.images:
        path = cimg.path
        if path not in unique_images:
            unique_images[path] = cimg.pilimage
    
    # get a mapping from texture coordinates to all of the images they get bound to
    tex2img = getTexcoordToImgMapping(mesh)
    
    # don't consider any texcoords that are used more than once with different images
    # could probably have an algorithm that takes this into account, but it would
    # require some complex groupings. any images referenced have to also not be
    # considered for atlasing, as well as any tex coords that reference them
    # also filter out any images that are >= 1024 in either dimension
    texs_to_delete = []
    imgs_to_delete = []
    for texset, imgpaths in tex2img.iteritems():
        
        texarray = mesh.geometries[texset.geom_id] \
                    .primitives[texset.prim_index] \
                    .texcoordset[texset.texcoordset_index]
        valid_range = False
        if numpy.min(texarray) >= -0.00001 and numpy.max(texarray) <= 1.00001:
            valid_range = True
        
        if len(imgpaths) > 1 or not valid_range:
            for imgpath in imgpaths:
                if imgpath not in imgs_to_delete:
                    imgs_to_delete.append(imgpath)
            texs_to_delete.append(texset)
    for imgpath, pilimg in unique_images.iteritems():
        if max(pilimg.size) >= 1024 and imgpath not in imgs_to_delete:
            imgs_to_delete.append(imgpath)
    for imgpath in imgs_to_delete:
        for texset, imgpaths in tex2img.iteritems():
            if imgpaths[0] == imgpath and texset not in texs_to_delete:
                texs_to_delete.append(texset)
        del unique_images[imgpath]
    for texset in texs_to_delete:
        del tex2img[texset]
    
    # now make a mapping between images and the tex coords that have to be
    # updated if it is atlased
    img2texs = {}
    for imgpath in unique_images:
        img2texs[imgpath] = []
        for texset, imgpaths in tex2img.iteritems():
            if imgpaths[0] == imgpath:
                img2texs[imgpath].append(texset)
    
    packImages(mesh, img2texs, unique_images)

def FilterGenerator():
    class MakeAtlasesFilter(OpFilter):
        def __init__(self):
            super(MakeAtlasesFilter, self).__init__('make_atlases', 'Makes a texture atlas with the textures referenced in the ' +
                                                    'given file. Extremely conservative: will only make an atlas from texture ' +
                                                    'coordinates inside the range (0,1). Atlas can be saved with --save_collada_zip.')
        def apply(self, mesh):
            makeAtlases(mesh)
            return mesh
    return MakeAtlasesFilter()
