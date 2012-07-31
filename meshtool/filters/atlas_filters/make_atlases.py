from meshtool.filters.base_filters import OptimizationFilter
from meshtool.filters.atlas_filters.rectpack import RectPack
from meshtool.util import Image
import math
import collada
import numpy
import itertools
from StringIO import StringIO

#The maximum width or height of a texture
MAX_IMAGE_DIMENSION = 4096
MAX_TILING_DIMENSION = 2048

class TexcoordSet(object):
    """Container class holding all the information needed to indentify and locate a
    single set of texture coordinates"""
    def __init__(self, geom_id, prim_index, texcoordset_index, setnum):
        self.geom_id = geom_id
        self.prim_index = prim_index
        self.texcoordset_index = texcoordset_index
        self.setnum = setnum
    def __eq__(self, other):
        return self.geom_id == other.geom_id and \
                self.prim_index == other.prim_index and \
                self.setnum == other.setnum
    def __repr__(self):
        return "<Texcoordset %s:%d:%d:%d>" % (self.geom_id, self.prim_index, self.texcoordset_index, self.setnum)
    def __str__(self): return self.__repr__()
    def __hash__(self):
        return hash("%s_%s_%s" % (self.geom_id, str(self.prim_index), str(self.setnum)))

def getTexcoordToImgMapping(mesh):

    #get a list of all texture coordinate sets
    all_texcoords = {}
    for geom in mesh.geometries:
        for prim_index, prim in enumerate(geom.primitives):
            inputs = prim.getInputList().getList()
            texindex = 0
            for offset, semantic, srcid, setid in inputs:
                if semantic == 'TEXCOORD':
                    try: setid = int(setid)
                    except (ValueError, TypeError): setid = 0
                    texset = TexcoordSet(geom.id, prim_index, texindex, setid)
                    texindex += 1
                    all_texcoords[texset] = []
    
    #create a mapping between each texcoordset and the images they get bound to by traversing scenes
    for scene in mesh.scenes:
        for boundobj in itertools.chain(scene.objects('geometry'), scene.objects('controller')):
            if isinstance(boundobj, collada.geometry.BoundGeometry):
                boundgeom = boundobj
            else:
                boundgeom = boundobj.geometry
            geom_id = boundgeom.original.id
            for prim_index, boundprim in enumerate(boundgeom.primitives()):
                if boundprim.material is not None:
                    effect = boundprim.material.effect
                    inputmap = boundprim.inputmap
                    for prop in itertools.chain(effect.supported, ['bumpmap']):
                        propval = getattr(effect, prop)
                        if type(propval) is collada.material.Map:
                            if propval.texcoord in inputmap:
                                cimg = propval.sampler.surface.image
                                semantic, setid = inputmap[propval.texcoord]
                                if not setid: setid = 0
                                else:
                                    try: setid = int(setid)
                                    except (ValueError, TypeError): setid = 0
                                if semantic == 'TEXCOORD':
                                    texset = TexcoordSet(geom_id, prim_index, -1, setid)
                                    if texset in all_texcoords:
                                        if cimg.path not in all_texcoords[texset]:
                                            all_texcoords[texset].append(cimg.path)
    
    #remove any texture coordinates that dont get mapped to textures
    all_texcoords = dict( (texset, imglist)
                          for texset, imglist in all_texcoords.iteritems()
                          if len(imglist) > 0 )
    
    return all_texcoords

def combinePacks(to_del1, to_del2):
    if to_del1 is None:
        return to_del2
    elif to_del2 is None:
        return to_del1
    else:
        for geom, primlist in to_del2.iteritems():
            if geom in to_del1:
                to_del1[geom].extend(primlist)
            else:
                to_del1[geom] = primlist
        return to_del1

def splitAlphas(unique_images):
    group1 = dict(( (path, pilimg) for path, pilimg in unique_images.iteritems() if 'A' in pilimg.getbands() ))
    group2 = dict(( (path, pilimg) for path, pilimg in unique_images.iteritems() if 'A' not in pilimg.getbands() ))

    return group1, group2

def packImages(mesh, img2texs, unique_images, image_scales):
    #if there aren't at least two images left, nothing to do
    if len(unique_images) < 2:
        return
    
    #okay, now we can start packing!
    rp = RectPack(MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION)
    for path, pilimg in unique_images.iteritems():
        width, height = pilimg.size
        rp.addRectangle(path, width, height)
    success = rp.pack()

    if not success:
        if len(rp.rejects) == len(unique_images):
            #this means that nothing could be packed into the max size
            # if not a single image can be packed into the max size
            # then there's no point in continuing
            return
                
        group1 = dict(( (path, pilimg) for path, pilimg in unique_images.iteritems() if path in rp.rejects ))
        group2 = dict(( (path, pilimg) for path, pilimg in unique_images.iteritems() if path not in rp.rejects ))
        
        return combinePacks(packImages(mesh, img2texs, group1, image_scales),
                    packImages(mesh, img2texs, group2, image_scales))
    
    width = rp.width
    height = rp.height
    
    print "actually making atlas of size %dx%d with %d subimages referenced by %d texcoords" % \
        (width, height, len(unique_images), sum([len(img2texs[imgpath]) for imgpath in unique_images]))
    atlasimg = Image.new('RGBA', (width, height), (0,0,0,255))
    
    to_del = {}
    for path, pilimg in unique_images.iteritems():
        x,y,w,h = rp.getPlacement(path)
        atlasimg.paste(pilimg, (x,y,x+w,y+h))
        
        x,y,w,h,width,height = (float(i) for i in (x,y,w,h,width,height))
        
        for texset in img2texs[path]:
            geom = mesh.geometries[texset.geom_id]
            prim = geom.primitives[texset.prim_index]
            texarray = numpy.copy(prim.texcoordset[texset.texcoordset_index])
            tile_x, tile_y = (float(i) for i in image_scales[path])
            
            #this shrinks the texcoords to 0,1 range for a tiled image
            if tile_x > 1.0:
                texarray[:,0] = texarray[:,0] / tile_x
            if tile_y > 1.0:
                texarray[:,1] = texarray[:,1] / tile_y
            
            #this computes the coordinates of the lowest and highest texel
            # if the texcoords go outside that range, rescale so they are inside
            # suggestion by nvidia texture atlasing white paper
            minx, maxx = numpy.min(texarray[:,0]), numpy.max(texarray[:,0])
            miny, maxy = numpy.min(texarray[:,1]), numpy.max(texarray[:,1])
            lowest_x = 0.5 / w
            lowest_y = 0.5 / h
            highest_x = 1.0 - lowest_x
            highest_y = 1.0 - lowest_y
            if minx < lowest_x or maxx > highest_x:
                texarray[:,0] = texarray[:,0] * (highest_x - lowest_x) + lowest_x
            if miny < lowest_y or maxy > highest_y:
                texarray[:,1] = texarray[:,1] * (highest_y - lowest_y) + lowest_y

            #this rescales the texcoords to map to the new atlas location
            texarray[:,0] = texarray[:,0] * (w / width) + (x / (width-1))
            texarray[:,1] = texarray[:,1] * (h / height) + (1.0 - (y+h)/height)
            
            oldsources = prim.getInputList().getList()
            newsources = collada.source.InputList()
            for (offset, semantic, source, setid) in oldsources:
                if semantic == 'TEXCOORD' and (setid is None or int(setid) == texset.texcoordset_index):
                    orig_source = source
                    i=0
                    while source[1:] in geom.sourceById:
                        source = orig_source + '-atlas-' + str(i)
                        i += 1
                    new_tex_src = collada.source.FloatSource(source[1:], texarray, ('S', 'T'))
                    geom.sourceById[source[1:]] = new_tex_src
                newsources.addInput(offset, semantic, source, setid)
            
            if geom not in to_del:
                to_del[geom] = []
            to_del[geom].append(texset.prim_index)
            
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
    
    baseimgid = imgs_deleted[0].id + '-atlas'
    baseimgpath = './atlas'
    newimgid = baseimgid
    newimgpath = baseimgpath
    ct = 0
    while newimgid in mesh.images or newimgpath + '.png' in [cimg.path for cimg in mesh.images]:
        newimgid = baseimgid + '-' + str(ct)
        newimgpath = baseimgpath + '-' + str(ct)
        ct += 1

    newimgpath = newimgpath + '.png'
    newcimage = collada.material.CImage(newimgid, newimgpath, mesh)
    
    strbuf = StringIO()
    atlasimg.save(strbuf, 'PNG', optimize=True)
    newcimage._data = strbuf.getvalue()
    mesh.images.append(newcimage)
    
    for effect in mesh.effects:
        for param in effect.params:
            if type(param) is collada.material.Surface:
                if param.image in imgs_deleted:
                    param.image = newcimage
                    
    return to_del

def makeAtlases(mesh):
    # get a mapping from path to actual image, since theoretically you could have
    # the same image file in multiple image nodes
    unique_images = {}
    image_scales = {}
    for cimg in mesh.images:
        path = cimg.path
        if path not in unique_images:
            unique_images[path] = cimg.pilimage
            image_scales[path] = (1,1)
    
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
        
        valid_range = False
        if len(imgpaths) == 1:
            texarray = mesh.geometries[texset.geom_id] \
                        .primitives[texset.prim_index] \
                        .texcoordset[texset.texcoordset_index]
            
            width, height = unique_images[imgpaths[0]].size
            tile_x = int(numpy.ceil(numpy.max(texarray[:,0])))
            tile_y = int(numpy.ceil(numpy.max(texarray[:,1])))
            stretched_width = tile_x * width
            stretched_height = tile_y * height
            
            #allow tiling of texcoords if the final tiled image is <= MAX_TILING_DIMENSION
            if numpy.min(texarray) < 0.0:
                valid_range = False
            elif stretched_width > MAX_TILING_DIMENSION or stretched_height > MAX_TILING_DIMENSION:
                valid_range = False
            else:
                valid_range = True
        
            if valid_range:
                scale_x, scale_y = image_scales[imgpaths[0]]
                scale_x = max(scale_x, tile_x)
                scale_y = max(scale_y, tile_y)
                image_scales[imgpaths[0]] = (scale_x, scale_y)
        
        if len(imgpaths) > 1 or not valid_range:
            for imgpath in imgpaths:
                if imgpath not in imgs_to_delete:
                    imgs_to_delete.append(imgpath)
            texs_to_delete.append(texset)
    for imgpath, pilimg in unique_images.iteritems():
        if max(pilimg.size) > MAX_IMAGE_DIMENSION and imgpath not in imgs_to_delete:
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
    
    for path, pilimg in unique_images.iteritems():
        tile_x, tile_y = image_scales[path]
        width, height = pilimg.size
        if tile_x > 1 or tile_y > 1:
            if 'A' in pilimg.getbands():
                imgformat = 'RGBA'
                initval = (0,0,0,255)
            else:
                imgformat = 'RGB'
                initval = (0,0,0)
            tiled_img = Image.new(imgformat, (width*tile_x, height*tile_y), initval)
            for x in range(tile_x):
                for y in range(tile_y):
                    tiled_img.paste(pilimg, (x*width,y*height))
            pilimg = tiled_img
            width, height = pilimg.size
        
        #round down to power of 2
        width = int(math.pow(2, int(math.log(width, 2))))
        height = int(math.pow(2, int(math.log(height, 2))))
        if (width, height) != pilimg.size:
            pilimg = pilimg.resize((width, height), Image.ANTIALIAS)
        
        unique_images[path] = pilimg
    
    group1, group2 = splitAlphas(unique_images)
    to_del = combinePacks(packImages(mesh, img2texs, group1, image_scales),
                packImages(mesh, img2texs, group2, image_scales))
    if to_del is not None:
        for geom, primindices in to_del.iteritems():
            for i in sorted(primindices, reverse=True):
                del geom.primitives[i]

def FilterGenerator():
    class MakeAtlasesFilter(OptimizationFilter):
        def __init__(self):
            super(MakeAtlasesFilter, self).__init__('make_atlases', 'Makes a texture atlas with the textures referenced in the ' +
                                                    'given file. Extremely conservative: will only make an atlas from texture ' +
                                                    'coordinates inside the range (0,1). Atlas can be saved with --save_collada_zip.')
        def apply(self, mesh):
            makeAtlases(mesh)
            return mesh
    return MakeAtlasesFilter()
from meshtool.filters import factory
factory.register(FilterGenerator().name, FilterGenerator)