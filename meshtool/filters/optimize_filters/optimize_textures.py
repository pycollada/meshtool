from meshtool.filters.base_filters import OptimizationFilter
from meshtool.util import Image
import sys
from StringIO import StringIO

import numpy

def optimizeTextures(mesh):
    
    previous_images = []

    for cimg in mesh.images:
        previous_images.append(cimg.path)
        
        pilimg = cimg.pilimage
        
        #PIL doesn't support DDS, so if loading failed, try and load it as a DDS with panda3d
        if pilimg is None:
            imgdata = cimg.data
            
            #if we can't even load the image's data, can't convert
            if imgdata is None:
                print >> sys.stderr, "Couldn't load image data"
                continue
            
            try:
                from panda3d.core import Texture
                from panda3d.core import StringStream
                from panda3d.core import PNMImage
            except ImportError:
                #if panda3d isn't installed and PIL failed, can't convert
                print >> sys.stderr, 'Tried loading image with PIL and DDS and both failed'
                continue
            
            t = Texture()
            success = t.readDds(StringStream(imgdata))
            if success == 0:
                #failed to load as DDS, so let's give up
                print >> sys.stderr, 'Tried loading image as DDS and failed'
                continue

            #convert DDS to PNG
            outdata = t.getRamImageAs('RGB').getData()
            try:
                im = Image.fromstring('RGB', (t.getXSize(), t.getYSize()), outdata)
                im.load()
            except IOError:
                #Any problem with panda3d might generate an invalid image buffer, so don't convert this
                print >> sys.stderr, 'Problem loading DDS file with PIL'
                continue
            
            pilimg = im
        
        if pilimg.format == 'JPEG':
            #PIL image is already in JPG format so don't convert
            continue
        
        if 'A' in pilimg.getbands():
            alpha = numpy.array(pilimg.split()[-1].getdata())
            if not numpy.any(alpha < 255):
                alpha = None
                #this means that none of the pixels are using alpha, so convert to RGB
                pilimg = pilimg.convert('RGB') 
        
        if 'A' in pilimg.getbands():
            #save textures with an alpha channel in PNG
            output_format = 'PNG'
            output_extension = '.png'
            output_options = {'optimize':True}
        else:
            if pilimg.format != 'RGB':
                pilimg = pilimg.convert("RGB")
            #otherwise save as JPEG since it gets 
            output_format = 'JPEG'
            output_extension = '.jpg'
            output_options = {'quality':95, 'optimize':True}
        
        if cimg.path.lower()[-len(output_extension):] != output_extension:
            dot = cimg.path.rfind('.')
            before_ext = cimg.path[0:dot] if dot != -1 else cimg.path
            while before_ext + output_extension in previous_images:
                before_ext = before_ext + '-x'
            cimg.path = before_ext + output_extension
            previous_images.append(cimg.path)
        
        outbuf = StringIO()
               
        try:
            pilimg.save(outbuf, output_format, **output_options)
        except IOError, ex:
            print ex

        cimg.data = outbuf.getvalue()

def FilterGenerator():
    class OptimizeTexturesFilter(OptimizationFilter):
        def __init__(self):
            super(OptimizeTexturesFilter, self).__init__('optimize_textures', 'Converts all textures with alpha channel to PNG and ones without to JPEG')
        def apply(self, mesh):
            optimizeTextures(mesh)
            return mesh
    return OptimizeTexturesFilter()
from meshtool.filters import factory
factory.register(FilterGenerator().name, FilterGenerator)