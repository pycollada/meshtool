from meshtool.args import *
from meshtool.filters.base_filters import *
import collada
import math
import itertools

def humanize_bytes(bytes, precision=1):
    abbrevs = (
        (1<<50L, 'PB'),
        (1<<40L, 'TB'),
        (1<<30L, 'GB'),
        (1<<20L, 'MB'),
        (1<<10L, 'KB'),
        (1, 'bytes')
    )
    if bytes == 1:
        return '1 byte'
    for factor, suffix in abbrevs:
        if math.fabs(bytes) >= factor:
            break
    return '%.*f %s' % (precision, bytes / factor, suffix)

def getTextureRAM(mesh):
    total_image_area = 0
    for cimg in mesh.images:
        pilimg = cimg.pilimage
        
        if pilimg:
            total_image_area += pilimg.size[0] * pilimg.size[1] * len(pilimg.getbands())
        else:
            #PIL doesn't support DDS, so if loading failed, try and load it as a DDS with panda3d
            imgdata = cimg.data
            
            #if we can't even load the image's data, can't convert
            if imgdata is None:
                continue
            
            try:
                from panda3d.core import Texture
                from panda3d.core import StringStream
            except ImportError:
                #if panda3d isn't installed and PIL failed, can't convert
                continue
            
            t = Texture()
            success = t.readDds(StringStream(imgdata))
            if success == 0:
                #failed to load as DDS, so let's give up
                continue
            total_image_area += t.getXSize() * t.getYSize() * 3

    return total_image_area

def getSceneInfo(mesh):
    num_triangles = 0
    num_draw_raw = 0
    num_draw_with_instances = 0
    num_draw_with_batching = 0
    num_lines = 0
    
    geom_name_cache = {}
    material_cache = {}
    
    scene = mesh.scene
    if scene is None:
        return (0, 0, 0, 0, 0)
    for boundobj in itertools.chain(scene.objects('geometry'), scene.objects('controller')):
        if isinstance(boundobj, collada.geometry.BoundGeometry):
            boundgeom = boundobj
        else:
            boundgeom = boundobj.geometry
        geom_id = boundgeom.original.id
        for boundprim in boundgeom.primitives():
            num_draw_raw += 1
            if geom_id not in geom_name_cache:
                num_draw_with_instances += 1
            if boundprim.material not in material_cache:
                num_draw_with_batching += 1
                material_cache[boundprim.material] = None
            if isinstance(boundprim, collada.lineset.BoundLineSet):
                num_lines += len(boundprim)
            else:
                if not isinstance(boundprim, collada.triangleset.BoundTriangleSet):
                    boundprim = boundprim.triangleset()
                num_triangles += len(boundprim)
        geom_name_cache[geom_id] = None

    return (num_triangles, num_draw_raw, num_draw_with_instances, num_draw_with_batching, num_lines)

def getRenderInfo(mesh):
    num_triangles, num_draw_raw, num_draw_with_instances, num_draw_with_batching, num_lines = getSceneInfo(mesh)
    return {'texture_ram': getTextureRAM(mesh),
            'num_triangles': num_triangles,
            'num_draw_raw': num_draw_raw,
            'num_draw_with_instances': num_draw_with_instances,
            'num_draw_with_batching': num_draw_with_batching,
            'num_lines': num_lines}

def printRenderInfo(mesh):
    render_info = getRenderInfo(mesh)
    print 'Total texture RAM required: %s' % humanize_bytes(render_info['texture_ram'])
    print 'Total triangles: %d' % render_info['num_triangles']
    print 'Raw number of draw calls: %d' % render_info['num_draw_raw']
    print 'Number of draw calls with instance batching: %d' % render_info['num_draw_with_instances']
    print 'Number of draw calls with instance and material batching: %d' % render_info['num_draw_with_batching']
    print 'Number of lines: %d' % render_info['num_lines']

def FilterGenerator():
    class PrintRenderInfoFilter(OpFilter):
        def __init__(self):
            super(PrintRenderInfoFilter, self).__init__('print_render_info', 'Prints estimated number of batches, total number of triangles, and total texture memory')
        def apply(self, mesh):
            printRenderInfo(mesh)
            return mesh
    return PrintRenderInfoFilter()
