from meshtool.filters.base_filters import PrintFilter
import collada

def printMeshInfo(mesh):
    indent = '  '
    
    print 'Cameras: %d' % len(mesh.cameras)
    for cam in mesh.cameras:
        print indent, cam

    print 'Lights: %d' % len(mesh.lights)
    for light in mesh.lights:
        print indent, light
        
    print 'Materials: %d' % len(mesh.materials)
    for material in mesh.materials:
        print indent, material
        
    print 'Effects: %d' % len(mesh.effects)
    for effect in mesh.effects:
        print indent, effect
        
    print 'Images: %d' % len(mesh.images)
    for image in mesh.images:
        print indent, image
        
    print 'Geometries: %d' % len(mesh.geometries)
    for geom in mesh.geometries:
        print indent, geom
        for srcid, src in geom.sourceById.iteritems():
            if isinstance(src, collada.source.Source):
                print indent, indent, "%s: %s" % (srcid, src)
        for prim in geom.primitives:
            print indent, indent, prim

def FilterGenerator():
    class PrintInfoFilter(PrintFilter):
        def __init__(self):
            super(PrintInfoFilter, self).__init__('print_info', 'Prints a bunch of information about the mesh to the console')
        def apply(self, mesh):
            printMeshInfo(mesh)
            return mesh
    return PrintInfoFilter()
from meshtool.filters import factory
factory.register(FilterGenerator().name, FilterGenerator)