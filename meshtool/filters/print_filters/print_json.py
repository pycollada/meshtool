from meshtool.filters.base_filters import PrintFilter
from meshtool.filters.print_filters.print_render_info import getRenderInfo

try:
    import json
except ImportError:
    import simplejson as json

def getJSON(mesh):
    cameras = []
    for cam in mesh.cameras:
        cameras.append({'id':cam.id})
        
    lights = []
    for light in mesh.lights:
        lights.append({'id':light.id, 'type': type(light).__name__})
        
    materials = []
    for material in mesh.materials:
        materials.append({'id':material.id, 'effect':material.effect.id})
        
    effects = []
    for effect in mesh.effects:
        effects.append({'id':effect.id, 'type':effect.shadingtype})
        
    images = []
    for image in mesh.images:
        images.append({'id':image.id, 'name':image.path})
        
    primitives = []
    for geom in mesh.geometries:
        for i, prim in enumerate(geom.primitives):
            primitives.append({'id':"%s%d" % (geom.id, i),
                               'type':type(prim).__name__,
                               'vertices':len(prim.vertex_index) if prim.vertex_index is not None else 0,
                               'normals': prim.normal_index is not None,
                               'texcoords': len(prim.texcoord_indexset) > 0})
        
    json_ret = {'cameras': cameras,
                'lights': lights,
                'materials': materials,
                'effects': effects,
                'images': images,
                'primitives': primitives}
    json_ret.update(getRenderInfo(mesh))
    return json.dumps(json_ret)

def FilterGenerator():
    class PrintJsonFilter(PrintFilter):
        def __init__(self):
            super(PrintJsonFilter, self).__init__('print_json', 'Prints a bunch of information about the mesh in a JSON format')
        def apply(self, mesh):
            print getJSON(mesh)
            return mesh
    return PrintJsonFilter()
from meshtool.filters import factory
factory.register(FilterGenerator().name, FilterGenerator)