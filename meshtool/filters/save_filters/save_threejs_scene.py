import os
from meshtool.filters.base_filters import SaveFilter, FilterException
import numpy
import collada

INDENT = 3
SPACE = " "
NEWLINE = "\n"

def to_json(o, level=0):
    ret = ""
    if isinstance(o, dict):
        ret += "{" + NEWLINE
        comma = ""
        for k,v in o.iteritems():
            ret += comma
            comma = ",\n"
            ret += SPACE * INDENT * (level+1)
            ret += '"' + str(k) + '":' + SPACE
            ret += to_json(v, level + 1)
            
        ret += NEWLINE + SPACE * INDENT * level + "}"
    elif isinstance(o, basestring):
        ret += '"' + o + '"'
    elif isinstance(o, list):
        ret += "[" + ",".join([to_json(e, level+1) for e in o]) + "]"
    elif isinstance(o, bool):
        ret += "true" if o else "false"
    elif isinstance(o, int):
        ret += str(o)
    elif isinstance(o, float):
        ret += '%.7g' % o
    elif isinstance(o, numpy.ndarray) and numpy.issubdtype(o.dtype, numpy.integer):
        ret += "[" + ','.join(map(str, o.flatten().tolist())) + "]"
    elif isinstance(o, numpy.ndarray) and numpy.issubdtype(o.dtype, numpy.inexact):
        ret += "[" + ','.join(map(lambda x: '%.7g' % x, o.flatten().tolist())) + "]"
    else:
        raise TypeError("Unknown type '%s' for json serialization" % str(type(o)))
    return ret

def deccolor(c):
    return ( int(c[0] * 255) << 16  ) + ( int(c[1] * 255) << 8 ) + int(c[2] * 255)

class FACE_BITS:
    # https://github.com/mrdoob/three.js/wiki/JSON-Model-format-3.0
    
    NUM_BITS = 8
    
    TRIANGLE_QUAD = 0
    # define these since they aren't simple on/off
    TRIANGLE = 0
    QUAD = 1
    
    # rest of these are on/off
    ON = 1
    OFF = 0
    
    FACE_MATERIAL = 1
    FACE_UVS = 2
    FACE_VERTEX_UVS = 3
    FACE_NORMAL = 4
    FACE_VERTEX_NORMALS = 5
    FACE_COLOR = 6
    FACE_VERTEX_COLORS = 7

class ThreeJSDictGenerator(object):
    
    def __init__(self, mesh):
        self.mesh = mesh
        self.texture_placements = None
    
    def toDict(self):
        outdict = {}
        outdict['metadata'] = {'formatVersion': 3,
                               'type': 'scene'}
        outdict['defaults'] = {'bgcolor': numpy.array([0,0,0], dtype=int)}
        outdict['objects'] = self.getObjects()
        outdict['embeds'] = self.getEmbeds()
        outdict['geometries'] = {}
        for embed_name in outdict['embeds'].keys():
            outdict['geometries'][embed_name] = {'type': 'embedded_mesh', 'id': embed_name}
        
        outdict['materials'] = self.getMaterials()
        
        texture_placements = self.getTexturePlacements()
        outdict['textures'] = dict((cimg.id, {'url': outname})
                                   for cimg, outname in
                                   texture_placements.iteritems())
        
        return outdict
    
    def saveTo(self, filename):
        outputfile = open(filename, "w")
        
        outdict = self.toDict()
        outputfile.write(to_json(outdict))
        outputfile.close()
        
        self.saveTextures(filename)
    
    def getMaterials(self):
        materials = {}
        for material in self.mesh.materials:
            effect = material.effect
            
            attrs = {}
            if effect.shadingtype == 'lambert':
                attrs['type'] = 'MeshLambertMaterial'
            elif effect.shadingtype == 'phong' or effect.shadingtype == 'blinn':
                attrs['type'] = 'MeshPhongMaterial'
            else:
                attrs['type'] = 'MeshBasicMaterial'
                
            params = {}
            attrs['parameters'] = params
            
            color_mapping = [('diffuse', 'color'),
                             ('ambient', 'ambient'),
                             ('specular', 'specular')]
            for effect_attr, three_name in color_mapping:
                val = getattr(effect, effect_attr, None)
                if val is not None and not isinstance(val, collada.material.Map):
                    params[three_name] = deccolor(val)
            
            float_mapping = [('shininess', 'shininess'),
                             ('transparency', 'opacity')]
            for effect_attr, three_name in float_mapping:
                val = getattr(effect, effect_attr, None)
                if val is not None and not isinstance(val, collada.material.Map):
                    params[three_name] = val
            
            map_mapping = [('diffuse', 'map'),
                           ('ambient', 'mapAmbient'),
                           ('specular', 'mapSpecular'),
                           ('bump_map', 'mapNormal')]
            for effect_attr, three_name in map_mapping:
                val = getattr(effect, effect_attr, None)
                if isinstance(val, collada.material.Map):
                    params[three_name] = val.sampler.surface.image.id
            
            materials[material.id] = attrs
            
        return materials
    
    def getEmbeds(self):
        embeds = {}
        
        for geom in self.mesh.geometries:
            for prim_num, prim in enumerate(geom.primitives):
                if isinstance(prim, collada.polylist.Polylist):
                    prim = prim.triangleset()
                
                attrs = {}
                attrs["metadata"] = {"formatVersion": 3}
                attrs["scale"] = 1.0
                attrs["materials"] = []
                attrs["morphTargets"] = []
                attrs["colors"] = []
                attrs["edges"] = []
                
                attrs["vertices"] = prim.vertex if prim.vertex is not None else []
                attrs["normals"] = prim.normal if prim.normal is not None else []
                attrs["uvs"] = [texset for texset in prim.texcoordset]
                
                to_stack = [prim.vertex_index]
                type_bits = [0] * FACE_BITS.NUM_BITS
                type_bits[FACE_BITS.TRIANGLE_QUAD] = FACE_BITS.TRIANGLE
                if len(prim.texcoordset) > 0:
                    type_bits[FACE_BITS.FACE_VERTEX_UVS] = FACE_BITS.ON
                    to_stack.append(prim.texcoord_indexset[0])
                if prim.normal is not None:
                    type_bits[FACE_BITS.FACE_VERTEX_NORMALS] = FACE_BITS.ON
                    to_stack.append(prim.normal_index)
    
                type_code = int(''.join(map(str, reversed(type_bits))), 2)
                type_codes = numpy.empty((len(prim), 1), dtype=numpy.int32)
                type_codes[:] = type_code
                to_stack.insert(0, type_codes)
                
                stacked = numpy.hstack(to_stack)
                attrs["faces"] = stacked
                
                embeds["%s-primitive-%d" % (geom.id, prim_num)] = attrs
        
        return embeds
    
    def getObjects(self):
        objects = {}
        
        matrix = numpy.identity(4)
        print self.mesh.assetInfo.upaxis
        if self.mesh.assetInfo.upaxis == collada.asset.UP_AXIS.X_UP:
            r = collada.scene.RotateTransform(0,1,0,-90)
            matrix = r.matrix
        elif self.mesh.assetInfo.upaxis == collada.asset.UP_AXIS.Z_UP:
            r = collada.scene.RotateTransform(1,0,0,-90)
            matrix = r.matrix
        
        if self.mesh.scene is not None:
            for boundgeom in self.mesh.scene.objects('geometry'):
                for prim_num, boundprim in enumerate(boundgeom.primitives()):
                    attrs = {}
                    geom_name = "%s-primitive-%d" % (boundgeom.original.id, prim_num)
                    attrs['geometry'] = geom_name
                    attrs['materials'] = []
                    if boundprim.material is not None:
                        attrs['materials'].append(boundprim.material.id)
                    
                    attrs["matrix"] = numpy.dot(matrix, boundgeom.matrix)
                    attrs["visible"] = True
                        
                    objects[geom_name] = attrs
        
        return objects
    
    def getTexturePlacements(self):
        if self.texture_placements is not None:
            return self.texture_placements
        
        unique_filenames = set()
        cimage_name_map = {}
        
        for cimg in self.mesh.images:
            orig_path = cimg.path
            texname = os.path.basename(orig_path)
            
            basetexname, ext = os.path.splitext(texname)
            ct = 1
            while texname in unique_filenames:
                texname = basetexname + str(ct) + ext
                ct += 1
            
            unique_filenames.add(texname)
            cimage_name_map[cimg] = texname
            
        self.texture_placements = cimage_name_map
        return self.texture_placements
    
    def saveTextures(self, output_file):
        output_dir = os.path.dirname(output_file)
        for cimg in self.mesh.images:
            texname = self.texture_placements[cimg]
            texpath = os.path.join(output_dir, texname)
            f = open(texpath, 'wb')
            f.write(cimg.data)
            f.close()

def FilterGenerator():
    class ThreeJsSceneSaveFilter(SaveFilter):
        def __init__(self):
            super(ThreeJsSceneSaveFilter, self).__init__('save_threejs_scene', 'Saves a collada model in three.js scene format')

        def apply(self, mesh, filename):
            if os.path.exists(filename):
                raise FilterException("specified filename already exists")
            
            generator = ThreeJSDictGenerator(mesh)
            generator.saveTo(filename)
            
            return mesh

    return ThreeJsSceneSaveFilter()

from meshtool.filters import factory
factory.register(FilterGenerator().name, FilterGenerator)
