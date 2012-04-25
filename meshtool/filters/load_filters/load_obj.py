import collections
import os
import sys
from meshtool.util import to_unicode, slugify
from StringIO import StringIO
import string
import posixpath

import numpy
import collada

from meshtool.filters.base_filters import FilterException, LoadFilter
from meshtool.filters import factory

class NameUniqifier(object):
    def __init__(self):
        self.names = []
    
    def name(self, name):
        basename = slugify(name)
        if basename[0] in string.digits:
            basename = "x" + basename
        name = basename
        ct = 0
        while name in self.names:
            name = basename + "-" + str(ct)
            ct += 1
        return name

class FACEMODE:
    UNKNOWN = 0
    V = 1
    VT = 2
    VTN = 3
    VN = 4

def detectFaceStyle(data):
    num_slashes = data.count('/')
    if num_slashes == 0:
        return FACEMODE.V
    elif num_slashes == 1:
        return FACEMODE.VT
    elif num_slashes == 2:
        if '//' in data:
            return FACEMODE.VN
        return FACEMODE.VTN

class ObjGroup(object):
    def __init__(self, name):
        self.name = name
        self.material = None
        self.face_indices = []
        self.face_lengths = []
        self.line_indices = []
        self.face_mode = FACEMODE.UNKNOWN
    
    def empty(self):
        if len(self.face_lengths) > 0 or len(self.line_indices) > 0:
            return False
        return True
    
    def __str__(self):
        return "<ObjGroup '%s' %d faces>" % (self.name, len(self.face_lengths))
    def __repr__(self):
        return str(self)

def decode_mtl_single(line):
    try:
        color = float(line)
    except ValueError:
        return None
    
    if color > 1.0:
        color /= 255.0
    if color > 1.0:
        return None
    
    return color

def decode_mtl_color(line):
    if 'spectral' in line or 'xyz' in line:
        # TODO: handle other color formats, xyz should be easy but probably not spectral
        return None
    
    # if can't convert to floats, give up
    try:
        color_values = map(float, line.split())
    except ValueError:
        return None
    
    color = None
    if len(color_values) == 1:
        # a single color means duplicate that color for all three rgb
        color = (color_values[0], color_values[0], color_values[0])
    elif len(color_values) == 3 or len(color_values) == 4:
        color = color_values
    
    # if not 1, 3, or 4, give up    
    if color is None:
        return None
    
    # if any color values are greater than 1.0, assume they probably used 0-255 range
    if any(c > 1.0 for c in color):
        color = (c / 255.0 for c in color)
        
    # if any color values are still greater than 1.0, give up
    if any(c > 1.0 for c in color):
        return None
    
    # if not a 4-tuple, add transparency to the end
    if len(color) != 4:
        color += (1.0,)
        
    # if somehow, it's still not a 4-tuple, give up (it should be)
    if len(color) != 4:
        return None
    
    return tuple(color)

def decode_mtl_texture(line, effect, aux_file_loader):
    texture_data = aux_file_loader(line)
    if texture_data is None:
        return (None, None)
    texture_slug = slugify(posixpath.splitext(line)[0])
    texture_path = texture_slug + posixpath.splitext(line)[1]
    cimage = collada.material.CImage(texture_slug, "./%s" % texture_path)
    cimage.data = texture_data
    surface = collada.material.Surface(texture_slug + "-surface", cimage)
    sampler = collada.material.Sampler2D(texture_slug + "-sampler", surface)
    _map = collada.material.Map(sampler, "TEX0")
    effect.params.append(surface)
    effect.params.append(sampler)
    return (cimage, _map)

def loadMaterialLib(data, namer, aux_file_loader=None):
    """Load an MTL file
    
    :param data: A binary string containing the mtl file
    :param namer: Should be an instance of :class:`ObjGroup`, used to generate unique
                  names for materials in the file, in case of duplicates or invalid
                  names containing funny characters (spaces, etc)
    :param aux_file_loader: Should be a callable function that takes one parameter.
                            The parameter will be a string containing an auxiliary
                            file that needs to be found, in this case usually a .mtl
                            file or a texture file.
    
    :returns: a `dict` containing 'material_map', 'images', and 'effects'
    """
    
    # maps MTL illumination types to collada shading types
    # note that 0,1,2,3 are mostly correct, but 4-10 have no
    # direct mapping to collada, so blinn is just a standin
    illumination_map = collections.defaultdict(lambda: 'blinn',
                                               **{0: 'constant',
                                                  1:'lambert'})
    
    cimages = []
    effects = []
    current_effect = collada.material.Effect(' empty ', [], 'blinn')
    
    file_like = StringIO(to_unicode(data))
    for line in file_like:
        line = line.strip()
        
        # ignore blank lines and comments
        if len(line) == 0 or line.startswith('#'):
            continue

        # split off the first non-whitespace token and ignore the line if there isn't > 1 token
        splitup = line.split(None, 1)
        if len(splitup) != 2:
            continue
        command, line = splitup
        
        if command == 'newmtl':
            if current_effect.id == ' empty ':
                current_effect.id = namer.name(line)
                continue
            
            effects.append(current_effect)
            current_effect = collada.material.Effect(namer.name(line),
                                                     [],
                                                     'blinn')
        
        elif command == 'illum':
            illum_num = None
            try:
                illum_num = int(line)
            except ValueError:
                pass
            
            current_effect.shadingtype = illumination_map[illum_num]
        
        elif command == 'Kd':
            color = decode_mtl_color(line)
            if color is not None:
                current_effect.diffuse = color
        elif command == 'Ka':
            color = decode_mtl_color(line)
            if color is not None:
                current_effect.ambient = color
        elif command == 'Ke':
            color = decode_mtl_color(line)
            if color is not None:
                current_effect.emission = color
        elif command == 'Ks':
            color = decode_mtl_color(line)
            if color is not None:
                current_effect.specular = color
        
        elif command == 'd' or command == 'Tr':
            color = decode_mtl_single(line)
            if color is not None:
                current_effect.transparency = color
        elif command == 'Ns':
            color = decode_mtl_single(line)
            if color is not None:
                current_effect.shininess = color
        
        elif command == 'map_Kd':
            cimg, texmap = decode_mtl_texture(line, current_effect, aux_file_loader)
            if texmap is not None:
                current_effect.diffuse = texmap
                cimages.append(cimg)
        elif command == 'map_Ka':
            cimg, texmap = decode_mtl_texture(line, current_effect, aux_file_loader)
            if texmap is not None:
                current_effect.ambient = texmap
                cimages.append(cimg)
        elif command == 'map_Ks':
            cimg, texmap = decode_mtl_texture(line, current_effect, aux_file_loader)
            if texmap is not None:
                current_effect.specular = texmap
                cimages.append(cimg)
        elif command == 'map_bump' or command == 'bump':
            cimg, texmap = decode_mtl_texture(line, current_effect, aux_file_loader)
            if texmap is not None:
                current_effect.bumpmap = texmap
                cimages.append(cimg)
        
        else:
            print 'MISSING MTL LINE', command, line
        
    if current_effect.id != ' empty ':
        effects.append(current_effect)
    
    material_map = {}
    for effect in effects:
        material_id = namer.name(effect.id + '-material')
        material = collada.material.Material(material_id, material_id, effect)
        material_map[effect.id] = material
    
    return {'material_map': material_map,
            'images': cimages,
            'effects': effects}

def loadOBJ(data, aux_file_loader=None, validate_output=False):
    """Loads an OBJ file
    
    :param data: A binary data string containing the OBJ file
    :param aux_file_loader: Should be a callable function that takes one parameter.
                            The parameter will be a string containing an auxiliary
                            file that needs to be found, in this case usually a .mtl
                            file or a texture file.
    
    :returns: An instance of :class:`collada.Collada` or None if could not be loaded
    """
    
    mesh = collada.Collada(validate_output=validate_output)
    namer = NameUniqifier()
    material_map = {}
    cimages = []
    materialNamer = NameUniqifier()
    
    vertices = []
    normals = []
    texcoords = []
    
    groups = []
    group = ObjGroup(namer.name("default"))
    geometry_name = namer.name("convertedobjgeometry")
    
    file_like = StringIO(to_unicode(data))
    for line in file_like:
        line = line.strip()
        
        # ignore blank lines and comments
        if len(line) == 0 or line.startswith('#'):
            continue
        
        # split off the first non-whitespace token and ignore the line if there isn't > 1 token
        splitup = line.split(None, 1)
        if len(splitup) != 2:
            continue
        command, line = splitup
        
        if command == 'v':
            line_tokens = line.split()
            vertices.extend(line_tokens[:3])
            
        elif command == 'vn':
            line_tokens = line.split()
            normals.extend(line_tokens[:3])
           
        elif command == 'vt':
            line_tokens = line.split()
            texcoords.extend(line_tokens[:2])
            
        # TODO: other vertex data statements
        # vp
        # cstype
        # deg
        # bmat
        # step
            
        elif command == 'f':
            faces = line.split()
            
            if group.face_mode == FACEMODE.UNKNOWN:
                group.face_mode = detectFaceStyle(faces[0])
                if group.face_mode is None:
                    sys.stderr.write("Error: could not detect face type for line '%s'" % line)
                    return
            
            group.face_lengths.append(len(faces))
            
            # Don't decode the faces here because the / separators have to be parsed out
            # and this is very slow to do one at a time. Instead, just append to a list
            # which is much faster than appending to a string, and it will get joined and
            # parsed later
            group.face_indices.append(line)
        
        elif command == 'l':
            faces = line.split()
            
            if group.face_mode == FACEMODE.UNKNOWN:
                group.face_mode = detectFaceStyle(faces[0])
                if group.face_mode is None:
                    sys.stderr.write("Error: could not detect face type for line '%s'" % line)
                    return
            
            # COLLADA defines lines as a pair of points, so the index values "1 2 3 4" would
            # refer to *two* lines, one between 1 and 2 and one between 3 and 4. OBJ defines
            # lines as continous, so it would be three lines: 1-2, 2-3, 3-4. This duplicates
            # the points to get pairs for COLLADA. This is not very efficient, but not sure
            # of a faster way to do this and I've never seen any files with a huge number of
            # lines in it anyway.
            line = faces[0] + " " + faces[1]
            prev = faces[1]
            for cur in faces[2:]:
                line += " " + prev + " " + cur
                prev = cur
            group.line_indices.append(line)
        
        elif command == 'p':
            faces = line.split()
            
            if group.face_mode == FACEMODE.UNKNOWN:
                group.face_mode = detectFaceStyle(faces[0])
                if group.face_mode is None:
                    sys.stderr.write("Error: could not detect face type for line '%s'" % line)
                    return
                
            # COLLADA does not have points, so this converts a point to a line with two
            # identical endpoints
            line = " ".join(f + " " + f for f in faces)
            group.line_indices.append(line)
        
        # TODO: other elements
        # curv
        # curv2
        # surf
        
        elif command == 'g':
            if group.empty():
                # first group without any previous data, so just set name
                group.name = namer.name(line)
                continue
            
            # end of previous group and start of new group
            groups.append(group)
            group = ObjGroup(namer.name(line))
        
        elif command == 's':
            # there is no way to map shading groups into collada
            continue
        
        elif command == 'o':
            geometry_name = namer.name(line)
        
        # TODO: grouping info
        # mg
        
        # TODO: Free-form curve/surface body statements
        # parm
        # trim
        # hole
        # scrv
        # sp
        # end
        # con
        
        elif command == 'mtllib':
            mtl_file = None
            if aux_file_loader is not None:
                mtl_file = aux_file_loader(line)
            if mtl_file is not None:
                material_data = loadMaterialLib(mtl_file, namer=materialNamer, aux_file_loader=aux_file_loader)
                material_map.update(material_data['material_map'])
                cimages.extend(material_data['images'])
            
        elif command == 'usemtl':
            group.material = slugify(line)
        
        # TODO: display and render attributes
        # bevel
        # c_interp
        # d_interp
        # lod
        # shadow_obj
        # trace_obj
        # ctech
        # stech
        
        else:
            print '  MISSING LINE: %s %s' % (command, line)
    
    # done, append last group
    if not group.empty():
        groups.append(group)
    
    for material in material_map.values():
        mesh.effects.append(material.effect)
        mesh.materials.append(material)
    for cimg in cimages:
        mesh.images.append(cimg)
    
    vertices = numpy.array(vertices, dtype=numpy.float32).reshape(-1, 3)
    normals = numpy.array(normals, dtype=numpy.float32).reshape(-1, 3)
    texcoords = numpy.array(texcoords, dtype=numpy.float32).reshape(-1, 2)
    
    sources = []
    # all modes have vertex source
    sources.append(collada.source.FloatSource("obj-vertex-source", vertices, ('X', 'Y', 'Z')))
    if len(normals) > 0:
        sources.append(collada.source.FloatSource("obj-normal-source", normals, ('X', 'Y', 'Z')))
    if len(texcoords) > 0:
        sources.append(collada.source.FloatSource("obj-uv-source", texcoords, ('S', 'T')))
    
    geom = collada.geometry.Geometry(mesh, geometry_name, geometry_name, sources)
    
    materials_mapped = set()
    for group in groups:
        input_list = collada.source.InputList()
        input_list.addInput(0, 'VERTEX', "#obj-vertex-source")
        if group.face_mode == FACEMODE.VN:
            input_list.addInput(1, 'NORMAL', '#obj-normal-source')
        elif group.face_mode == FACEMODE.VT:
            input_list.addInput(1, 'TEXCOORD', '#obj-uv-source')
        elif group.face_mode == FACEMODE.VTN:
            input_list.addInput(1, 'TEXCOORD', '#obj-uv-source')
            input_list.addInput(2, 'NORMAL', '#obj-normal-source')
        
        if len(group.face_lengths) > 0:
            face_lengths = numpy.array(group.face_lengths, dtype=numpy.int32)
    
            # First, join the individual face lines together, separated by spaces. Then,        
            # just replace 1/2/3 and 1//3 with "1 2 3" and "1  3", as numpy.fromstring can
            # handle any whitespace it's given, similar to python's split(). Concatenating
            # together this way is much faster than parsing the numbers in python - let
            # numpy do it. Note that sep=" " is actually misleading - it handles tabs and
            # other whitespace also
            group.face_indices = (" ".join(group.face_indices)).replace("/", " ")
            face_indices = numpy.fromstring(group.face_indices, dtype=numpy.int32, sep=" ")
            
            # obj indices start at 1, while collada start at 0
            face_indices -= 1
            
            polylist = geom.createPolylist(face_indices, face_lengths, input_list, group.material or namer.name("nullmaterial"))
            geom.primitives.append(polylist)
            
        if len(group.line_indices) > 0:
            group.line_indices = (" ".join(group.line_indices)).replace("/", " ")
            line_indices = numpy.fromstring(group.line_indices, dtype=numpy.int32, sep=" ")
            line_indices -= 1
            lineset = geom.createLineSet(line_indices, input_list, group.material or namer.name("nullmaterial"))
            geom.primitives.append(lineset)
        
        if group.material in material_map:
            materials_mapped.add(group.material)
    
    mesh.geometries.append(geom)
    
    matnodes = []
    for matref in materials_mapped:
        matnode = collada.scene.MaterialNode(matref, material_map[matref], inputs=[('TEX0', 'TEXCOORD', '0')])
        matnodes.append(matnode)
    geomnode = collada.scene.GeometryNode(geom, matnodes)
    node = collada.scene.Node(namer.name("node"), children=[geomnode])
    myscene = collada.scene.Scene(namer.name("scene"), [node])
    mesh.scenes.append(myscene)
    mesh.scene = myscene
    
    return mesh

def filepath_loader(obj_filename):
    obj_dir = os.path.dirname(obj_filename)
    
    def aux_loader(auxpath):
        auxloc = os.path.normpath(os.path.join(obj_dir, auxpath))
        if not os.path.isfile(auxloc):
            # try with replacing backslashes
            auxpath = auxpath.replace('\\', '/')
            auxloc = os.path.normpath(os.path.join(obj_dir, auxpath))
        if not os.path.isfile(auxloc):
            # try turning absolute paths into relative by stripping out leading part
            if auxpath.startswith('/'):
                auxpath = auxpath[1:]
                auxloc = os.path.normpath(os.path.join(obj_dir, auxpath))
        if os.path.isfile(auxloc):
            f = open(auxloc, 'rb')
            return f.read()
        return None
    
    return aux_loader

class OBJLoadFilter(LoadFilter):
    def __init__(self):
        super(OBJLoadFilter, self).__init__('load_obj', 'Loads a Wavefront OBJ file')
    
    def apply(self, filename):
        if not os.path.isfile(filename):
            raise FilterException("argument is not a valid file")
        
        col = loadOBJ(open(filename, 'rb').read(),
                      aux_file_loader=filepath_loader(filename))
            
        return col    

def FilterGenerator():
    return OBJLoadFilter()

factory.register(FilterGenerator().name, FilterGenerator)
