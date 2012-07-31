import numpy
import collada
from meshtool.filters.panda_filters import pdae_utils
from meshtool.args import FileArgument, FilterArgument
from meshtool.filters.base_filters import SimplifyFilter, FilterException

def add_back_pm(mesh, pm_file, percent):
    refinements = pdae_utils.readPDAE(pm_file)
    
    num_to_load = len(refinements) * (percent / 100.0)
    num_to_load = min(num_to_load, len(refinements))
    num_to_load = int(round(num_to_load))
    if num_to_load <= 0:
        return mesh
    
    assert(len(mesh.geometries) == 1)
    geom = mesh.geometries[0]
    assert(len(geom.primitives) == 1)
    triset = geom.primitives[0]
    assert(isinstance(triset, collada.triangleset.TriangleSet))
    
    indices2stack = [triset.vertex_index.reshape(-1, 1), triset.normal_index.reshape(-1, 1), triset.texcoord_indexset[0].reshape(-1, 1)]
    stacked_indices = numpy.hstack(indices2stack).flatten().reshape((-1, 3))
    
    #index_map - maps each unique value back to a location in the original array it came from
    #   eg. stacked_indices[index_map] == unique_stacked_indices
    #inverse_map - maps original array locations to their location in the unique array
    #   e.g. unique_stacked_indices[inverse_map] == stacked_indices
    unique_stacked_indices, index_map, inverse_map = numpy.unique(stacked_indices.view([('',stacked_indices.dtype)]*stacked_indices.shape[1]), return_index=True, return_inverse=True)
    unique_stacked_indices = unique_stacked_indices.view(stacked_indices.dtype).reshape(-1,stacked_indices.shape[1])
    
    #unique returns as int64, so cast back
    index_map = numpy.cast['uint32'](index_map)
    inverse_map = numpy.cast['uint32'](inverse_map)
    
    #sort the index map to get a list of the index of the first time each value was encountered
    sorted_map = numpy.cast['uint32'](numpy.argsort(index_map))
    
    #since we're sorting the unique values, we have to map the inverse_map to the new index locations
    backwards_map = numpy.zeros_like(sorted_map)
    backwards_map[sorted_map] = numpy.arange(len(sorted_map), dtype=numpy.uint32)
    
    #now this is the new unique values and their indices
    unique_stacked_indices = unique_stacked_indices[sorted_map]
    inverse_map = backwards_map[inverse_map]

    alldata = [triset.vertex, triset.normal, triset.texcoordset[0]]
    data2stack = []
    for idx, data in enumerate(alldata):
        data2stack.append(data[unique_stacked_indices[:,idx]])
    unique_stacked_data = numpy.hstack(data2stack).flatten()

    num_added_verts = 0
    num_added_tris = 0
    for refinement_index in range(num_to_load):
        for op in refinements[refinement_index]:
            if op[0] == pdae_utils.PM_OP.VERTEX_ADDITION:
                num_added_verts += 1
            elif op[0] == pdae_utils.PM_OP.TRIANGLE_ADDITION:
                num_added_tris += 1
    
    cur_vertex = len(unique_stacked_data)
    cur_triangle = len(inverse_map)
    
    unique_stacked_data = numpy.append(unique_stacked_data, numpy.zeros(num_added_verts * 8, dtype=numpy.float32))
    inverse_map = numpy.append(inverse_map, numpy.zeros(num_added_tris * 3, dtype=numpy.uint32))
    
    for refinement_index in range(num_to_load):
        for operation in refinements[refinement_index]:
            vals = list(operation)
            op = vals.pop(0)
            if op == pdae_utils.PM_OP.VERTEX_ADDITION:
                vx, vy, vz, nx, ny, nz, s, t = map(float, vals)
                unique_stacked_data[cur_vertex:cur_vertex+8] = vx, vy, vz, nx, ny, nz, s, t
                cur_vertex += 8
            elif op == pdae_utils.PM_OP.TRIANGLE_ADDITION:
                v1, v2, v3 = map(int, vals)
                inverse_map[cur_triangle:cur_triangle+3] = v1, v2, v3
                cur_triangle += 3
            elif op == pdae_utils.PM_OP.INDEX_UPDATE:
                tindex, vindex = map(int, vals)
                inverse_map[tindex] = vindex
    
    oldgeom = geom
    mesh.geometries.pop(0)
    
    unique_stacked_data.shape = (-1, 8)
    vertex = numpy.copy(unique_stacked_data[:,0:3])
    normal = numpy.copy(unique_stacked_data[:,3:6])
    uvs = numpy.copy(unique_stacked_data[:,6:8])
    
    vert_src = collada.source.FloatSource("sander-verts-array", vertex, ('X', 'Y', 'Z'))
    normal_src = collada.source.FloatSource("sander-normals-array", normal, ('X', 'Y', 'Z'))
    uv_src = collada.source.FloatSource("sander-uv-array", uvs, ('S', 'T'))
    geom = collada.geometry.Geometry(mesh, geom.id, geom.name, [vert_src, normal_src, uv_src])
    
    input_list = collada.source.InputList()
    input_list.addInput(0, 'VERTEX', '#sander-verts-array')
    input_list.addInput(0, 'NORMAL', '#sander-normals-array')
    input_list.addInput(0, 'TEXCOORD', '#sander-uv-array')

    triset = geom.createTriangleSet(inverse_map, input_list, triset.material)
    geom.primitives.append(triset)
    mesh.geometries.append(geom)

    for scene in mesh.scenes:
        nodes_to_check = []
        nodes_to_check.extend(scene.nodes)
        while len(nodes_to_check) > 0:
            curnode = nodes_to_check.pop()
            for i, node in enumerate(curnode.children):
                if isinstance(node, collada.scene.Node):
                    nodes_to_check.append(node)
                elif isinstance(node, collada.scene.GeometryNode):
                    if node.geometry == oldgeom:
                        node.geometry = geom

    return mesh

def FilterGenerator():
    class AddBackPm(SimplifyFilter):
        def __init__(self):
            super(AddBackPm, self).__init__('add_back_pm', 'Adds back mesh data from a progressive PDAE file')
            self.arguments.append(FileArgument('pm_file', 'PDAE file to load from'))
            self.arguments.append(FilterArgument('percent', 'Percent of progressive file to add back'))
        def apply(self, mesh, pm_file, percent):
            try:
                pmin = open(pm_file, 'r')
            except IOError:
                raise FilterException("Invalid pm file")
            
            try:
                percent = float(percent)
            except ValueError:
                percent = None
            
            if percent is None or percent < 0.0:
                raise FilterException("Invalid percentage")
            
            mesh = add_back_pm(mesh, pmin, percent)
            return mesh

    return AddBackPm()
from meshtool.filters import factory
factory.register(FilterGenerator().name, FilterGenerator)
