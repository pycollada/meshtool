from meshtool.args import *
from meshtool.filters.base_filters import *

import collada
import numpy

def combinePrimitives(mesh):
    for geometry in mesh.geometries:
        
        # first lets make a list of all the times the geometry is instantiated
        instantiations = []
        for scene in mesh.scenes:
            nodes_to_check = []
            nodes_to_check.extend(scene.nodes)
            while len(nodes_to_check) > 0:
                curnode = nodes_to_check.pop()
                for node in curnode.children:
                    if isinstance(node, collada.scene.Node):
                        nodes_to_check.append(node)
                    elif isinstance(node, collada.scene.GeometryNode):
                        if node.geometry == geometry:
                            instantiations.append(node)
        
        # now we will group the primitives into sets of primitives that get
        # bound to the same material for each instantiation
        primitive_sets = []
        for primitive in geometry.primitives:
            if not isinstance(primitive, collada.triangleset.TriangleSet):
                continue
            
            material_symbol = primitive.material
            bindings = []
            for geomnode in instantiations:
                for matnode in geomnode.materials:
                    if matnode.symbol == material_symbol:
                        bindings.append(matnode.target)
            
            input_list = primitive.getInputList()
            
            matched = False
            for s in primitive_sets:
                if s['bindings'] == bindings and input_list.getList() == s['inputs'].getList():
                    matched = True
                    s['members'].append(primitive)
            if not matched:
                primitive_sets.append({'bindings': bindings,
                                       'inputs': input_list,
                                       'members': [primitive]})
                
        for s in primitive_sets:
            if len(s['members']) == 1:
                continue
            
            # okay, now we have a list of primitives within the geometry that are all
            # being bound to the same material whenever the geometry is instantiated
            # so we can combine them into a single primitive
            
            input_list = s['inputs'].getList()
            max_offset = 0
            for offset, semantic, source, set in input_list:
                if offset > max_offset:
                    max_offset = offset
                 
            index_arrays = []
            for i in range(max_offset+1):
                index_arrays.append([])
                
            for member in s['members']:
                for i in range(max_offset+1):
                    index_arrays[i].append(member.index[:,:,i])
                
            concat_arrays = []
            for i in range(max_offset+1):
                concat_arrays.append(numpy.concatenate(index_arrays[i]))
                
            combined_index = numpy.dstack(concat_arrays).flatten()
            material_symbol = s['members'][0].material
            combined_triset = geometry.createTriangleSet(combined_index, s['inputs'], material_symbol)
            
            #now find each primitive and delete it from the geometry
            todel = []
            for i, primitive in enumerate(geometry.primitives):
                if primitive in s['members']:
                    todel.append(i)
            todel.sort()
            todel.reverse()
            for i in todel:
                del geometry.primitives[i]
                
            geometry.primitives.append(combined_triset)
        
        # now lets go through the instantiations and delete any material nodes that
        # reference symbols that no longer exist in the geometry
        for geomnode in instantiations:
            todel = []
            all_symbols = [primitive.material for primitive in geometry.primitives]
            for i, matnode in enumerate(geomnode.materials):
                if matnode.symbol not in all_symbols:
                    todel.append(i)
            todel.sort()
            todel.reverse()
            for i in todel:
                del geomnode.materials[i]
            
    return mesh

def FilterGenerator():
    class CombinePrimitivesFilter(OpFilter):
        def __init__(self):
            super(CombinePrimitivesFilter, self).__init__('combine_primitives', 'Combines primitives within a geometry if they have the same sources and scene material mapping (triangle sets only)')
        def apply(self, mesh):
            combinePrimitives(mesh)
            return mesh
    return CombinePrimitivesFilter()
