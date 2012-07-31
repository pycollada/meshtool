from meshtool.filters.base_filters import OptimizationFilter

import collada
import numpy

def getSemanticCount(inp):
    inpct = {}
    for srctup in inp:
        if srctup[1] not in inpct:
            inpct[srctup[1]] = 0
        inpct[srctup[1]] += 1
    return inpct

def canMergeInputs(inp1, inp2):
    inp1ct = getSemanticCount(inp1)
    inp2ct = getSemanticCount(inp2)
    return inp1ct == inp2ct

def combinePrimitives(mesh):
    
    # first lets make a list of all the times each geometry is instantiated
    all_instantiations = {}
    for geometry in mesh.geometries:
        all_instantiations[geometry.id] = []
    for scene in mesh.scenes:
        nodes_to_check = []
        nodes_to_check.extend(scene.nodes)
        while len(nodes_to_check) > 0:
            curnode = nodes_to_check.pop()
            for node in curnode.children:
                if isinstance(node, collada.scene.Node):
                    nodes_to_check.append(node)
                elif isinstance(node, collada.scene.GeometryNode):
                    all_instantiations[node.geometry.id].append(node)
                elif isinstance(node, collada.scene.ControllerNode):
                    all_instantiations[node.controller.geometry.id].append(node)
    
    for geometry in mesh.geometries:
        
        instantiations = all_instantiations[geometry.id]
        
        # now we will group the primitives into sets of primitives that get
        # bound to the same material for each instantiation and have aligned
        # inputs
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
                if s['bindings'] == bindings and canMergeInputs(input_list.getList(), s['inputs'].getList()):
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
            # and that have similar aligned inputs, so we can combine their source
            # arrays into a single source and the primitives into a single primitive
            
            semantic_counts = getSemanticCount(s['inputs'].getList())
            source_count = sum(semantic_counts.itervalues())

            src_arrays = []
            for member in s['members']:
                i = 0
                for semantic in collada.source.InputList.semantics:
                    for offset, semantic, srcid, setid, srcobj in member.sources[semantic]:
                        selected_data = srcobj.data[member.index[:,:,offset]]
                        if len(src_arrays) == i:
                            src_arrays.append((semantic, [selected_data], srcobj.components))
                        else:
                            src_arrays[i][1].append(selected_data)
                        i += 1

            concat_arrays = {}
            for semantic, src_list, components in src_arrays:
                if semantic not in concat_arrays:
                    concat_arrays[semantic] = []
                all_concat = numpy.concatenate(src_list)
                all_concat.shape = -1
                concat_arrays[semantic].append((components, all_concat))
            src_arrays = None
            
            inpl = collada.source.InputList()
            index_arrays = []
            offset = 0
            for semantic in collada.source.InputList.semantics:
                if semantic in concat_arrays:
                    for setid in range(len(concat_arrays[semantic])):
                        components, concat_array = concat_arrays[semantic][setid]
                        
                        base_source_name = "%s-%s-%s" % (geometry.id, semantic.lower(), setid)
                        source_name = base_source_name
                        ct = 0
                        while source_name in geometry.sourceById:
                            source_name = '%s-%d' % (base_source_name, ct)
                            ct += 1
                            
                        new_src = collada.source.FloatSource(source_name, concat_array, components)
                        geometry.sourceById[source_name] = new_src
                        
                        inpl.addInput(offset, semantic, '#%s' % source_name, setid)
                        index_arrays.append(numpy.arange(len(new_src)))
                        
                        offset += 1

            combined_index = numpy.dstack(index_arrays).flatten()
            material_symbol = s['members'][0].material
            combined_triset = geometry.createTriangleSet(combined_index, inpl, material_symbol)

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
    class CombinePrimitivesFilter(OptimizationFilter):
        def __init__(self):
            super(CombinePrimitivesFilter, self).__init__('combine_primitives', 'Combines primitives within a geometry if they have the same sources and scene material mapping (triangle sets only)')
        def apply(self, mesh):
            combinePrimitives(mesh)
            return mesh
    return CombinePrimitivesFilter()
from meshtool.filters import factory
factory.register(FilterGenerator().name, FilterGenerator)