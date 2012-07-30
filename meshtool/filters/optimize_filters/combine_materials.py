from meshtool.filters.base_filters import OptimizationFilter

import collada

def combineMaterials(mesh):
    material_sets = []
    for m in mesh.materials:
        matched = False
        for s in material_sets:
            if s[0].effect == m.effect:
                s.append(m)
                matched = True
                break
        if not matched:
            material_sets.append([m])
    
    for s in material_sets:
        if len(s) <= 1:
            continue
        
        #keep the first one in the document
        to_keep = s.pop(0)
        
        #update all other material nodes in the scene to refer to the first one
        for scene in mesh.scenes:
            nodes_to_check = []
            nodes_to_check.extend(scene.nodes)
            while len(nodes_to_check) > 0:
                curnode = nodes_to_check.pop()
                for node in curnode.children:
                    if isinstance(node, collada.scene.Node):
                        nodes_to_check.append(node)
                    elif isinstance(node, collada.scene.GeometryNode):
                        for matnode in node.materials:
                            if matnode.target in s:
                                matnode.target = to_keep

        #delete all of the other materials from the mesh
        for material in s:
            del mesh.materials[material.id]

def FilterGenerator():
    class CombineMaterialsFilter(OptimizationFilter):
        def __init__(self):
            super(CombineMaterialsFilter, self).__init__('combine_materials', 'Combines identical materials')
        def apply(self, mesh):
            combineMaterials(mesh)
            return mesh
    return CombineMaterialsFilter()
from meshtool.filters import factory
factory.register(FilterGenerator().name, FilterGenerator)