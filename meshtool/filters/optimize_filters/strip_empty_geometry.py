from meshtool.filters.base_filters import OptimizationFilter
import collada

def stripEmptyGeometry(mesh):
    to_delete = []
    geoms_to_delete = []
    for i, geom in enumerate(mesh.geometries):
        if len(geom.primitives) == 0:
            to_delete.append(i)
            geoms_to_delete.append(geom)
            
    for scene in mesh.scenes:
        nodes_to_check = []
        nodes_to_check.extend(scene.nodes)
        while len(nodes_to_check) > 0:
            curnode = nodes_to_check.pop()
            scene_nodes_to_delete = []
            for i, node in enumerate(curnode.children):
                if isinstance(node, collada.scene.Node):
                    nodes_to_check.append(node)
                elif isinstance(node, collada.scene.GeometryNode):
                    if node.geometry in geoms_to_delete:
                        scene_nodes_to_delete.append(i)
            scene_nodes_to_delete.sort(reverse=True)
            for i in scene_nodes_to_delete:
                del curnode.children[i]
    
    #TODO: also delete from any controllers referencing the geometry
            
    to_delete.sort(reverse=True)
    for i in to_delete:
        del mesh.geometries[i]

def FilterGenerator():
    class StripEmptyGeometryFilter(OptimizationFilter):
        def __init__(self):
            super(StripEmptyGeometryFilter, self).__init__('strip_empty_geometry', 'Strips any empty geometry from the document and removes them from any scenes')
        def apply(self, mesh):
            stripEmptyGeometry(mesh)
            return mesh
    return StripEmptyGeometryFilter()
from meshtool.filters import factory
factory.register(FilterGenerator().name, FilterGenerator)