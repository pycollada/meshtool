from meshtool.filters.base_filters import PrintFilter
import collada

def printInstances(mesh):
    num_instances = 0
    if mesh.scene:
        nodes_to_check = []
        nodes_to_check.extend(mesh.scene.nodes)
        while len(nodes_to_check) > 0:
            curnode = nodes_to_check.pop()
            for node in curnode.children:
                if isinstance(node, collada.scene.Node):
                    nodes_to_check.append(node)
                elif isinstance(node, collada.scene.GeometryNode) or \
                        isinstance(node, collada.scene.ControllerNode):
                    num_instances += 1
                    print node
    print 'Total geometries instantiated in default scene: %d' % num_instances

def FilterGenerator():
    class PrintInstancesFilter(PrintFilter):
        def __init__(self):
            super(PrintInstancesFilter, self).__init__('print_instances', 'Prints geometry instances from the default scene')
        def apply(self, mesh):
            printInstances(mesh)
            return mesh
    return PrintInstancesFilter()
from meshtool.filters import factory
factory.register(FilterGenerator().name, FilterGenerator)