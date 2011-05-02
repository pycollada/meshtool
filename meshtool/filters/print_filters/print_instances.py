from meshtool.args import *
from meshtool.filters.base_filters import *
import collada
import copy

def printNode(node, previous, instances):
    if isinstance(node, collada.scene.Node):
        withme = copy.copy(previous)
        withme.append(node)
        for child in node.children:
            printNode(child, withme, instances)
    else:
        indent = ''
        for prev in previous:
            print indent + str(prev)
            indent += ' '
        print indent + str(node)
        instances.append(node)

def printInstances(mesh):
    instances = []
    if mesh.scene is not None:
        for node in mesh.scene.nodes:
            printNode(node, [], instances)
    print
    print 'Total instances in scene: %d' % len(instances)

def FilterGenerator():
    class PrintInstancesFilter(OpFilter):
        def __init__(self):
            super(PrintInstancesFilter, self).__init__('print_instances', 'Prints geometry instances from the default scene')
        def apply(self, mesh):
            printInstances(mesh)
            return mesh
    return PrintInstancesFilter()
