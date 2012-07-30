from meshtool.filters.base_filters import PrintFilter
import collada

def printNode(node, depth):
    indent = '  '
    print indent*depth + str(node)
    if isinstance(node, collada.scene.Node):
        for child in node.children:
            printNode(child, depth+1)
        print indent*depth + '</Node>'

def printScene(mesh):
    if mesh.scene is not None:
        for node in mesh.scene.nodes:
            printNode(node, 0)

def FilterGenerator():
    class PrintSceneFilter(PrintFilter):
        def __init__(self):
            super(PrintSceneFilter, self).__init__('print_scene', 'Prints the default scene tree')
        def apply(self, mesh):
            printScene(mesh)
            return mesh
    return PrintSceneFilter()
from meshtool.filters import factory
factory.register(FilterGenerator().name, FilterGenerator)