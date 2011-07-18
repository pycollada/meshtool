from meshtool.args import *
from meshtool.filters.base_filters import *
from meshtool.filters import factory

def mediumOptimizations(mesh):
    optimize_filters = [
                        'triangulate',
                        'generate_normals',
                        'combine_effects',
                        'combine_materials',
                        'combine_primitives',
                        'optimize_sources',
                        'strip_unused_sources',
                        'optimize_textures'
                        ]
    
    for filter in optimize_filters:
        inst = factory.getInstance(filter)
        mesh = inst.apply(mesh)
        
    return mesh


def FilterGenerator():
    class MediumOptimizationsFilter(OpFilter):
        def __init__(self):
            super(MediumOptimizationsFilter, self).__init__('medium_optimizations', 
                    'A meta filter that runs a safe, medium-level of optimizations. Performs these filters in this order: ' +
                     'triangulate, generate_normals, combine_effects, combine_materials, combine_primitives, optimize_sources, ' +
                     'strip_unused_sources, optimize_textures')
        def apply(self, mesh):
            return mediumOptimizations(mesh)
    return MediumOptimizationsFilter()
from meshtool.filters import factory
factory.register(FilterGenerator().name, FilterGenerator)