from meshtool.args import *
from meshtool.filters.base_filters import *
from meshtool.filters import factory

def fullOptimizations(mesh):
    optimize_filters = [
                        'triangulate',
                        'generate_normals',
                        'combine_effects',
                        'combine_materials',
                        'combine_primitives',
                        'adjust_texcoords',
                        'optimize_textures',
                        'split_triangle_texcoords',
                        'normalize_indices',
                        'make_atlases',
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
    class FullOptimizationsFilter(OpFilter):
        def __init__(self):
            super(FullOptimizationsFilter, self).__init__('full_optimizations', 
                    'A meta filter that runs all optimizations. Performs these filters in this order: ' +
                    'triangulate, generate_normals, combine_effects, combine_materials, combine_primitives, ' + 
                    'adjust_texcoords, optimize_textures, split_triangle_texcoords, normalize_indices, ' + 
                    'make_atlases, combine_effects, combine_materials, combine_primitives, optimize_sources' + 
                    'strip_unused_sources, optimize_textures')
        def apply(self, mesh):
            return fullOptimizations(mesh)
    return FullOptimizationsFilter()
from meshtool.filters import factory
factory.register(FilterGenerator().name, FilterGenerator)