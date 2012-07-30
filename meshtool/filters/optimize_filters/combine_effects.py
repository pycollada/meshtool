from meshtool.filters.base_filters import OptimizationFilter

def combineEffects(mesh):
    effect_sets = []
    for e in mesh.effects:
        matched = False
        for s in effect_sets:
            if e.almostEqual(s[0]):
                s.append(e)
                matched = True
                break
        if not matched:
            effect_sets.append([e])
    
    for s in effect_sets:
        if len(s) <= 1:
            continue
        
        #keep the first one in the document
        to_keep = s.pop(0)
        
        #update all other materials referencing others to the first
        for other in s:
            for mat in mesh.materials:
                if mat.effect == other:
                    mat.effect = to_keep
            del mesh.effects[other.id]

def FilterGenerator():
    class CombineEffectsFilter(OptimizationFilter):
        def __init__(self):
            super(CombineEffectsFilter, self).__init__('combine_effects', 'Combines identical effects')
        def apply(self, mesh):
            combineEffects(mesh)
            return mesh
    return CombineEffectsFilter()
from meshtool.filters import factory
factory.register(FilterGenerator().name, FilterGenerator)