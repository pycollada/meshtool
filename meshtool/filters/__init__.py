from meshtool.filters.base_filters import *
import sys

factory = FilterFactory()

def registerModule(mod):
    try: m = __import__(mod)
    except ImportError: return
    
    if mod not in sys.modules:
        return
    m = sys.modules[mod]
    factory.register(m.FilterGenerator().name, m.FilterGenerator)

#Load filters first
registerModule('meshtool.filters.load_filters.load_collada')

#Op filters next
registerModule('meshtool.filters.print_filters.print_textures')
registerModule('meshtool.filters.panda_filters.viewer')
registerModule('meshtool.filters.print_filters.print_json')
registerModule('meshtool.filters.print_filters.print_info')
registerModule('meshtool.filters.print_filters.print_instances')
registerModule('meshtool.filters.optimize_filters.combine_effects')
registerModule('meshtool.filters.optimize_filters.combine_materials')
registerModule('meshtool.filters.optimize_filters.combine_primitives')
registerModule('meshtool.filters.optimize_filters.strip_lines')
registerModule('meshtool.filters.optimize_filters.strip_empty_geometry')
registerModule('meshtool.filters.optimize_filters.triangulate')
registerModule('meshtool.filters.optimize_filters.generate_normals')
registerModule('meshtool.filters.optimize_filters.save_mipmaps')
registerModule('meshtool.filters.atlas_filters.make_atlases')
registerModule('meshtool.filters.simplify_filters.simplify')
registerModule('meshtool.filters.simplify_filters.load_pm')

#Save filters last
registerModule('meshtool.filters.panda_filters.save_screenshot')
registerModule('meshtool.filters.panda_filters.save_rotate_screenshots')
registerModule('meshtool.filters.save_filters.save_collada')
registerModule('meshtool.filters.save_filters.save_collada_zip')
