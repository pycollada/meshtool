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

#Save filters last
registerModule('meshtool.filters.panda_filters.save_screenshot')
registerModule('meshtool.filters.panda_filters.save_rotate_screenshots')
registerModule('meshtool.filters.save_filters.save_collada')
