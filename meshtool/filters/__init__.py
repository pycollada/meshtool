from base_filters import *
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
registerModule('filters.load_filters.load_collada')

#Op filters next
registerModule('filters.print_filters.print_textures')
registerModule('filters.panda_filters.viewer')
registerModule('filters.print_filters.print_json')

#Save filters last
registerModule('filters.panda_filters.save_screenshot')
registerModule('filters.panda_filters.save_rotate_screenshots')
registerModule('filters.save_filters.save_collada')
