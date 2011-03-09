from base_filters import *

factory = FilterFactory()
def registerModule(m):
    factory.register(m.FilterGenerator().name, m.FilterGenerator)

#Load filters first
import load_filters.load_collada as m
registerModule(m)

#Op filters next
import print_filters.print_textures as m
registerModule(m)

#Save filters last
try:
    import save_filters.save_screenshot as m
    registerModule(m)
except: pass

import save_filters.save_collada as m
registerModule(m)
