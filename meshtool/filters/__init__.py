from meshtool.filters.base_filters import *
import sys

factory = FilterFactory()

#Load filters first
try: import meshtool.filters.load_filters.load_collada
except ImportError: pass
try: import meshtool.filters.load_filters.load_obj
except ImportError: pass

#Print filters
try: import meshtool.filters.print_filters.print_textures
except ImportError: pass
try: import meshtool.filters.print_filters.print_json
except ImportError: pass
try: import meshtool.filters.print_filters.print_info
except ImportError: pass
try: import meshtool.filters.print_filters.print_instances
except ImportError: pass
try: import meshtool.filters.print_filters.print_scene
except ImportError: pass
try: import meshtool.filters.print_filters.print_render_info
except ImportError: pass

#Viewer
try: import meshtool.filters.panda_filters.viewer
except ImportError: pass
try: import meshtool.filters.panda_filters.collada_viewer
except ImportError: pass
try: import meshtool.filters.panda_filters.pm_viewer
except ImportError: pass

#Optimizations
try: import meshtool.filters.optimize_filters.combine_effects
except ImportError: pass
try: import meshtool.filters.optimize_filters.combine_materials
except ImportError: pass
try: import meshtool.filters.optimize_filters.combine_primitives
except ImportError: pass
try: import meshtool.filters.optimize_filters.strip_lines
except ImportError: pass
try: import meshtool.filters.optimize_filters.strip_empty_geometry
except ImportError: pass
try: import meshtool.filters.optimize_filters.strip_unused_sources
except ImportError: pass
try: import meshtool.filters.optimize_filters.triangulate
except ImportError: pass
try: import meshtool.filters.optimize_filters.generate_normals
except ImportError: pass
try: import meshtool.filters.optimize_filters.save_mipmaps
except ImportError: pass
try: import meshtool.filters.optimize_filters.optimize_textures
except ImportError: pass
try: import meshtool.filters.optimize_filters.adjust_texcoords
except ImportError: pass
try: import meshtool.filters.optimize_filters.normalize_indices
except ImportError: pass
try: import meshtool.filters.optimize_filters.split_triangle_texcoords
except ImportError: pass
try: import meshtool.filters.optimize_filters.optimize_sources
except ImportError: pass

#Atlasing
try: import meshtool.filters.atlas_filters.make_atlases
except ImportError: pass

#Simplification
try: import meshtool.filters.simplify_filters.sander_simplify
except ImportError: pass
try: import meshtool.filters.simplify_filters.add_back_pm
except ImportError: pass

#Meta filters
try: import meshtool.filters.meta_filters.medium_optimizations
except ImportError: pass
try: import meshtool.filters.meta_filters.full_optimizations
except ImportError: pass

#Save filters last
try: import meshtool.filters.panda_filters.save_screenshot
except ImportError: pass
try: import meshtool.filters.panda_filters.save_rotate_screenshots
except ImportError: pass
try: import meshtool.filters.save_filters.save_collada
except ImportError: pass
try: import meshtool.filters.save_filters.save_collada_zip
except ImportError: pass
try: import meshtool.filters.save_filters.save_badgerfish
except ImportError: pass
try: import meshtool.filters.save_filters.save_ply
except ImportError: pass
try: import meshtool.filters.save_filters.save_obj
except ImportError: pass
try: import meshtool.filters.save_filters.save_obj_zip
except ImportError: pass
try: import meshtool.filters.save_filters.save_bam
except ImportError: pass
