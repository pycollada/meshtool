from meshtool.filters.base_filters import FilterFactory
import sys

factory = FilterFactory()

def warn(name, e):
    sys.stderr.write("Warning: filter '%s' disabled because of ImportError: %s\n" % (name, str(e)))

#Load filters first
try: import meshtool.filters.load_filters.load_collada
except ImportError as e: warn('load_collada', e)
try: import meshtool.filters.load_filters.load_obj
except ImportError as e: warn('load_obj', e)

#Print filters
try: import meshtool.filters.print_filters.print_textures
except ImportError as e: warn('print_textures', e)
try: import meshtool.filters.print_filters.print_json
except ImportError as e: warn('print_json', e)
try: import meshtool.filters.print_filters.print_info
except ImportError as e: warn('print_info', e)
try: import meshtool.filters.print_filters.print_instances
except ImportError as e: warn('print_instances', e)
try: import meshtool.filters.print_filters.print_scene
except ImportError as e: warn('print_scene', e)
try: import meshtool.filters.print_filters.print_render_info
except ImportError as e: warn('print_render_info', e)
try: import meshtool.filters.print_filters.print_bounds
except ImportError as e: warn('print_bounds', e)

#Viewer
try: import meshtool.filters.panda_filters.viewer
except ImportError as e: warn('viewer', e)
try: import meshtool.filters.panda_filters.collada_viewer
except ImportError as e: warn('collada_viewer', e)
try: import meshtool.filters.panda_filters.pm_viewer
except ImportError as e: warn('pm_viewer', e)

#Optimizations
try: import meshtool.filters.optimize_filters.combine_effects
except ImportError as e: warn('combine_effects', e)
try: import meshtool.filters.optimize_filters.combine_materials
except ImportError as e: warn('combine_materials', e)
try: import meshtool.filters.optimize_filters.combine_primitives
except ImportError as e: warn('combine_primitives', e)
try: import meshtool.filters.optimize_filters.strip_lines
except ImportError as e: warn('strip_lines', e)
try: import meshtool.filters.optimize_filters.strip_empty_geometry
except ImportError as e: warn('strip_empty_geometry', e)
try: import meshtool.filters.optimize_filters.strip_unused_sources
except ImportError as e: warn('strip_unused_sources', e)
try: import meshtool.filters.optimize_filters.triangulate
except ImportError as e: warn('triangulate', e)
try: import meshtool.filters.optimize_filters.generate_normals
except ImportError as e: warn('generate_normals', e)
try: import meshtool.filters.optimize_filters.save_mipmaps
except ImportError as e: warn('save_mipmaps', e)
try: import meshtool.filters.optimize_filters.optimize_textures
except ImportError as e: warn('optimize_textures', e)
try: import meshtool.filters.optimize_filters.adjust_texcoords
except ImportError as e: warn('adjust_texcoords', e)
try: import meshtool.filters.optimize_filters.normalize_indices
except ImportError as e: warn('normalize_indices', e)
try: import meshtool.filters.optimize_filters.split_triangle_texcoords
except ImportError as e: warn('split_triangle_texcoords', e)
try: import meshtool.filters.optimize_filters.optimize_sources
except ImportError as e: warn('optimize_sources', e)

#Atlasing
try: import meshtool.filters.atlas_filters.make_atlases
except ImportError as e: warn('make_atlases', e)

#Simplification
try: import meshtool.filters.simplify_filters.sander_simplify
except ImportError as e: warn('sander_simplify', e)
try: import meshtool.filters.simplify_filters.add_back_pm
except ImportError as e: warn('add_back_pm', e)

#Meta filters
try: import meshtool.filters.meta_filters.medium_optimizations
except ImportError as e: warn('medium_optimizations', e)
try: import meshtool.filters.meta_filters.full_optimizations
except ImportError as e: warn('full_optimizations', e)

#Save filters last
try: import meshtool.filters.panda_filters.save_screenshot
except ImportError as e: warn('save_screenshot', e)
try: import meshtool.filters.panda_filters.save_rotate_screenshots
except ImportError as e: warn('save_rotate_screenshots', e)
try: import meshtool.filters.save_filters.save_collada
except ImportError as e: warn('save_collada', e)
try: import meshtool.filters.save_filters.save_collada_zip
except ImportError as e: warn('save_collada_zip', e)
try: import meshtool.filters.save_filters.save_badgerfish
except ImportError as e: warn('save_badgerfish', e)
try: import meshtool.filters.save_filters.save_ply
except ImportError as e: warn('save_ply', e)
try: import meshtool.filters.save_filters.save_obj
except ImportError as e: warn('save_obj', e)
try: import meshtool.filters.save_filters.save_obj_zip
except ImportError as e: warn('save_obj_zip', e)
try: import meshtool.filters.save_filters.save_bam
except ImportError as e: warn('save_bam', e)
try: import meshtool.filters.save_filters.save_threejs_scene
except ImportError as e: warn('save_threejs_scene', e)
