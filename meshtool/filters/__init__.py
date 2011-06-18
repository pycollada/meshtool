from meshtool.filters.base_filters import *
import sys

factory = FilterFactory()

#Load filters first
import meshtool.filters.load_filters.load_collada

#Print filters
import meshtool.filters.print_filters.print_textures
import meshtool.filters.print_filters.print_json
import meshtool.filters.print_filters.print_info
import meshtool.filters.print_filters.print_instances
import meshtool.filters.print_filters.print_scene
import meshtool.filters.print_filters.print_render_info

#Viewer
import meshtool.filters.panda_filters.viewer

#Optimizations
import meshtool.filters.optimize_filters.combine_effects
import meshtool.filters.optimize_filters.combine_materials
import meshtool.filters.optimize_filters.combine_primitives
import meshtool.filters.optimize_filters.strip_lines
import meshtool.filters.optimize_filters.strip_empty_geometry
import meshtool.filters.optimize_filters.strip_unused_sources
import meshtool.filters.optimize_filters.triangulate
import meshtool.filters.optimize_filters.generate_normals
import meshtool.filters.optimize_filters.save_mipmaps
import meshtool.filters.optimize_filters.optimize_textures
import meshtool.filters.optimize_filters.adjust_texcoords
import meshtool.filters.optimize_filters.normalize_indices
import meshtool.filters.optimize_filters.split_triangle_texcoords
import meshtool.filters.optimize_filters.optimize_sources

#Atlasing
import meshtool.filters.atlas_filters.make_atlases

#Simplification
import meshtool.filters.simplify_filters.simplify
import meshtool.filters.simplify_filters.load_pm

#Meta filters
import meshtool.filters.meta_filters.full_optimizations

#Save filters last
import meshtool.filters.panda_filters.save_screenshot
import meshtool.filters.panda_filters.save_rotate_screenshots
import meshtool.filters.save_filters.save_collada
import meshtool.filters.save_filters.save_collada_zip
