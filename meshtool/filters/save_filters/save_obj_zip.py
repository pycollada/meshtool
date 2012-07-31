from meshtool.filters.base_filters import SaveFilter, FilterException
import os
import zipfile
from StringIO import StringIO
import posixpath
import save_obj_util

def FilterGenerator():
    class ObjZipSaveFilter(SaveFilter):
        def __init__(self):
            super(ObjZipSaveFilter, self).__init__('save_obj_zip', 'Saves an OBJ file and textures in a zip file. Normalizes texture paths.')
        def apply(self, mesh, filename):
            if os.path.exists(filename):
                raise FilterException("specified filename already exists")

            z = zipfile.ZipFile(filename, 'w', zipfile.ZIP_DEFLATED)

            basename = os.path.basename(filename)
            dotloc = basename.find('.')
            if dotloc == -1:
                dirname = basename
            else:
                dirname = basename[:dotloc]

            names_used = [dirname + '.obj']

            prev_written = []
            for cimg in mesh.images:
                img_data = cimg.data
                img_name = posixpath.basename(cimg.path)

                dotloc = img_name.find('.')
                if dotloc == -1:
                    base_img_name = img_name
                    img_ext = ''
                else:
                    base_img_name = img_name[:dotloc]
                    img_ext = img_name[dotloc:]

                while base_img_name + img_ext in names_used:
                    base_img_name = base_img_name + 'x'

                img_path = "%s/%s%s" % (dirname, base_img_name, img_ext)
                if img_path not in prev_written:
                    z.writestr(img_path, img_data)
                    prev_written.append(img_path)

                cimg.path = "./%s%s" % (base_img_name, img_ext)

            mtl_data = StringIO()
            save_obj_util.write_mtl(mesh, mtl_data)
            z.writestr("%s/%s.mtl" % (dirname, dirname), mtl_data.getvalue())
            mtl_data.close()

            obj_data = StringIO()
            save_obj_util.write_obj(mesh, "%s.mtl" % (dirname), obj_data)
            z.writestr("%s/%s.obj" % (dirname, dirname), obj_data.getvalue())
            obj_data.close()

            z.close()

            return mesh
    return ObjZipSaveFilter()
from meshtool.filters import factory
factory.register(FilterGenerator().name, FilterGenerator)
