from meshtool.args import *
from meshtool.filters.base_filters import *
import os
import zipfile
from StringIO import StringIO
import posixpath

def FilterGenerator():
    class ColladaZipSaveFilter(SaveFilter):
        def __init__(self):
            super(ColladaZipSaveFilter, self).__init__('save_collada_zip', 'Saves a collada file and textures in a zip file. Normalizes texture paths.')
        def apply(self, mesh, filename):
            if os.path.exists(filename):
                raise FilterException("specified filename already exists")
                       
            zip = zipfile.ZipFile(filename, 'w', zipfile.ZIP_DEFLATED)
            
            basename = os.path.basename(filename)
            dotloc = basename.find('.')
            if dotloc == -1:
                dirname = basename
            else:
                dirname = basename[:dotloc]
                
            names_used = [dirname + '.dae']
            
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
                
                zip.writestr("%s/%s%s" % (dirname, base_img_name, img_ext), img_data)
                
                cimg.path = "./%s%s" % (base_img_name, img_ext)
            
            dae_data = StringIO()
            mesh.write(dae_data)
            dae_data = dae_data.getvalue()

            zip.writestr("%s/%s.dae" % (dirname, dirname), dae_data)
            
            zip.close()
            
            return mesh
    return ColladaZipSaveFilter()
