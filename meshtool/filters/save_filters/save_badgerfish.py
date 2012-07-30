from meshtool.filters.base_filters import SaveFilter, FilterException
import os
import badgerfish

def FilterGenerator():
    class BadgerFishSaveFilter(SaveFilter):
        def __init__(self):
            super(BadgerFishSaveFilter, self).__init__('save_badgerfish', 'Saves a collada file as JSON badgerfish')
        def apply(self, mesh, filename):
            if os.path.exists(filename):
                raise FilterException("specified filename already exists")
            
            json_data = badgerfish.to_json(mesh.xmlnode.getroot(), indent=4)
            f = open(filename, 'w')
            f.write(json_data)
            f.close()
            
            return mesh
    return BadgerFishSaveFilter()
from meshtool.filters import factory
factory.register(FilterGenerator().name, FilterGenerator)
