import unittest
import os
import collada
from meshtool.filters.load_filters.load_obj import loadOBJ, filepath_loader

CURDIR = os.path.dirname(os.path.abspath(__file__))
DATADIR = os.path.join(CURDIR, 'data')
OBJDIR = os.path.join(DATADIR, 'obj')

class ObjTester(unittest.TestCase):
    def setUp(self):
        self.obj_box = os.path.join(OBJDIR, 'box.obj')
        self.obj_spider = os.path.join(OBJDIR, 'spider.obj')
        self.jpg_wal67ar_small = os.path.join(OBJDIR, 'wal67ar_small.jpg')
        self.obj_regr01 = os.path.join(OBJDIR, 'regr01.obj')
        self.obj_testline = os.path.join(OBJDIR, 'testline.obj')
        self.obj_testpoints = os.path.join(OBJDIR, 'testpoints.obj')
        self.obj_testmixed = os.path.join(OBJDIR, 'testmixed.obj')
    
    def load_obj(self, filename):
        f = open(filename, 'rb')
        col = loadOBJ(f.read(), aux_file_loader=filepath_loader(filename), validate_output=True)
        return col
    
    def assertIterableAlmostEqual(self, i1, i2):
        self.assertEqual(len(i1), len(i2))
        for x, y in zip(i1, i2):
            self.assertAlmostEqual(x, y)
    
    def test_box(self):
        col = self.load_obj(self.obj_box)
        self.assertEqual(len(col.geometries), 1)
        geom = col.geometries[0]
        self.assertEqual(len(geom.primitives), 1)
        prim = geom.primitives[0]
        self.assertIsInstance(prim, collada.polylist.Polylist)
        self.assertEqual(len(prim), 6)
        self.assertEqual(len(prim.vertex), 8)
        self.assertEqual(len(prim.texcoordset), 0)
        self.assertIsNone(prim.normal)
        
        col.save()

    def test_spider(self):
        col = self.load_obj(self.obj_spider)
        self.assertEqual(len(col.geometries), 1)
        geom = col.geometries[0]
        self.assertEqual(len(geom.primitives), 19)
        
        self.assertEqual(len(geom.primitives[0]), 80)
        self.assertEqual(len(geom.primitives[1]), 60)
        self.assertEqual(len(geom.primitives[2]), 98)
        self.assertEqual(len(geom.primitives[3]), 98)
        self.assertEqual(len(geom.primitives[4]), 98)
        self.assertEqual(len(geom.primitives[5]), 98)
        self.assertEqual(len(geom.primitives[6]), 98)
        self.assertEqual(len(geom.primitives[7]), 98)
        self.assertEqual(len(geom.primitives[8]), 98)
        self.assertEqual(len(geom.primitives[9]), 98)
        self.assertEqual(len(geom.primitives[10]), 42)
        self.assertEqual(len(geom.primitives[11]), 42)
        self.assertEqual(len(geom.primitives[12]), 90)
        self.assertEqual(len(geom.primitives[13]), 20)
        self.assertEqual(len(geom.primitives[14]), 90)
        self.assertEqual(len(geom.primitives[15]), 42)
        self.assertEqual(len(geom.primitives[16]), 42)
        self.assertEqual(len(geom.primitives[17]), 38)
        self.assertEqual(len(geom.primitives[18]), 38)
        
        prim = geom.primitives[0]
        self.assertEqual(len(prim.vertex), 762)
        self.assertEqual(len(prim.texcoordset[0]), 302)
        self.assertEqual(len(prim.normal), 747)
        
        self.assertEqual(len(col.materials), 5)
        self.assertEqual(len(col.effects), 5)
        
        skin = col.effects['Skin']
        brust = col.effects['Brusttex']
        leib = col.effects['HLeibTex']
        bein = col.effects['BeinTex']
        aug = col.effects['Augentex']
        
        self.assertIterableAlmostEqual(skin.ambient, [0.2, 0.2, 0.2, 1])
        self.assertIterableAlmostEqual(skin.specular, [0, 0, 0, 1])
        self.assertAlmostEqual(skin.transparency, 0)
        difftex = skin.diffuse.sampler.surface.image
        texdata = difftex.data
        blessed_data = open(self.jpg_wal67ar_small, 'rb').read()
        self.assertEqual(len(texdata), 9288)
        self.assertEqual(texdata, blessed_data)
        
        image_paths = [cimg.path for cimg in col.images]
        self.assertIn('./engineflare1.jpg', image_paths)
        self.assertIn('./wal67ar_small.jpg', image_paths)
        self.assertIn('./wal69ar_small.jpg', image_paths)
        self.assertIn('./SpiderTex.jpg', image_paths)
        self.assertIn('./drkwood2.jpg', image_paths)
        
        col.save()

    def test_regr01(self):
        col = self.load_obj(self.obj_regr01)
        
        self.assertEqual(len(col.geometries), 1)
        geom = col.geometries[0]
        self.assertEqual(len(geom.primitives), 55)
        
        #     0    g Base
        #    43    g Tag-stor
        
        boundgeoms = list(col.scene.objects('geometry'))
        self.assertEqual(len(boundgeoms), 1)
        boundgeom = boundgeoms[0]
        prims = list(boundgeom.primitives())
        
        tagstar = prims[43]
        # this has vertex and texcoords, no normals
        self.assertEqual(len(tagstar), 20)
        self.assertGreaterEqual(len(tagstar.vertex), 0)
        self.assertIsNone(tagstar.normal)
        self.assertEqual(len(tagstar.texcoordset), 1)
        self.assertGreaterEqual(len(tagstar.texcoordset[0]), 0)
        
        base = prims[0]
        # this has vertex only
        self.assertEqual(len(base), 48)
        self.assertGreaterEqual(len(base.vertex), 0)
        self.assertIsNone(base.normal)
        self.assertEqual(len(base.texcoordset), 0)
        
        col.save()
        
    def test_lines(self):
        col = self.load_obj(self.obj_testline)
        
        self.assertEqual(len(col.geometries), 1)
        geom = col.geometries[0]
        self.assertEqual(len(geom.primitives), 1)
        
        prim = geom.primitives[0]
        self.assertIsInstance(prim, collada.lineset.LineSet)
        self.assertEqual(len(prim), 18)
        
        col.save()
        
    def test_points(self):
        col = self.load_obj(self.obj_testpoints)
        
        self.assertEqual(len(col.geometries), 1)
        geom = col.geometries[0]
        self.assertEqual(len(geom.primitives), 1)
        
        prim = geom.primitives[0]
        self.assertIsInstance(prim, collada.lineset.LineSet)
        self.assertEqual(len(prim), 24)
        
        col.save()
        
    def test_mixed(self):
        col = self.load_obj(self.obj_testmixed)
        
        self.assertEqual(len(col.geometries), 1)
        geom = col.geometries[0]
        self.assertEqual(len(geom.primitives), 2)
        
        poly = geom.primitives[0]
        lines = geom.primitives[1]
        self.assertIsInstance(lines, collada.lineset.LineSet)
        # 24 points + 18 lines
        self.assertEqual(len(lines), 42)
        self.assertIsInstance(poly, collada.polylist.Polylist)
        self.assertEqual(len(poly), 6)
        
        col.save()
    