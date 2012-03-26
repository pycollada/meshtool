try:
    from pdae_cython import *
except ImportError:
    from pdae_python import *
    
import unittest
import time
import os
try:
    import cStringIO as StringIO
except ImportError:
    import StringIO
CURDIR = os.path.abspath(os.path.dirname(__file__))

class PM_OP:
    INDEX_UPDATE = 1
    TRIANGLE_ADDITION = 2
    VERTEX_ADDITION = 3

class PDAETest(unittest.TestCase):
    def testLoad(self):
        for i in range(4):
            f = open(os.path.join(CURDIR, 'test.pdae'))
            if i % 2 == 0:
                f = StringIO.StringIO(f.read())
            start = time.time()
            refinements = readPDAE(f)
            after = time.time()
            print 'Took', after - start, 'seconds to load'
            self.assertEqual(len(refinements), 23222)
            
    def testPartial(self):
        start = time.time()
        f = open(os.path.join(CURDIR, 'test.pdae'))
        curdata = f.read(50000)
        refinements_read = 0
        num_refinements = None
        all_pm_refinements = []
        while len(curdata) > 0:
            (refinements_read, num_refinements, pm_refinements, data_left) = readPDAEPartial(curdata, refinements_read, num_refinements)
            all_pm_refinements.extend(pm_refinements)
            if data_left is not None:
                curdata = data_left + f.read(50000)
            else:
                curdata = f.read(50000)
        after = time.time()
        print 'Took', after - start, 'seconds to load'
        self.assertEqual(len(all_pm_refinements), 23222)
    
if __name__ == '__main__':
    unittest.main()
