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
        for i in range(2):
            f = open(os.path.join(CURDIR, 'test.pdae'))
            if i % 2 == 0:
                f = StringIO.StringIO(f.read())
            start = time.time()
            refinements = readPDAE(f)
            after = time.time()
            print 'Took', after - start, 'seconds to load'
            self.assertEqual(len(refinements), 23222)
            
    def testPartial(self):
        to_test = ['test.pdae', 'planterfeeder.dae.pdae', 'terrain_test_2.dae.pdae']
        
        for test_file in to_test:
            f = open(os.path.join(CURDIR, test_file))
            blessed_refinements = readPDAE(f)
            
            B = 1
            KB = 1024
            MB = KB * KB
            test_block_sizes = [100 * B,
                                500 * B,
                                1 * KB,
                                5 * KB,
                                50 * KB,
                                100 * KB,
                                1 * MB,
                                2 * MB,
                                5 * MB]
            
            for block_size in test_block_sizes:
                start = time.time()
                block_size = 50000
                f = open(os.path.join(CURDIR, test_file))
                curdata = f.read(block_size)
                refinements_read = 0
                num_refinements = None
                all_pm_refinements = []
                while len(curdata) > 0:
                    (refinements_read, num_refinements, pm_refinements, data_left) = readPDAEPartial(curdata, refinements_read, num_refinements)
                    all_pm_refinements.extend(pm_refinements)
                    if data_left is not None:
                        curdata = data_left + f.read(block_size)
                    else:
                        curdata = f.read(block_size)
                after = time.time()
                print 'Took', after - start, 'seconds to load'
                self.assertEqual(all_pm_refinements, blessed_refinements)
    
if __name__ == '__main__':
    unittest.main()
