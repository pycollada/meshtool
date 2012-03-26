import unittest
import sys

if __name__ == '__main__':
    suite = unittest.TestLoader().discover("tests")
    ret = unittest.TextTestRunner(verbosity=2).run(suite)
    if ret.wasSuccessful():
        sys.exit(0)
    sys.exit(1)
