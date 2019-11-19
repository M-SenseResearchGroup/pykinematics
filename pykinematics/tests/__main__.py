if __name__ == '__main__':
    import sys
    import os
    import pytest

    errcode = pytest.main([os.path.dirname(__file__)] + sys.argv[1:])
    sys.exit(errcode)
