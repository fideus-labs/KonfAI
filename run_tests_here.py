import sys, os
sys.meta_path = [f for f in sys.meta_path if "editable" not in type(f).__module__.lower()]
sys.path.insert(0, os.getcwd())
import konfai; assert konfai.__file__.startswith(os.getcwd()), konfai.__file__
import pytest
sys.exit(pytest.main(sys.argv[1:]))
