import collections


version_tuple = collections.namedtuple("Row", ["major", "minor"])
MIN_PYTHON_VERSION = version_tuple(major=3, minor=7)
MIN_TENSORFLOW_VERSION = version_tuple(major=2, minor=4)
MIN_NUMPY_VERSION = version_tuple(major=1, minor=19)