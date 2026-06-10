#!/usr/bin/env python3

try:
    from iopath.common.file_io import PathManager
except ModuleNotFoundError:
    class PathManager:
        def open(self, path, mode="r", *args, **kwargs):
            return open(path, mode, *args, **kwargs)


path_manager = PathManager()
