# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import sys
import importlib
import importlib.util

# 兼容 Python 3
if sys.version_info[0] == 3:

    # 从 https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path
    # 导入模块的函数
    def import_file(module_name, file_path, make_importable=False):
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if make_importable:
            sys.modules[module_name] = module
        return module

else:
    # Python 2
    import imp

    def import_file(module_name, file_path, make_importable=None):
        module = imp.load_source(module_name, file_path)
        return module
