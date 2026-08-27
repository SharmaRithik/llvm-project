import os

import lit.formats
from lit.llvm import llvm_config

config.name = "CIR_TILE"
config.test_format = lit.formats.ShTest()
config.suffixes = [".cir"]
config.excludes = ["CMakeLists.txt", "lit.cfg.py", "lit.site.cfg.py"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.cir_tile_obj_root, "test")

llvm_config.use_default_substitutions()

tool_dirs = [
    config.cir_tile_tool_dir,
    config.cuda_tile_tool_dir,
    config.llvm_tools_dir,
]
tools = [
    "cir-tile-translate",
    "cuda-tile-opt",
    "cuda-tile-translate",
    "FileCheck",
    "not",
    "split-file",
]

llvm_config.add_tool_substitutions(tools, tool_dirs)
