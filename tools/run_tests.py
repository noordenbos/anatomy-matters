#!/usr/bin/env python3
"""Run unittest cases and standalone test functions without a pytest dependency."""
from pathlib import Path
import runpy
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(str(ROOT/'tests')))
if not result.wasSuccessful():sys.exit(1)
count=0
for path in sorted((ROOT/'tests').glob('test_*.py')):
    for name,func in runpy.run_path(str(path)).items():
        if name.startswith('test_') and callable(func):func();count+=1
print(f'{result.testsRun} unittest cases and {count} standalone checks passed')
