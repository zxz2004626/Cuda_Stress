#!/usr/bin/env python3
"""Top-level entry: `python3 pipeline.py loop --config config/pipeline.yaml`."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.cli import main

if __name__ == "__main__":
    sys.exit(main())
