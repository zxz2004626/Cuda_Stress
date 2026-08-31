#!/usr/bin/env python3
"""Module entry point: `python3 -m cuda_stress ...` (requires running from repo root)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.cli import main

if __name__ == "__main__":
    sys.exit(main())
