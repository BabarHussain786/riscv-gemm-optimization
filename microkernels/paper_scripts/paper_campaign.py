#!/usr/bin/env python3
"""Compatibility entry point for the organized paper-script layout."""
from pathlib import Path
import runpy

runpy.run_path(
    str(Path(__file__).resolve().parent / 'orchestration' / 'paper_campaign.py'),
    run_name='__main__',
)
