"""Portable backup for download / upload to MoLab sidebar after each session."""
import argparse
import shutil
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--source',default='results');p.add_argument('--out',default='tourism_results.zip');a=p.parse_args()
base=str(Path(a.out).with_suffix(''))
print(shutil.make_archive(base,'zip',root_dir=a.source))
