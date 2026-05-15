#!/usr/bin/env python3

import pathlib
import sys


if __package__ in {None, ""}:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from gpi_repro.policy.cli import main


if __name__ == "__main__":
    main()
