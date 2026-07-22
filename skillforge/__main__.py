# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Entry point for `python -m skillforge`.

Invokes the offline GEPA skill evolver CLI.

Examples:
    python -m skillforge --skill git-review
    python -m skillforge --skill git-review --dry-run
    python -m skillforge --skill git-review --iterations 5 --reuse-dataset
    python -m skillforge --help
"""
from skillforge.cli import main

if __name__ == "__main__":
    main()
