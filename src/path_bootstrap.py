import os
import sys


PROJECT_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODETR_DIR = os.path.join(PROJECT_BASE_DIR, "Co-DETR")
PARSEQ_DIR = os.path.join(PROJECT_BASE_DIR, "parseq")


def _prepend_once(path):
    if path not in sys.path:
        sys.path.insert(0, path)


def ensure_codetr_path():
    _prepend_once(CODETR_DIR)
    return CODETR_DIR


def ensure_parseq_path():
    _prepend_once(PARSEQ_DIR)
    return PARSEQ_DIR
