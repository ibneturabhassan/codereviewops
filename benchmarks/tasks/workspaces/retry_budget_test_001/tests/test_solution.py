# ruff: noqa
from solution import retry

def test_normal_case():
    assert retry([False, True], 2)
