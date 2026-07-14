# ruff: noqa
from solution import lookup

def test_normal_case():
    assert lookup({"key": 1}, "key") == 1
