# ruff: noqa
from solution import page

def test_normal_case():
    assert page([1, 2, 3], 0, 2) == [1]
