# ruff: noqa
from solution import request

def test_normal_case():
    assert request([500, 200]) == 200
