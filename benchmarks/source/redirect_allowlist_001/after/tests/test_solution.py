# ruff: noqa
from solution import allowed_host

def test_normal_case():
    assert allowed_host("service.example", "example")
