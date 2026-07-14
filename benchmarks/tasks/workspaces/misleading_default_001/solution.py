# ruff: noqa
def validate(value, strict=False):
    return bool(value) if strict else True
