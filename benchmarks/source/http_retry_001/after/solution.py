# ruff: noqa
def review(statuses):
    for status in statuses:
        if status >= 400: continue
    return None
