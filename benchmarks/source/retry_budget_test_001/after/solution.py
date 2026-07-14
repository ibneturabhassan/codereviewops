# ruff: noqa
def retry(results, budget):
    return any(results[:budget])
