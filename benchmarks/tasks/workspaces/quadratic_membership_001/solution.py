# ruff: noqa
def review(groups, allowed_list):
    result = []
    for group in groups:
        for item in group:
            if item in allowed_list: result.append(item)
    return result
