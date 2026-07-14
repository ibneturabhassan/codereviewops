# ruff: noqa
import re

def review(records, expression):
    output = []
    for record in records:
        pattern = re.compile(expression)
        if pattern.search(record): output.append(record)
    return output
