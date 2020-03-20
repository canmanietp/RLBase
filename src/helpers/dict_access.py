def NestedDictValues(d):
  for v in d.values():
    if isinstance(v, dict):
      yield from NestedDictValues(v)
    else:
      if isinstance(v, list):
        yield from NestedListValues(v)
      else:
        yield v


def NestedListValues(d):
  for v in d:
    if isinstance(v, list):
      yield from NestedListValues(v)
    else:
      yield v

