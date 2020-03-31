import collections

# def NestedDictValues(d):
#     for v in d.values():
#         if isinstance(v, dict):
#             yield from NestedDictValues(v)
#         else:
#             if isinstance(v, list):
#                 yield from NestedListValues(v)
#             else:
#                 yield v
#
#
# def NestedListValues(d):
#     for v in d:
#         if isinstance(v, list):
#             yield from NestedListValues(v)
#         else:
#             yield v


def flatten(l):
    for el in l:
        if isinstance(el, collections.abc.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el
