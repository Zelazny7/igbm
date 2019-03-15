
def indices(l):
    """return sorted positions of elements in l"""
    seen = set()
    uniq = [x for x in sorted(l) if x not in seen and not seen.add(x)]
    lookup = {k: i for (i, k) in enumerate(uniq)}
    return [lookup[x] for x in l]
