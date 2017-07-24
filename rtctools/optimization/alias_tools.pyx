import collections


# From https://code.activestate.com/recipes/576694/
class OrderedSet(collections.MutableSet):

    def __init__(self, iterable=None):
        self.end = end = [] 
        end += [None, end, end]         # sentinel node for doubly linked list
        self.map = {}                   # key --> [key, prev, next]
        if iterable is not None:
            self |= iterable

    def __len__(self):
        return len(self.map)

    def __contains__(self, key):
        return key in self.map

    def __getitem__(self, index):
        # Method added by JB
        if isinstance(index, slice):
            start, stop, stride = index.indices(len(self))
            return [self.__getitem__(i) for i in xrange(start, stop, stride)]
        else:
            end = self.end
            curr = end[2]
            i = 0
            while curr is not end:
                if i == index:
                    return curr[0]
                curr = curr[2]
                i += 1
            raise IndexError('set index {} out of range with length {}'.format(index, len(self)))

    def add(self, key):
        if key not in self.map:
            end = self.end
            curr = end[1]
            curr[2] = end[1] = self.map[key] = [key, curr, end]

    def discard(self, key):
        if key in self.map:        
            key, prev, next = self.map.pop(key)
            prev[2] = next
            next[1] = prev

    def __iter__(self):
        end = self.end
        curr = end[2]
        while curr is not end:
            yield curr[0]
            curr = curr[2]

    def __reversed__(self):
        end = self.end
        curr = end[1]
        while curr is not end:
            yield curr[0]
            curr = curr[1]

    def pop(self, last=True):
        if not self:
            raise KeyError('set is empty')
        key = self.end[1][0] if last else self.end[2][0]
        self.discard(key)
        return key

    def __repr__(self):
        if not self:
            return '%s()' % (self.__class__.__name__,)
        return '%s(%r)' % (self.__class__.__name__, list(self))

    def __eq__(self, other):
        if isinstance(other, OrderedSet):
            return len(self) == len(other) and list(self) == list(other)
        return set(self) == set(other)
# End snippet


class AliasRelation:
    def __init__(self):
        self._aliases = {}
        self._canonical_variables = OrderedSet()

    def add(self, a, b):
        aliases = self.aliases(a)
        for v in self.aliases(b):
            aliases.add(v)
        for v in aliases:
            self._aliases[v] = aliases
        self._canonical_variables.add(aliases[0])
        for v in aliases[1:]:
            try:
                self._canonical_variables.remove(v)
            except KeyError:
                pass

    def aliases(self, a):
        return self._aliases.get(a, OrderedSet([a]))

    def canonical_signed(self, a):
        if a in self._aliases:
            return self.aliases(a)[0], 1
        else:
            if a[0] == '-':
                b = a[1:]
            else:
                b = '-' + a
            if b in self._aliases:
                return self.aliases(b)[0], -1
            else:
                return self.aliases(a)[0], 1

    @property
    def canonical_variables(self):
        return self._canonical_variables

    def __iter__(self):
        return ((canonical_variable, self.aliases(canonical_variable)[1:]) for canonical_variable in self._canonical_variables)


class AliasDict:
    def __init__(self, relation, other=None, signed_values=True):
        self._relation = relation
        self._d = {}
        self._signed_values = signed_values
        if other:
            self.update(other)

    def _canonical_signed(self, key):
        var, sign = self._relation.canonical_signed(key)
        if self._signed_values:
            return var, sign
        else:
            return var, 1

    def __setitem__(self, key, val):
        var, sign = self._canonical_signed(key)
        if isinstance(val, tuple):
            self._d[var] = [-c if sign < 0 else c for c in val]
        else:
            self._d[var] = -val if sign < 0 else val

    def __getitem__(self, key):
        var, sign = self._canonical_signed(key)
        val = self._d[var]
        if isinstance(val, tuple):
            return [-c if sign < 0 else c for c in val]
        else:
            return -val if sign < 0 else val

    def __delitem__(self, key):
        var, sign = self._canonical_signed(key)
        del self._d[var]

    def __contains__(self, key):
        var, sign = self._canonical_signed(key)
        return var in self._d

    def __len__(self):
        return len(self._d)

    def __iter__(self):
        return self._d.iteritems()

    def update(self, other):
        for key, value in other.iteritems():
            self[key] = value

    def get(self, key, default=None):
        if key in self:
            return self[key]
        else:
            return default

    def setdefault(self, key, default):
        if key in self:
            return self[key]
        else:
            self[key] = default
            return default

    def keys(self):
        return self._d.keys()

    def values(self):
        return self._d.values()

    def iteritems(self):
        return self.__iter__()