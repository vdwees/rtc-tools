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
            return [self.__getitem__(i) for i in range(start, stop, stride)]
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
                self._canonical_variables.remove(b)
            except KeyError:
                pass

    def aliases(self, a):
        return self._aliases.setdefault(a, OrderedSet([a]))

    def canonical(self, a):
        return self.aliases(a)[0]

    def canonical_signed(self, a):
        if a in self._aliases:
            return self.aliases(a)[0], 1
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


class AliasSet:
    def __init__(self, relation):
        self._relation = relation
        self._s = set()

    def add(self, a):
        self._s.add(self._relation.canonical(a))

    def remove(self, a):
        self._s.remove(self._relation.canonical(a))

    def __add__(self, other):
        new = AliasSet(self._relation)
        for s in [self, other]:
            for a in s:
                new.add(a)
        return new
    
    def __sub__(self, other):
        new = AliasSet(self._relation)
        for a in self:
            new.add(a)
        for a in other:
            new.remove(a)
        return new

    def __and__(self, other):
        new = AliasSet(self._relation)
        for a in self:
            if a in other:
                new.add(a)
        return new
    
    def __or__(self, other):
        return self.__add__(other)

    def __len__(self):
        return len(self._s)

    def __iter__(self):
        return self._s.__iter__()


class AliasDict:
    def __init__(self, relation, other=None):
        self._relation = relation
        self._d = {}
        if other:
            self.update(other)

    def __setitem__(self, key, val):
        varp = self._relation.canonical(key)
        if varp in self._d:
            self._d[varp] = val
        else:
            varn = self._relation.canonical('-' + key)
            if varn in self._d:
                if hasattr(val, '__iter__'):
                    self._d[varn] = [-c for c in val]
                else:
                    self._d[varn] = -val
            else:
                self._d[varp] = val

    def __getitem__(self, key):
        try:
            return self._d[self._relation.canonical(key)]
        except KeyError:
            try:
                val = self._d[self._relation.canonical('-' + key)]
            except KeyError:
                raise KeyError(key)
            else:
                if hasattr(val, '__iter__'):
                    return [-c for c in val]
                else:
                    return -val

    def __delitem__(self, key):
        try:
            del self._d[self._relation.canonical(key)]
        except KeyError:
            del self._d[self._relation.canonical('-' + key)]

    def __contains__(self, key):
        if self._relation.canonical(key) in self._d:
            return True
        else:
            if self._relation.canonical('-' + key) in self._d:
                return True
            else:
                return False

    def __len__(self):
        return len(self._d)

    def __iter__(self):
        return self._d.items()

    def update(self, other):
        for key, value in other.items():
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

    def items(self):
        return self.__iter__()