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


class _AliasVariable:
    def __init__(self, relation, name, sign=1):
        self._relation = relation
        if sign < 0:
            if name.startswith('-'):
                self._name = name[1:]
            else:
                self._name = '-' + name
        else:
            self._name = name

    @property
    def name(self):
        return self._name

    def __eq__(self, other):
        return self._relation.equal(self._name, other._name)

    def __hash__(self):
        return hash(self._relation.canonical(self._name))


class AliasRelation:
    def __init__(self):
        self._aliases = {}

    def add(self, a, b):
        aliases = self.aliases(a) | self.aliases(b)
        self._aliases[a] = aliases
        self._aliases[b] = aliases

    def aliases(self, a):
        return self._aliases.setdefault(a, OrderedSet([a]))

    def equal(self, a, b):
        return a in self.aliases(b)

    def canonical(self, a):
        return self.aliases(a)[0]

    @property
    def canonical_variables(self):
        return set([aliases[0] for aliases in self._aliases.values()])

    def __iter__(self):
        # TODO optimize
        for canonical_variable in self.canonical_variables:
            yield canonical_variable, self.aliases(canonical_variable)[1:]


class AliasSet:
    def __init__(self, relation):
        self._relation = relation
        self._s = set()

    def add(self, a):
        self._s.add(_AliasVariable(self._relation, a))

    def remove(self, a):
        self._s.remove(_AliasVariable(self._relation, a))

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
        for alias_variable in self._s:
            yield alias_variable.name


class AliasDict:
    def __init__(self, relation):
        self._relation = relation
        self._d = {}

    def __setitem__(self, key, val):
        varp = _AliasVariable(self._relation, key, sign=1)
        if varp.name in self._relation:
            self._d[varp] = val
        else:
            varn = _AliasVariable(self._relation, key, sign=-1)
            if varn.name in self._relation:
                self._d[varn] = -val
            else:
                self._d[varp] = val

    def __getitem__(self, key):
        try:
            return self._d[_AliasVariable(self._relation, key, sign=1)]
        except KeyError:
            return -self._d[_AliasVariable(self._relation, key, sign=-1)]

    def __delitem__(self, key):
        try:
            del self._d[_AliasVariable(self._relation, key, sign=1)]
        except KeyError:
            del self._d[_AliasVariable(self._relation, key, sign=-1)]

    def __contains__(self, key):
        if _AliasVariable(self._relation, key, sign=1) in self._d:
            return True
        else:
            if _AliasVariable(self._relation, key, sign=-1) in self._d:
                return True
            else:
                return False

    def __len__(self):
        return len(self._d)

    def __iter__(self):
        for key, value in self._d:
            yield key.name, value

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
        for key in self._d.keys():
            yield key.name

    def values(self):
        return self._d.values()

    def iteritems(self):
        return self.__iter__()