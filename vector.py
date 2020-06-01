class vec:
    def __init__(self, *coords):
        self.pos = coords
    def __getitem__(self, i):
        return self.pos[i]
    def __len__(self):
        return len(self.pos)
    def __repr__(self):
        return str(self.pos)
    def __str__(self):
        return self.__repr__()

    @property
    def mag(self):
        return sum(el**2 for el in self) ** (1/2)
    @property
    def unit(self):
        return vec(*(el / self.mag for el in self))

    @classmethod
    def from_unit(cls, unit, mag):
        if isinstance(unit, tuple) or isinstance(unit, list):
            unit = vec(*unit).unit
        return cls(*(mag * el for el in unit))

    def type_conv(self, x):
        if isinstance(x, v) and len(x) == len(self):
            return x
        if isinstance(x, tuple) or isinstance(x, list):
            return vec(*x)
        if isinstance(x, int) or isinstance(x, float):
            return vec(*(x for el in self))
        print("vec error: operand of incorrect type or len")
        return None

    def __add__(self, x):
        if x_conv := self.type_conv(x):
            return vec(*(el + x_conv[idx] for idx, el in enumerate(self)))
    def __sub__(self, x):
        if x_conv := self.type_conv(x):
            return vec(*(el - x_conv[idx] for idx, el in enumerate(self)))
    def __mul__(self, x):
        if isinstance(x, int) or isinstance(x, float):
            return vec(*(el * x for el in self))

    @staticmethod
    def dot(a, b):
        if len(a) == len(b):
            return sum(ae * be for ae, be in zip(a, b))

if __name__ == "__main__":
    v = vec(1, 1)
    print("Test __getitem__: ", str(v[0]))
    print("Test __len__: ", str(len(v)))
    print("Test __repr__: ", v)
    print("Test __str__: ", str(v))

    print("Test vec.mag: ", str(v.mag))
    print("Test vec.unit: ", str(v.unit))

    a = vec.from_unit((0.707, 0.707), 1.414)
    print("Test vec.from_unit, should be 1.414: ", str(a.mag))
    print("Test vec.from_unit, should be (0.707, 0.707): ", str(a.unit))