class vec:
    """ A tuple-like class that supports vector operations, similar to numpy arrays.
    """

    def __init__(self, *coords):
        self._pos = tuple([float(pt) for pt in coords])
        self._unit = None

        # TODO: Make this a private attribute and use getter/setter
        self.mag = sum(el ** 2 for el in self._pos) ** (1 / 2)

    def __getitem__(self, i):
        return self._pos[i]

    def __len__(self):
        return len(self._pos)

    def __repr__(self):
        return str(self._pos)

    def __str__(self):
        return self.__repr__()

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < len(self):
            result = self[self.n]
            self.n += 1
            return result
        else:
            raise StopIteration

    @property
    def unit(self) -> "vec":
        """ Compute a unit vector in the direction of self.

        Returns:
            vec -- a unit vector in the direction of self.
        """
        if not self._unit:
            self._unit = vec(*(el / self.mag for el in self._pos))
        return self._unit

    @classmethod
    def from_unit(cls, unit: tuple, mag: float) -> "vec":
        """ Create a vec instance by specifying a direction and length.

        Arguments:
            unit {tuple} -- The direction of desired vector. If not of mag 1, will be internally reduced to 1.
            mag {float} -- [description]

        Returns:
            vec -- A vector in the direction of unit and the length of mag.
        """
        if isinstance(unit, tuple) or isinstance(unit, list):
            unit = vec(*unit).unit._pos
        return cls(*(mag * el for el in unit))

    def __add__(self, x: "vec") -> "vec":
        return vec(*(el + x[idx] for idx, el in enumerate(self)))

    def __sub__(self, x: "vec") -> "vec":
        return vec(*(el - x[idx] for idx, el in enumerate(self)))

    def __mul__(self, x: float) -> "vec":
        return vec(*(el * x for el in self))

    @staticmethod
    def dot(a: "vec", b: "vec") -> float:
        return float(sum(ae * be for ae, be in zip(a, b)))

    @staticmethod
    def angle(a: "vec", b: "vec") -> float:
        pass
