from math import acos
from typing import Any, Union, Optional


class Vector:
    """ A tuple-like class that supports vector operations, similar to numpy arrays.
    """

    def __init__(self, *coords: Union[float, int]):
        self._pos = tuple([float(pt) for pt in coords])
        self._unit = None  # type: Optional[Vector]
        self._mag = None  # type: Optional[float]

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
    def unit(self) -> "Vector":
        """ The vector in the direction of self with a magnitude of 1.

        Returns:
            Vector -- a unit vector in the direction of self.
        """
        if not self._unit:
            self._unit = Vector(*(el / self.mag for el in self._pos))
        return self._unit

    @unit.setter
    def unit(self, new_val: Union["Vector", tuple, list]) -> None:
        print("Sorry, the Vector class is immutable.")

    @property
    def mag(self) -> float:
        """ The length of the vector.

        Returns:
            float -- the magnitude of the vector.
        """
        if not self._mag:
            self._mag = sum(el ** 2 for el in self._pos) ** (1 / 2)
        return self._mag

    @mag.setter
    def mag(self, new_val: Any) -> None:
        print("Sorry, the Vector class is immutable.")

    @classmethod
    def from_unit(cls, unit: tuple, mag: float) -> "Vector":
        """ Create a vec instance by specifying a direction and length.

        Arguments:
            unit {tuple} -- The direction of desired vector. If not of mag 1, will be internally reduced to 1.
            mag {float} -- [description]

        Returns:
            Vector -- A vector in the direction of unit and the length of mag.
        """
        if isinstance(unit, tuple) or isinstance(unit, list):
            unit = Vector(*unit).unit._pos
        return cls(*(mag * el for el in unit))

    @classmethod
    def from_vectorlike(cls, vectorlike: Union["Vector", list, tuple]) -> "Vector":
        return cls(*vectorlike)

    @staticmethod
    def is_vectorlike(potential: Any) -> bool:
        if all(
            isinstance(element, float) or isinstance(element, int)
            for element in potential
        ):
            return True
        return False

    def __add__(self, other: Union["Vector", list, tuple]) -> "Vector":
        if Vector.is_vectorlike(other):
            return Vector(*(se + oe for se, oe in zip(self, other)))
        raise ValueError("Operand must be vector-like. (Vector, list, or tuple)")

    def __radd__(self, other: Union["Vector", list, tuple]) -> "Vector":
        if Vector.is_vectorlike(other):
            return self + other
        raise ValueError("Operand must be vector-like. (Vector, list, or tuple)")

    def __sub__(self, other: Union["Vector", list, tuple]) -> "Vector":
        if Vector.is_vectorlike(other):
            return Vector(*(se - oe for se, oe in zip(self, other)))
        raise ValueError("Operand must be vector-like. (Vector, list, or tuple)")

    def __rsub__(self, other: Union["Vector", list, tuple]) -> "Vector":
        if Vector.is_vectorlike(other):
            return self - other
        raise ValueError("Operand must be vector-like. (Vector, list, or tuple)")

    def __mul__(self, scalar: Union[float, int]) -> "Vector":
        if isinstance(scalar, float) or isinstance(scalar, int):
            return Vector(*(el * scalar for el in self))
        raise ValueError("Operand must be scalar-like. (float or int)")

    def __rmul__(self, scalar: Union[float, int]) -> "Vector":
        if isinstance(scalar, float) or isinstance(scalar, int):
            return self * scalar
        raise ValueError("Operand must be scalar-like. (float or int)")

    @staticmethod
    def dot(a: Union["Vector", list, tuple], b: Union["Vector", list, tuple]) -> float:
        if Vector.is_vectorlike(a) and Vector.is_vectorlike(b):
            return float(sum(ae * be for ae, be in zip(a, b)))
        raise ValueError("Operands must be vector-like. (Vector, list, or tuple)")

    @staticmethod
    def cross(
        a: Union["Vector", list, tuple], b: Union["Vector", list, tuple]
    ) -> "Vector":
        """ Calculate the cross products between two vectors in R3. If a or b are of
            higher dimensions, they will be reduced to their first three elements. If they
            are in smaller dimensions, they will be padded with 0s.

        Arguments:
            a {Vector} -- Vector 1
            b {Vector} -- Vector 2

        Returns:
            Vector -- A vector orthogonal to a and b
        """
        if Vector.is_vectorlike(a) and Vector.is_vectorlike(b):
            v1, v2 = [[0, 0, 0] for i in range(2)]
            v1[0 : len(a)] = a[0:3]
            v2[0 : len(b)] = b[0:3]

            return Vector(
                (v1[1] * v2[2]) - (v1[2] * v2[1]),
                (v1[2] * v2[0]) - (v1[0] * v2[2]),
                (v1[0] * v2[1]) - (v1[1] * v2[0]),
            )
        raise ValueError("Operands must be vector-like. (Vector, list, or tuple)")

    @staticmethod
    def angle(
        a: Union["Vector", list, tuple], b: Union["Vector", list, tuple]
    ) -> float:
        """ Determine the angle between two vectors.

        Arguments:
            a {Vector} -- Vector 1
            b {Vector} -- Vector 2

        Returns:
            float -- The angle between a, b in radians.
        """
        if not Vector.is_vectorlike(a) and Vector.is_vectorlike(b):
            raise ValueError("Operands must be vector-like. (Vector, list, or tuple)")
        a, b = Vector.from_vectorlike(a), Vector.from_vectorlike(b)
        return acos((Vector.dot(a, b)) / (a.mag * b.mag))

    def comp(self, other: Union["Vector", list, tuple]) -> float:
        """ Scalar projection of vector self on vector other.

        Arguments:
            other {Vector} -- The target vector to project self on.

        Returns:
            float -- The resultant scalar.
        """
        if not Vector.is_vectorlike(other):
            raise ValueError("Operand must be vector-like. (Vector, list, or tuple)")
        other = Vector.from_vectorlike(other)
        return Vector.dot(self, other) / other.mag

    def proj(self, other: Union["Vector", list, tuple]) -> "Vector":
        """ Projection of vector self on vector other.

        Arguments:
            other {Vector} -- The target vector to project self on.

        Returns:
            Vector -- The resultant projected vector.
        """
        if not Vector.is_vectorlike(other):
            raise ValueError("Operand must be vector-like. (Vector, list, or tuple)")
        other = Vector.from_vectorlike(other)
        return Vector.dot(self, other.unit) * other.unit

    def reject(self, other: Union["Vector", list, tuple]) -> "Vector":
        """ Rejection of vector self on vector other.

        Arguments:
            other {Vector} -- The target vector to reject self on

        Returns:
            Vector -- The resultant reject vector
        """
        if not Vector.is_vectorlike:
            raise ValueError("Operand must be vector-like. (Vector, list, or tuple)")
        other = Vector.from_vectorlike(other)
        return self - ((Vector.dot(self, other) / Vector.dot(self, other)) * other)
