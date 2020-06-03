from math import acos, cos, sin, floor, ceil
from typing import Any, Union, Optional, Tuple


class Vector:
    """ A tuple-like class that supports vector operations, similar to numpy arrays.
    """

    vector_like = Union["Vector", tuple, list]
    scalar_like = Union[float, int]

    def __init__(self, *coords: "Vector.scalar_like"):
        self._coords = coords if coords else (0, 0, 0)
        if not all(Vector.is_scalarlike(el) for el in self._coords):
            raise TypeError("Vector coords must be of int or float type.")

        self._pos = None  # type: Optional[Tuple[float]]
        self._unit = None  # type: Optional[Vector]
        self._mag = None  # type: Optional[float]

        self.cross = self._cross
        self.dot = self._cross

    def __getitem__(self, i: Union[slice, int]) -> Union[tuple, float]:
        """ Use the subscript [] operator and with vectors as you would with lists
        and tuples.

        Arguments:
            i {Union[slice, int]} -- Index for element selection.

        Returns:
            float -- Element at selected index.
        """
        if isinstance(i, slice):
            return self.pos[i.start : i.stop : i.step]
        elif isinstance(i, int):
            if i < len(self):
                return self.pos[i]
            raise IndexError("Index '{}' exceeds vector length.".format(i))
        else:
            raise TypeError("Operand must have type int or slice.")

    def __len__(self) -> int:
        """ Use the len() function on vectors as you lists and tuples.

        Returns:
            int -- The length (dimension) of the vector self.
        """
        return len(self.pos)

    def __repr__(self) -> str:
        """ An unambiguous representation for vector self printing. (useful for debugging)

        Returns:
            str -- vector representation
        """
        return "Vector: " + self.__str__()

    def __str__(self) -> str:
        """ A simple string representation of vector self.

        Returns:
            str -- vector self as a string.
        """
        return str(self.pos)

    def __iter__(self) -> "Vector":
        """ Resets iteration; needed for Iterator Protocol.

        Returns:
            Vector -- self
        """
        self.n = 0
        return self

    def __next__(self) -> float:
        """ Defines Vector iterator behavior by stepping through vector dimensions.

        Returns:
            float -- Vector value on dimension self.n.
        """
        if self.n < len(self):
            result = self[self.n]
            self.n += 1
            return result
        else:
            raise StopIteration

    @property
    def pos(self) -> Tuple[float]:
        if not self._pos:
            self._pos = tuple([float(pt) for pt in self._coords])
            del self._coords
        return self._pos

    @pos.setter
    def pos(self, new_val: "Tuple[Vector.scalar_like]") -> None:
        raise AttributeError("Vectors are immutable.")

    @property
    def unit(self) -> "Vector":
        """ Vector in the direction of self with a magnitude of 1.

        Returns:
            Vector -- Unit vector in the direction of self.
        """
        if not self._unit:
            self._unit = Vector(*(el / self.mag for el in self.pos))
        return self._unit

    @unit.setter
    def unit(self, new_val: "Vector.vector_like") -> None:
        raise AttributeError("Vectors are immutable.")

    @property
    def mag(self) -> float:
        """ Length of the vector self.

        Returns:
            float -- the magnitude of the vector.
        """
        if not self._mag:
            self._mag = sum(el ** 2 for el in self.pos) ** (1 / 2)
        return self._mag

    @mag.setter
    def mag(self, new_val: "Vector.scalar_like") -> None:
        raise AttributeError("Vectors are immutable.")

    @classmethod
    def from_unit(
        cls, unit: "Vector.vector_like", mag: "Vector.scalar_like"
    ) -> "Vector":
        """ Create a Vector instance by specifying a direction and length.

        Arguments:
            unit {Vector.vector_like} -- The direction of desired vector.
                                        If not of mag 1, will be internally reduced to 1.
            mag {Vector.scalar_like} -- The length of the desired vector.

        Returns:
            Vector -- A vector in the direction of unit and the length of mag.
        """
        if not Vector.is_vectorlike(unit):
            raise TypeError("Arg unit must be vector-like.")
        unit = Vector.from_vectorlike(unit).unit.pos
        return cls(*(mag * el for el in unit))

    @classmethod
    def from_vectorlike(cls, vectorlike: "Vector.vector_like") -> "Vector":
        """ Create a vector instance from a vector-like object.

        Arguments:
            vectorlike {vector.vector_like} -- Vector-like to base Vector on.

        Returns:
            Vector -- A vector with the same coordinates as vectorlike.
        """
        return cls(*vectorlike)

    @classmethod
    def from_angle(
        cls, theta: "Vector.scalar_like", mag: "Vector.scalar_like"
    ) -> "Vector":
        if not Vector.is_scalarlike(theta) or not Vector.is_scalarlike(mag):
            raise TypeError("Operands must be scalar-like. (int or float)")
        return cls(float(mag) * cos(theta), float(mag) * sin(theta))

    @staticmethod
    def is_vectorlike(potential: Any) -> bool:
        """ Test for object being vector-like.

        Arguments:
            potential {Any} -- Object to test for vector-likeness.

        Returns:
            bool -- Whether potential is vector-like or not.
        """
        if all(Vector.is_scalarlike(element) for element in potential):
            return True
        return False

    @staticmethod
    def is_scalarlike(potential: Any) -> bool:
        """ Test for object being scalar-like.

        Arguments:
            potential {Any} -- Object to test for scalar-likeness.

        Returns:
            bool -- Whether potential is scalar-like or not.
        """
        if isinstance(potential, int) or isinstance(potential, float):
            return True
        return False

    def __add__(self, other: "Vector.vector_like") -> "Vector":
        """ Vector addition implementation.

        Arguments:
            other {Vector.vector_like} -- Vector-like argument, second + operand.

        Returns:
            Vector -- New vector object, sum of self and other.
        """
        if not Vector.is_vectorlike(other):
            raise TypeError("Operand must be vector-like. (Vector, list, or tuple)")
        return Vector(*(se + oe for se, oe in zip(self, other)))

    def __radd__(self, other: "Vector.vector_like") -> "Vector":
        """ Vector (reflexive) addition implementation.

        Arguments:
            other {Vector.vector_like} -- Vector-like argument, first + operand.

        Returns:
            Vector -- New vector object, sum of self and other.
        """
        if not Vector.is_vectorlike(other):
            raise TypeError("Operand must be vector-like. (Vector, list, or tuple)")
        return self + other

    def __iadd__(self, other: "Vector.vector_like") -> "Vector":
        """ Vector addition implementation.

        Returns:
            Vector -- New vector object, sum of self and other, assigned to self.
        """
        if not Vector.is_vectorlike(other):
            raise TypeError("Operand must be vector-like. (Vector, list, or tuple)")
        return self + other

    def __sub__(self, other: "Vector.vector_like") -> "Vector":
        """ Vector subtraction implementation.

        Returns:
            Vector -- New vector object, difference of self and other.
        """
        if not Vector.is_vectorlike(other):
            raise TypeError("Operand must be vector-like. (Vector, list, or tuple)")
        return Vector(*(se - oe for se, oe in zip(self, other)))

    def __rsub__(self, other: "Vector.vector_like") -> "Vector":
        """ Vector (reflexive) subtraction implementation.

        Returns:
            Vector -- New vector object, difference of other and self.
        """
        if not Vector.is_vectorlike(other):
            raise TypeError("Operand must be vector-like. (Vector, list, or tuple)")
        return -self + other

    def __isub__(self, other: "Vector.vector_like") -> "Vector":
        """ Vector subtraction implementation.

        Returns:
            Vector -- New vector object, difference of self and other, assigned to self.
        """
        if not Vector.is_vectorlike(other):
            raise TypeError("Operand must be vector-like. (Vector, list, or tuple)")
        return self - other

    def __mul__(self, scalar: "Vector.scalar_like") -> "Vector":
        """ Vector scalar multiplication implementation.

        Returns:
            Vector -- New vector object, scalar product of self and other.
        """
        if not Vector.is_scalarlike(scalar):
            raise TypeError("Operand must be scalar-like. (float or int)")
        return Vector(*(el * scalar for el in self))

    def __rmul__(self, scalar: "Vector.scalar_like") -> "Vector":
        """ Vector (reflexive) scalar multiplication implementation.

        Returns:
            Vector -- New vector object, scalar product of other and self.
        """
        if not Vector.is_scalarlike(scalar):
            raise TypeError("Operand must be scalar-like. (float or int)")
        return self * scalar

    def __imul__(self, scalar: "Vector.scalar_like") -> "Vector":
        """ Vector scalar multiplication implementation, with assignment.

        Returns:
            Vector - New vector object, scalar product of self and other, assigned to self.
        """
        if not Vector.is_scalarlike(scalar):
            raise TypeError("Operand must be scalar-like. (float or int)")
        return self * scalar

    def __neg__(self):
        """ Negation implementation.

        Returns:
            Vector -- a new vector == -1 * self
        """
        return Vector(*(-el for el in self))

    def __pos__(self):
        """ Positation implementation.

        Returns:
            Vector -- a new vector == 1 * self
        """
        return Vector(*(+el for el in self))

    def __abs__(self):
        """ abs() implementation.

        Returns:
            Vector -- a new vector whose elements are the absolute value of the elements of self.
        """
        return Vector(*(abs(el) for el in self))

    def __floor__(self):
        """ math.floor() implementation.

        Returns
            Vector -- a new vector with all elements rounded down to the closest integer (as floats.)
        """
        return Vector(*(float(floor(el)) for el in self))

    def __ceil__(self):
        """ math.ceil() implementation.

        Returns
            Vector -- a new vector with all elements rounded up to the closest integer (as floats.)
        """
        return Vector(*(float(ceil(el)) for el in self))

    def __round__(self, n):
        """ round() implementation.

        Returns
            Vector -- a new vector with all elements rounded to n decimal places.
        """
        return Vector(*(float(round(el, n)) for el in self))

    def __complex__(self):
        v = [0, 0]
        v[0 : len(self)] = self[0:2]
        return complex(v[0], v[1])

    @staticmethod
    def dot(  # pylint: disable=method-hidden
        a: "Vector.vector_like", b: "Vector.vector_like"
    ) -> float:
        """ Calculate the dot product between two vectors. If a and b are different lengths,
            summing stops at the end of the shorter operand.

        Arguments:
            a {Vector.vector_like} -- Vector 1
            b {Vector.vector_like} -- Vector 2

        Returns:
            float -- the dot product between a and b.
        """
        if not Vector.is_vectorlike(a) and Vector.is_vectorlike(b):
            raise TypeError("Operands must be vector-like. (Vector, list, or tuple)")
        return float(sum(ae * be for ae, be in zip(a, b)))

    def _dot(self, other: "Vector.vector_like") -> "Vector.vector_like":
        return Vector.dot(self, other)

    @staticmethod
    def cross(  # pylint: disable=method-hidden
        a: "Vector.vector_like", b: Union[None, "Vector.vector_like"]
    ) -> "Vector":
        """ Calculate the cross product between two vectors in R3. If a or b are of
            higher dimensions, they will be reduced to their first three elements. If they
            are in smaller dimensions, they will be padded with 0s.

        Arguments:
            a {Vector.vector_like} -- Vector 1
            b {Vector.vector_like} -- Vector 2

        Returns:
            Vector -- A vector orthogonal to a and b
        """
        if not Vector.is_vectorlike(a) and Vector.is_vectorlike(b):
            raise TypeError("Operands must be vector-like. (Vector, list, or tuple)")

        v1, v2 = [[0, 0, 0] for i in range(2)]
        v1[0 : len(a)] = a[0:3]
        v2[0 : len(b)] = b[0:3]

        return Vector(
            (v1[1] * v2[2]) - (v1[2] * v2[1]),
            (v1[2] * v2[0]) - (v1[0] * v2[2]),
            (v1[0] * v2[1]) - (v1[1] * v2[0]),
        )

    def _cross(self, other: "Vector.vector_like") -> "Vector.vector_like":
        return Vector.cross(self, other)

    @staticmethod
    def angle(a: "Vector.vector_like", b: "Vector.vector_like") -> float:
        """ Determine the angle between two vectors.

        Arguments:
            a {Vector.vector_like} -- Vector 1
            b {Vector.vector_like} -- Vector 2

        Returns:
            float -- The angle between a, b in radians.
        """
        if not Vector.is_vectorlike(a) and Vector.is_vectorlike(b):
            raise TypeError("Operands must be vector-like. (Vector, list, or tuple)")
        a, b = Vector.from_vectorlike(a), Vector.from_vectorlike(b)
        return acos((Vector.dot(a, b)) / (a.mag * b.mag))

    def comp(self, other: "Vector.vector_like") -> float:
        """ Scalar projection of vector self on vector other.

        Arguments:
            other {Vector.vector_like} -- The target vector to project self on.

        Returns:
            float -- The resultant scalar.
        """
        if not Vector.is_vectorlike(other):
            raise TypeError("Operand must be vector-like. (Vector, list, or tuple)")
        other = Vector.from_vectorlike(other)
        return Vector.dot(self, other) / other.mag

    def proj(self, other: "Vector.vector_like") -> "Vector":
        """ Projection of vector self on vector other.

        Arguments:
            other {Vector.vector_like} -- The target vector to project self on.

        Returns:
            Vector -- The resultant projected vector.
        """
        if not Vector.is_vectorlike(other):
            raise TypeError("Operand must be vector-like. (Vector, list, or tuple)")
        other = Vector.from_vectorlike(other)
        return Vector.dot(self, other.unit) * other.unit

    def reject(self, other: "Vector.vector_like") -> "Vector":
        """ Rejection of vector self on vector other.

        Arguments:
            other {Vector.vector_like} -- The target vector to reject self on

        Returns:
            Vector -- The resultant reject vector
        """
        if not Vector.is_vectorlike:
            raise TypeError("Operand must be vector-like. (Vector, list, or tuple)")
        other = Vector.from_vectorlike(other)
        return self - ((Vector.dot(self, other) / Vector.dot(self, other)) * other)
