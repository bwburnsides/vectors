from math import acos, cos, sin, floor, ceil
from typing import Any, Union, Optional, Tuple


class Vector:
    "A tuple-like class that supports vector operations, similar to numpy arrays."

    vector_like = Union["Vector", tuple, list]
    scalar_like = Union[float, int]

    # TODO: Unify and clean up messages.
    messages = {
        "vector_like_singular": "Operand must be vector-like. ",
        "vector_like_plural": "Operands must be vector-like. ",
        "scalar_like_singular": "Operand must be scalar-like. ",
        "scalar_like_plural": "Operands must be scalar-like. ",
        "scalar_types": "(float, int, bool)",
        "vector_types": "(Vector, tuple, list)",
    }

    def __init__(self, *coords: "Vector.scalar_like") -> None:
        """ Initialize a new Vector() object.

        Arguments:
            *coords {Vector.scalar_like} -- The coordinates of the vector.

        Raises:
            TypeError -- One or more coords were not scalar-like.
        """
        self._coords = coords if coords else (0, 0, 0)
        if not all(Vector.is_scalarlike(el) for el in self._coords):
            raise TypeError(
                "Vector coords must be scalar-like. " + Vector.messages["scalar_types"]
            )

        self._pos = None  # type: Optional[Tuple[float]]
        self._unit = None  # type: Optional[Vector]
        self._mag = None  # type: Optional[float]

        # TODO: Find an more automated way to do this! (descriptor class/ decorator)
        self.cross = self._cross
        self.dot = self._dot
        self.box = self._box
        self.angle = self._angle
        self.comp = self._comp
        self.proj = self._proj
        self.reject = self._reject

    def __getitem__(self, i: Union[slice, int]) -> Union[tuple, float]:
        """ Use the subscript [] operator and with vectors as you would with lists
        and tuples.

        Arguments:
            i {Union[slice, int]} -- Index for element selection.

        Raises:
            IndexError -- Index is greater than length of vector.
            TypeError -- [] accepts only int or slice.

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

        Raises:
            StopIteration -- The end of self.pos has been reached.

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
        """ A tuple of coordinates from Vector self.

        Returns:
            Tuple[float] -- The vector coordinates.
        """
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
            self._unit = self / self.mag
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
            self._mag = (self.dot(self)) ** (1 / 2)
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

        Raises:
            TypeError -- Either `unit` was not vector-like, or mag was not scalar-like.

        Returns:
            Vector -- A vector in the direction of unit and the length of mag.
        """
        if not Vector.is_vectorlike(unit):
            raise TypeError(
                "Arg `unit` must be vector-like. " + Vector.messages["vector_types"]
            )
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
        """ Create a Vector() object from an angle and length.

        Arguments:
            theta {Vector.scalar_like} -- Angle of vector in radians, relative to positive x-axis.

        Raises:
            TypeError -- If either theta or magnitude are not scalar-like.

        Returns:
            Vector -- A new Vector() instance with specified theta and mag.
        """
        if not (Vector.is_scalarlike(theta) and Vector.is_scalarlike(mag)):
            raise TypeError(
                Vector.messages["scalar_like_plural"] + Vector.messages["scalar_types"]
            )
        return cls(float(mag) * cos(theta), float(mag) * sin(theta))

    @staticmethod
    def is_vectorlike(potential: Any) -> bool:
        """ Test for object being vector-like.

        Arguments:
            potential {Any} -- Object to test for vector-likeness.

        Returns:
            bool -- Whether potential is vector-like or not.
        """
        if hasattr(potential, "__iter__") and all(
            Vector.is_scalarlike(element) for element in potential
        ):
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

    @staticmethod
    def pad_vectors(*vecs: "Vector.vector_like") -> Tuple["Vector"]:
        """ Returned vectors will be padded with zeroes such that they all share the same length
            as the vector with the most components.

        Arguments:
            *vecs {Vector.vector_like} -- Vectors to be padded.

        Raises:
            TypeError -- One or more vecs are not vector-like.

        Returns:
            vecs {Vector} -- Padded vectors.
        """
        if not all(Vector.is_vectorlike(vec) for vec in vecs):
            raise TypeError(
                Vector.messages["vector_like_plural"] + Vector.messages["vector_types"]
            )
        vecs = [list(vec) for vec in vecs]
        m = max(len(vec) for vec in vecs)
        vecs = [[vec[i] if i < len(vec) else 0 for i in range(m)] for vec in vecs]
        return (Vector.from_vectorlike(vec) for vec in vecs)

    @staticmethod
    def reduce_vectors(*vecs: "Vector.vector_like") -> Tuple["Vector"]:
        raise NotImplementedError

    def __add__(self, other: "Vector.vector_like") -> "Vector":
        """ Vector addition implementation.

        Arguments:
            other {Vector.vector_like} -- Vector-like argument, second + operand.

        Raises:
            TypeError -- other is not vector-like.

        Returns:
            Vector -- New vector object, sum of self and other.
        """
        if not Vector.is_vectorlike(other):
            raise TypeError(
                Vector.messages["vector_like_singular"]
                + Vector.messages["vector_types"]
            )
        a, b = Vector.pad_vectors(self, other)
        return Vector(*(sum(n) for n in zip(a, b)))

    def __radd__(self, other: "Vector.vector_like") -> "Vector":
        """ Vector (reflexive) addition implementation.

        Arguments:
            other {Vector.vector_like} -- Vector-like argument, first '+' operand.

        Returns:
            Vector -- New vector object, sum of self and other.
        """
        return self + other

    def __iadd__(self, other: "Vector.vector_like") -> "Vector":
        """ Vector addition implementation.

        Arguments:
            other {Vector.vector_like} -- Vector-like argument, second '+=' operand.

        Returns:
            Vector -- New vector object, sum of self and other, assigned to self.
        """
        return self + other

    def __sub__(self, other: "Vector.vector_like") -> "Vector":
        """ Vector subtraction implementation.

        Arguments:
            other {Vector.vector_like} -- second '-' operand.

        Raises:
            TypeError -- other is not vector-like.

        Returns:
            Vector -- new vector object, difference of self and other.
        """
        if not Vector.is_vectorlike(other):
            raise TypeError(
                Vector.messages["vector_like_singular"]
                + Vector.messages["vector_types"]
            )
        a, b = Vector.pad_vectors(self, other)
        return Vector(*(ae - be for ae, be in zip(a, b)))

    def __rsub__(self, other: "Vector.vector_like") -> "Vector":
        """ Vector (reflexive) subtraction implementation.

        Arguments:
            other {Vector.vector_like} -- first '-' operand.

        Returns:
            Vector -- new vector object, difference of other and self.
        """
        return -self + other

    def __isub__(self, other: "Vector.vector_like") -> "Vector":
        """ Vector subtraction implementation.

        Arguments:
            other {Vector.vector_like} -- second '-=' operand.

        Returns:
            Vector -- New vector object, difference of self and other, assigned to self.
        """
        return self - other

    def __mul__(self, scalar: "Vector.scalar_like") -> "Vector":
        """ Vector scalar multiplication implementation.

        Arguments:
            scalar {Vector.scalar_like} -- the second '*' argument.

        Raises:
            TypeError -- `scalar` is not scalar-like.

        Returns:
            Vector -- New vector object, scalar product of self and other.
        """
        if not Vector.is_scalarlike(scalar):
            raise TypeError(
                Vector.messages["scalar_like_singular"]
                + Vector.messages["scalar_types"]
            )
        return Vector(*(el * float(scalar) for el in self))

    def __rmul__(self, scalar: "Vector.scalar_like") -> "Vector":
        """ Vector (reflexive) scalar multiplication implementation.

        Arguments:
            scalar {Vector.scalar_like} -- the first '*' operand.

        Returns:
            Vector -- New vector object, scalar product of other and self.
        """
        return self * scalar

    def __imul__(self, scalar: "Vector.scalar_like") -> "Vector":
        """ Vector scalar multiplication implementation, with assignment.

        Arguments:
            scalar {Vector.scalar_like} -- second '*=' operand.

        Returns:
            Vector - New vector object, scalar product of self and other, assigned to self.
        """
        return self * scalar

    def __truediv__(self, scalar: "Vector.scalar_like") -> "Vector":
        """ An implementation of "vector scalar division" aka vector scalar multiplication
            of the form self * (1/scalar)

        Arguments:
            scalar {Vector.scalar_like} -- The second '/' operand.

        Raises:
            TypeError -- If scalar is not scalar-like.

        Returns:
            Vector -- A new vector as the result of self / scalar.
        """
        if not Vector.is_scalarlike(scalar):
            raise TypeError(
                Vector.messages["scalar_like_singular"]
                + Vector.messages["scalar_types"]
            )
        return (1 / scalar) * self

    def __itruediv__(self, scalar: "Vector.scalar_like") -> "Vector":
        """ An implementation of "vector scalar division" with assignment. See __truediv__.

        Arguments:
            scalar {Vector.scalar_like} -- seoncd '/=' operand.

        Returns:
            Vector -- A new vector, the result of self / scalar, assigned to self.
        """
        return self / scalar

    def __eq__(self, other: Any) -> bool:
        """ Implementation of equality check. Vectors are equivalent if they have
            all the same coords, and are not equal to any other data type.

        Arguments:
            other {Any} -- Another object being checked for equality.

        Returns:
            bool -- Whether self and other are equivalent.
        """
        if not Vector.is_vectorlike(other):
            return False
        return self.pos == Vector.from_vectorlike(other).pos

    def __gt__(self, other: "Vector.vector_like") -> bool:
        """ Vector greater-than comparison. A vector is greater than another
            iff its magnitude is greater.

        Arguments:
            other {Vector.vector_like} -- Another vector being checked for greater-than.

        Raises:
            TypeError -- other is not vector-like.

        Returns:
            bool -- Whether self is greater than other.
        """
        if not Vector.is_vectorlike(other):
            raise TypeError(
                Vector.messages["vector_like_singular"]
                + Vector.messages["vector_types"]
            )
        return self.mag > Vector.from_vectorlike(other).mag

    def __lt__(self, other: "Vector.vector_like") -> bool:
        """ Vector less-than comparison. A vector is less than another
            iff its magnitude is lesser.

        Arguments:
            other {Vector.vector_like} -- Another vector being checked for less-than.

        Raises:
            TypeError -- if other is not vector-like.

        Returns:
            bool -- Whether self is less than other.
        """
        if not Vector.is_vectorlike(other):
            raise TypeError(
                Vector.messages["vector_like_singular"]
                + Vector.messages["vector_types"]
            )
        return self.mag < Vector.from_vectorlike(other).mag

    def __ge__(self, other: "Vector.vector_like") -> bool:
        """ Vector greater-than-or-equal comparison. A vector is greater-than-or-equal to another
            iff its magnitude is greater-than-or-equal.

        Arguments:
            other {Vector.vector_like} -- Another vector being checked for greater-than-or-equal.

        Raises:
            TypeError -- other is not vector-like.

        Returns:
            bool -- Whether self is greater-than-or-equal to other.
        """
        if not Vector.is_vectorlike(other):
            raise TypeError(
                Vector.messages["vector_like_singular"]
                + Vector.messages["vector_types"]
            )
        return self.mag >= Vector.from_vectorlike(other).mag

    def __le__(self, other: "Vector.vector_like") -> bool:
        """ Vector less-than-or-equal comparison. A vector is less-than-or-equal to another
            iff its magnitude is less-than-or-equal.

        Arguments:
            other {Vector.vector_like} -- Another vector being checked for less-than-or-equal.

        Raises:
            TypeError -- other is not vector-like.

        Returns:
            bool -- Whether self is less-than-or-equal to other.
        """
        if not Vector.is_vectorlike(other):
            raise TypeError(
                Vector.messages["vector_like_singular"]
                + Vector.messages["vector_types"]
            )
        return self.mag <= Vector.from_vectorlike(other).mag

    def __bool__(self) -> bool:
        """ Vector boolean conversion. A vector is equal to True if the sum of its
            components is not 0. Otherwise, it is False.

        Returns:
            bool -- The boolean status of vector self.
        """
        return not (sum(self) == 0)

    def __neg__(self) -> "Vector":
        """ Negation implementation.

        Returns:
            Vector -- a new vector == -1 * self
        """
        return Vector(*(-el for el in self))

    def __pos__(self) -> "Vector":
        """ Positation implementation.

        Returns:
            Vector -- a new vector == 1 * self
        """
        return Vector(*(+el for el in self))

    def __abs__(self) -> "Vector":
        """ abs() implementation.

        Returns:
            Vector -- a new vector whose elements are the absolute value of the elements of self.
        """
        return Vector(*(abs(el) for el in self))

    def __floor__(self) -> "Vector":
        """ math.floor() implementation.

        Returns
            Vector -- a new vector with all elements rounded down to the closest integer (as floats.)
        """
        return Vector(*(float(floor(el)) for el in self))

    def __ceil__(self) -> "Vector":
        """ math.ceil() implementation.

        Returns
            Vector -- a new vector with all elements rounded up to the closest integer (as floats.)
        """
        return Vector(*(float(ceil(el)) for el in self))

    def __round__(self, n: int) -> "Vector":
        """ round() implementation.

        Arguments:
            n {int} -- number of decimals to round components of self to.

        Raises:
            TypeError -- if n isn't an int
            ValueError -- if n < 0

        Returns:
            Vector -- a new vector with all elements rounded to n decimal places.
        """
        if not isinstance(n, int):
            raise TypeError("n must be an int.")
        if n < 0:
            raise ValueError("n must be >= 0.")
        return Vector(*(float(round(el, n)) for el in self))

    def __complex__(self) -> complex:
        """ complex() implementation. Takes the first two dimensions of a vector.
            If there are less than two dimensions, they are padded with zeros.

        Returns:
            complex -- A complex-number representation of vector self.
        """
        v = [0, 0][0 : len(self)] = self[0:2]
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

        Raises:
            TypeError -- at least a or b not vector-like

        Returns:
            float -- the dot product between a and b.
        """
        if not (Vector.is_vectorlike(a) and Vector.is_vectorlike(b)):
            raise TypeError(
                Vector.messages["vector_like_plural"] + Vector.messages["vector_types"]
            )
        return float(sum(x * y for x, y in zip(a.pos, b.pos)))

    def _dot(self, other: "Vector.vector_like") -> float:
        "Instance-method implementation of dot product."
        return Vector.dot(self, other)

    @staticmethod
    def cross(  # pylint: disable=method-hidden
        a: "Vector.vector_like", b: "Vector.vector_like"
    ) -> "Vector":
        """ Calculate the cross product between two vectors in R3. If a or b are of
            higher dimensions, they will be reduced to their first three elements. If they
            are in smaller dimensions, they will be padded with 0s.

        Arguments:
            a {Vector.vector_like} -- Vector 1
            b {Vector.vector_like} -- Vector 2

        Raises:
            TypeError -- a or b not vector-like.

        Returns:
            Vector -- A vector orthogonal to a and b
        """
        if not (Vector.is_vectorlike(a) and Vector.is_vectorlike(b)):
            raise TypeError(
                Vector.messages["vector_like_plural"] + Vector.messages["vector_types"]
            )

        v1, v2 = [[0, 0, 0] for i in range(2)]
        v1[0 : len(a)] = a[0:3]
        v2[0 : len(b)] = b[0:3]

        return Vector(
            (v1[1] * v2[2]) - (v1[2] * v2[1]),
            (v1[2] * v2[0]) - (v1[0] * v2[2]),
            (v1[0] * v2[1]) - (v1[1] * v2[0]),
        )

    def _cross(self, other: "Vector.vector_like") -> "Vector":
        "Instance-method implementation of cross product."
        return Vector.cross(self, other)

    @staticmethod
    def box(  # pylint: disable=method-hidden
        a: "Vector.vector_like", b: "Vector.vector_like", c: "Vector.vector_like"
    ) -> float:
        """ Vector scalar triple product, or box product

        Arguments:
            a {Vector.vector_like} -- Vector 1
            b {Vector.vector_like} -- Vector 2
            c {Vector.vector_like} -- Vector 3

        Raises:
            TypeError -- a, b, or c not vector-like.

        Returns:
            float -- scalar triple product of a, b, c.
        """
        if not (
            Vector.is_vectorlike(a)
            and Vector.is_vectorlike(b)
            and Vector.vector_like(c)
        ):
            raise TypeError(
                Vector.messages["vector_like_plural"] + Vector.messages["vector_types"]
            )
        a = Vector.from_vectorlike(a)
        b = Vector.from_vectorlike(b)
        c = Vector.from_vectorlike(c)
        return a.dot(b.cross(c))

    def _box(self, *vecs: "Vector.vector_like") -> float:
        "Instance-method implementation of scalar triple product."
        return Vector.box(self, vecs[0], vecs[1])

    @staticmethod
    def angle(  # pylint: disable=method-hidden
        a: "Vector.vector_like", b: "Vector.vector_like"
    ) -> float:
        """ Determine the angle between two vectors.

        Arguments:
            a {Vector.vector_like} -- Vector 1
            b {Vector.vector_like} -- Vector 2

        Raises:
            TypeError -- a or b not vector-like.

        Returns:
            float -- The angle between a, b in radians.
        """
        if not (Vector.is_vectorlike(a) and Vector.is_vectorlike(b)):
            raise TypeError(
                Vector.messages["vector_like_plural"] + Vector.messages["vector_types"]
            )
        a, b = Vector.from_vectorlike(a), Vector.from_vectorlike(b)
        return acos((a.dot(b)) / (a.mag * b.mag))

    def _angle(self, other: "Vector.vector_like") -> float:
        "Instance-method implementation of angle-between."
        return Vector.angle(self, other)

    @staticmethod
    def comp(  # pylint: disable=method-hidden
        a: "Vector.vector_like", b: "Vector.vector_like"
    ) -> float:
        """ Scalar projection of vector a on vector b.

        Arguments:
            a {Vector.vector_like} -- The base vector to project.
            b {Vector.vector_like} -- The target vector to project self on.

        Raises:
            TypeError -- a or b not vector-like.

        Returns:
            float -- The resultant scalar.
        """
        if not (Vector.is_vectorlike(a) and Vector.is_vectorlike(b)):
            raise TypeError(
                Vector.messages["vector_like_plural"] + Vector.messages["vector_types"]
            )
        a, b = Vector.from_vectorlike(a), Vector.from_vectorlike(b)
        return a.dot(a) / b.mag

    def _comp(self, other: "Vector.vector_like") -> float:
        "Instance-method implementation of scalar projection."
        return Vector.comp(self, other)

    @staticmethod
    def proj(  # pylint: disable=method-hidden
        a: "Vector.vector_like", b: "Vector.vector_like"
    ) -> "Vector":
        """ Projection of vector a on vector b.

        Arguments:
            a {Vector.vector_like} -- The base vector to project.
            b {Vector.vector_like} -- The target vector to project self on.

        Raises:
            TypeError -- a or b not vector-like.

        Returns:
            Vector -- The resultant projected vector.
        """
        if not (Vector.is_vectorlike(a) and Vector.is_vectorlike(b)):
            raise TypeError(
                Vector.messages["vector_like_plural"] + Vector.messages["vector_types"]
            )
        a, b = Vector.from_vectorlike(a), Vector.from_vectorlike(b)
        return a.dot(b.unit) * b.unit

    def _proj(self, other: "Vector.vector_like") -> "Vector":
        "Instance-method implementation of vector projection."
        return Vector.proj(self, other)

    @staticmethod
    def reject(  # pylint: disable=method-hidden
        a: "Vector.vector_like", b: "Vector.vector_like"
    ) -> "Vector":
        """ Rejection of vector a on vector b.

        Arguments:
            a {Vector.vector_like} -- The base vector to reject.
            b {Vector.vector_like} -- The target vector to reject self on.

        Raises:
            TypeError -- a or b not vector-like.

        Returns:
            Vector -- The resultant reject vector
        """
        if not (Vector.is_vectorlike(a) and Vector.is_vectorlike(b)):
            raise TypeError(
                Vector.messages["vector_like_plural"] + Vector.messages["vector_types"]
            )
        a, b = Vector.from_vectorlike(a), Vector.from_vectorlike(b)
        return a - ((a.dot(b) / b.dot(b)) * b)

    def _reject(self, other: "Vector.vector_like") -> "Vector":
        "Instance-method implementation of vector rejection."
        return Vector.reject(self, other)
