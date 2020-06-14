"""Contains Vector() class for doing vector operations easily.

Recommend importing as such, for terse initialization:
    >>> from vector import Vector as v
"""

from math import acos, cos, sin, floor, ceil
from typing import TypeVar, Any, Union, Optional, List, Tuple, Callable, overload, cast


class Vector:
    """A `tuple`-like class that supports vector operations, similar to numpy arrays."""

    vector_like = Union["Vector", tuple, list]
    scalar_like = Union[float, int]
    arbitrary_signature = TypeVar("arbitrary_signature", bound=Callable[..., Any])

    # TODO: Unify and clean up messages.
    messages = {
        "vector_like_singular": "Operand must be vector-like. ",
        "vector_like_plural": "Operands must be vector-like. ",
        "scalar_like_singular": "Operand must be scalar-like. ",
        "scalar_like_plural": "Operands must be scalar-like. ",
        "scalar_types": "(float, int, bool)",
        "vector_types": "(Vector, tuple, list)",
    }

    class VectorDecorators:
        """Namespace decorators for methods in Vector() in an attempt to DRY up code."""

        @staticmethod
        def check_for_veclike(
            func: "Vector.arbitrary_signature",
        ) -> "Vector.arbitrary_signature":
            """Handle checking of args for veclikeness, and raising TypeErrors.

            Arguments:
                func {Vector() method}: function for arg validating
            """

            def wrapper(*args, **kwargs):
                # TODO: figure out how to generalize this for many methods with differing kind of
                # inputs: ie - 1 vectorlike, 2+ vectorlikes, vectorlike and a scalar, etc
                return func(*args, **kwargs)

            return cast("Vector.arbitrary_signature", wrapper)

    def __init__(self, *coords: "Vector.scalar_like") -> None:
        """Initialize a new `Vector()` object.

        Arguments:
            `*coords` {`Vector.scalar_like`} -- The coordinates of the vector.

        Raises:
            `TypeError` -- One or more coords were not scalar-like.

        Examples:
            >>> a = Vector(1, 2, 3)
            >>> a
            <1.0, 2.0, 3.0>
        """
        self._coords = coords if coords else tuple()
        if not all(Vector.is_scalarlike(el) for el in self._coords):
            raise TypeError(
                "Vector coords must be scalar-like. " + Vector.messages["scalar_types"]
            )

        self._pos: Optional[Tuple[float, ...]] = None
        self._pos_rounded: Optional[Tuple[float, ...]] = None
        self._unit: Optional[Vector] = None
        self._mag: Optional[float] = None

        # TODO: Find a more automated way to do this! (descriptor class/ decorator)
        self.cross = self._cross  # type: ignore
        self.dot = self._dot  # type: ignore
        self.box = self._box  # type: ignore
        self.angle = self._angle  # type: ignore
        self.comp = self._comp  # type: ignore
        self.proj = self._proj  # type: ignore
        self.reject = self._reject  # type: ignore

    @classmethod
    def from_unit(
        cls, unit: "Vector.vector_like", mag: "Vector.scalar_like"
    ) -> "Vector":
        """Create a `Vector()` instance by specifying a direction and length.

        Arguments:
            `unit` {`Vector.vector_like`} -- The direction of desired vector.
                                            If not of mag 1, will be internally reduced to 1.
            `mag` {`Vector.scalar_like`} -- The length of the desired vector.

        Raises:
            `TypeError` -- Either `unit` was not vector-like, or `mag` was not scalar-like.

        Returns:
            `Vector` -- A vector in the direction of `unit` and the length of `mag`.

        Examples:
            >>> a = Vector.from_unit((0, 1), 5)
            >>> b = Vector.from_unit(Vector(0, 1), 5)
            >>> c = Vector.from_unit([0, 1], 5)
            >>> a == b == c
            True
        """
        if not Vector.is_vectorlike(unit):
            raise TypeError(
                "Arg `unit` must be vector-like. " + Vector.messages["vector_types"]
            )
        unit = Vector.from_vectorlike(unit).unit.pos
        return cls(*(mag * el for el in unit))

    @classmethod
    def from_vectorlike(cls, vectorlike: "Vector.vector_like") -> "Vector":
        """Create a `Vector()` instance from a vector-like object.

        Arguments:
            `vectorlike` {`vector.vector_like`} -- Vector-like to base `Vector` on.

        Raises:
            `TypeError` -- `vectorlike` is not vector-like

        Returns:
            `Vector` -- A vector with the same coordinates as `vectorlike`.

        Examples:
            >>> a = Vector.from_vectorlike([1, 1, 1])
            >>> b = Vector.from_vectorlike((1, 1, 1))
            >>> a == b == Vector(1, 1, 1)
            True
        """
        if not Vector.is_vectorlike(vectorlike):
            raise TypeError(
                Vector.messages["vector_like_plural"] + Vector.messages["vector_types"]
            )
        return cls(*vectorlike)

    @classmethod
    def from_angle(
        cls, theta: "Vector.scalar_like", mag: "Vector.scalar_like"
    ) -> "Vector":
        """Create a `Vector()` object from an angle and length.

        Arguments:
            `theta` {`Vector.scalar_like`} -- Angle of vector in [radians]
                                                relative to positive x-axis.

        Raises:
            `TypeError` -- If either `theta` or `mag` are not scalar-like.

        Returns:
            `Vector` -- A new `Vector()` instance with specified `theta` and `mag`.

        Examples:
        >>> from math import pi
        >>> a = Vector.from_angle(pi / 2, 1)
        >>> a
        <0.0, 1.0>
        """
        if not (Vector.is_scalarlike(theta) and Vector.is_scalarlike(mag)):
            raise TypeError(
                Vector.messages["scalar_like_plural"] + Vector.messages["scalar_types"]
            )
        return cls(float(mag) * cos(theta), float(mag) * sin(theta))

    @overload
    def __getitem__(self, i: int) -> float:
        """Indexing method signature for index subscripts."""
        ...

    @overload
    def __getitem__(self, i: slice) -> Tuple[float, ...]:
        """Indexing method signature for slice subscripts."""
        ...

    def __getitem__(self, i: Union[slice, int]) -> Union[Tuple[float, ...], float]:
        """Use the subscript `[]` operator and with vectors as you would with `list`s and `tuple`s.

        Arguments:
            `i` {`Union[slice, int]`} -- Index for element selection.

        Raises:
            `IndexError` -- Index is greater than length of vector.
            `NotImplementedError` -- `tuple` indexing not supported.
            `TypeError` -- `[]` accepts only int or slice.

        Returns:
            `float` -- Element at selected index.

        Example:
            >>> a = Vector(1, 2, 3)
            >>> a[0], a[-1]
            (1.0, 3.0)
            >>> a[0:2]
            (1.0, 2.0)
        """
        if isinstance(i, slice):
            return tuple([self[j] for j in range(*i.indices(len(self)))])
        elif isinstance(i, int):
            if i < len(self):
                return self.pos[i]
            raise IndexError(f"Index '{i}' exceeds vector length.")
        elif isinstance(i, tuple):
            raise NotImplementedError(
                "Indexing by tuple not supported by vector class."
            )
        else:
            raise TypeError("Operand must have type int or slice.")

    def __len__(self) -> int:
        """Use the `len()` function on vectors as you `list`s and `tuple`s.

        Returns:
            `int` -- The length (dimension) of the vector `self`.

        Examples:
            >>> a = Vector(1, 1, 1)
            >>> len(a)
            3
        """
        return len(self.pos)

    def __repr__(self) -> str:
        """Return an unambiguous representation for vector self printing (useful for debugging).

        Returns:
            `str` -- vector representation

        Examples:
            >>> a = Vector(1, 1, 1)
            >>> repr(a)
            '<1.0, 1.0, 1.0>'
        """
        return str(self)

    def __str__(self) -> str:
        """Return an simple string representation of vector self.

        Returns:
            `str` -- vector self as a string.

        Examples:
            >>> a = Vector(1, 1, 1)
            >>> str(a)
            '<1.0, 1.0, 1.0>'
        """
        if not self.pos:
            self.pos
        return str(self._pos_rounded).replace("(", "<").replace(")", ">")

    def __iter__(self) -> "Vector":
        """Reset iteration; needed for Iterator Protocol.

        Returns:
            `Vector` -- `self`
        """
        self.n = 0
        return self

    def __next__(self) -> float:
        """Define `Vector()` iterator behavior by stepping through vector dimensions.

        Raises:
            `StopIteration` -- The end of `self.pos` has been reached.

        Returns:
            `float` -- Vector value on dimension `self.n`.
        """
        if self.n < len(self):
            result = self[self.n]
            self.n += 1
            return result
        else:
            raise StopIteration

    @property
    def pos(self) -> Tuple[float, ...]:
        """Return a `tuple` of coordinates from `Vector` `self`.

        Returns:
            `Tuple[float, ...]` -- The vector coordinates.

        Examples:
            >>> a = Vector(1, 2, 3)
            >>> a.pos
            (1.0, 2.0, 3.0)
        """
        if not self._pos:
            self._pos = tuple([float(pt) for pt in self._coords])
            self._pos_rounded = tuple([round(pt, 3) for pt in self._pos])
        return self._pos

    @pos.setter
    def pos(self, new_val: Tuple["Vector.scalar_like", ...]) -> None:
        raise AttributeError("Vectors are immutable.")

    @property
    def unit(self) -> "Vector":
        """`Vector` in the direction of `self` with a magnitude of 1.

        Returns:
            `Vector` -- Unit vector in the direction of self.

        Examples:
            >>> a = Vector(5, 0)
            >>> a.unit
            <1.0, 0.0>
        """
        if not self._unit:
            try:
                self._unit = self / self.mag
            except ZeroDivisionError:
                self._unit = self * 0
        return self._unit

    @unit.setter
    def unit(self, new_val: "Vector.vector_like") -> None:
        raise AttributeError("Vectors are immutable.")

    @property
    def mag(self) -> float:
        """Length of the vector `self`.

        Returns:
            `float` -- the magnitude of the vector.

        Examples:
            >>> Vector(5, 0).mag
            5.0
        """
        if not self._mag:
            self._mag = (Vector.dot(self, self)) ** (1 / 2)
        return self._mag

    @mag.setter
    def mag(self, new_val: "Vector.scalar_like") -> None:
        raise AttributeError("Vectors are immutable.")

    def __eq__(self, other: Any) -> bool:
        """Perfoms an equality check.

            Vectors are equivalent to other vector-likes that have all the
            same elements. They are not equal to any other data types.

        Arguments:
            `other` {`Any`} -- Another object being checked for equality.

        Returns:
            `bool` -- Whether `self` and `other` are equivalent.

        Examples:
            >>> a = Vector(1, 1, 1)
            >>> b = [1, 1, 1]
            >>> c = Vector(1, 2, 3)
            >>> a == b
            True
            >>> a == c
            False
        """
        if not Vector.is_vectorlike(other):
            return False
        return self.pos == Vector.from_vectorlike(other).pos

    @VectorDecorators.check_for_veclike
    def __gt__(self, other: "Vector.vector_like") -> bool:
        """`Vector()` greater-than comparison.

            A vector is greater than another iff its magnitude is greater.

        Arguments:
            `other` {`Vector.vector_like`} -- Another vector being checked for greater-than.

        Raises:
            `TypeError` -- `other` is not vector-like.

        Returns:
            `bool` -- Whether `self` is greater than `other`.

        Examples:
            >>> Vector(0, 5) > Vector(1, 0)
            True
        """
        if not Vector.is_vectorlike(other):
            raise TypeError(
                Vector.messages["vector_like_singular"]
                + Vector.messages["vector_types"]
            )
        return self.mag > Vector.from_vectorlike(other).mag

    # @VectorDecorators.check_for_veclike  # TODO: why does mypy not like this decorator?
    def __lt__(self, other: "Vector.vector_like") -> bool:
        """`Vector()` less-than comparison.

            A vector is less than another iff its magnitude is lesser.

        Arguments:
            `other` {`Vector.vector_like`} -- Another vector being checked for less-than.

        Raises:
            `TypeError` -- if other is not vector-like.

        Returns:
            `bool` -- Whether self is less than other.

        Examples:
            >>> Vector(1, 0) < Vector(0, 5)
            True
        """
        if not Vector.is_vectorlike(other):
            raise TypeError(
                Vector.messages["vector_like_singular"]
                + Vector.messages["vector_types"]
            )
        return self.mag < Vector.from_vectorlike(other).mag

    @VectorDecorators.check_for_veclike
    def __ge__(self, other: "Vector.vector_like") -> bool:
        """`Vector()` greater-than-or-equal comparison.

            A vector is greater-than-or-equal to another ...
            iff its magnitude is greater-than-or-equal.

        Arguments:
            `other` {`Vector.vector_like`} -- Another vector being checked for
                                            greater-than-or-equal.

        Raises:
            `TypeError` -- `other` is not vector-like.

        Returns:
            `bool` -- Whether `self` is greater-than-or-equal to `other`.

        Examples:
            >>> Vector(1, 0) >= Vector(0, 1)
            True
        """
        if not Vector.is_vectorlike(other):
            raise TypeError(
                Vector.messages["vector_like_singular"]
                + Vector.messages["vector_types"]
            )
        return self.mag >= Vector.from_vectorlike(other).mag

    # @VectorDecorators.check_for_veclike  # TODO: why does mypy not like this decorator?
    def __le__(self, other: "Vector.vector_like") -> bool:
        """`Vector()` less-than-or-equal comparison.

            A vector is less-than-or-equal to another iff its magnitude is less-than-or-equal.

        Arguments:
            `other` {`Vector.vector_like`} -- Another vector being checked for less-than-or-equal.

        Raises:
            `TypeError` -- other is not vector-like.

        Returns:
            `bool` -- Whether `self` is less-than-or-equal to `other`.

        Examples:
            >>> Vector(1, 0) <= Vector(0, 1)
            True
        """
        if not Vector.is_vectorlike(other):
            raise TypeError(
                Vector.messages["vector_like_singular"]
                + Vector.messages["vector_types"]
            )
        return self.mag <= Vector.from_vectorlike(other).mag

    @VectorDecorators.check_for_veclike
    def __add__(self, other: "Vector.vector_like") -> "Vector":
        """`Vector()` addition implementation.

        Arguments:
            `other` {`Vector.vector_like`} -- Vector-like argument, second `+` operand.

        Raises:
            `TypeError` -- `other` is not vector-like.

        Returns:
            `Vector` -- New vector object, sum of `self` and `other`.

        Examples:
            >>> a = Vector(1, 2, 3)
            >>> b = Vector(4, 5, 6)
            >>> c = [7, 8]
            >>> a + b
            <5.0, 7.0, 9.0>
            >>> b + c
            <11.0, 13.0, 6.0>
        """
        if not Vector.is_vectorlike(other):
            raise TypeError(
                Vector.messages["vector_like_singular"]
                + Vector.messages["vector_types"]
            )
        a, b = Vector.to_dimension(max(len(self), len(other)), self, other)
        return Vector(*(sum(n) for n in zip(a.pos, b.pos)))

    @VectorDecorators.check_for_veclike
    def __radd__(self, other: "Vector.vector_like") -> "Vector":
        """`Vector()` (reflexive) addition implementation.

        Arguments:
            `other` {`Vector.vector_like`} -- Vector-like argument, first `+` operand.

        Returns:
            `Vector` -- New vector, sum of `self` and `other`.

        Examples:
            >>> [1, 2, 3] + Vector(2, 5, 6)
            <3.0, 7.0, 9.0>
        """
        return self + other

    @VectorDecorators.check_for_veclike
    def __iadd__(self, other: "Vector.vector_like") -> "Vector":
        """`Vector()` addition implementation.

        Arguments:
            `other` {`Vector.vector_like`} -- Vector-like argument, second `+=` operand.

        Returns:
            `Vector` -- New vector object, sum of `self` and `other`, assigned to `self`.

        Examples:
            >>> a = Vector(1, 2, 3)
            >>> a += [4, 5, 6]
            >>> a
            <5.0, 7.0, 9.0>
        """
        return self + other

    @VectorDecorators.check_for_veclike
    def __sub__(self, other: "Vector.vector_like") -> "Vector":
        """`Vector()` subtraction implementation.

        Arguments:
            `other` {`Vector.vector_like`} -- second `-` operand.

        Raises:
            `TypeError` -- `other` is not vector-like.

        Returns:
            `Vector` -- new vector object, difference of `self` and `other`.

        Examples:
            >>> a = Vector(1, 2, 3)
            >>> b = Vector(1, 1, 1)
            >>> a - b
            <0.0, 1.0, 2.0>
        """
        if not Vector.is_vectorlike(other):
            raise TypeError(
                Vector.messages["vector_like_singular"]
                + Vector.messages["vector_types"]
            )
        a, b = Vector.to_dimension(max(len(self), len(other)), self, other)
        return Vector(*(ae - be for ae, be in zip(a.pos, b.pos)))

    @VectorDecorators.check_for_veclike
    def __rsub__(self, other: "Vector.vector_like") -> "Vector":
        """`Vector()` (reflexive) subtraction implementation.

        Arguments:
            `other` {`Vector.vector_like`} -- first `-` operand.

        Returns:
            `Vector` -- new vector object, difference of `other` and `self`.

        Examples:
            >>> [1, 2, 3] - Vector(1, 1, 1)
            <0.0, 1.0, 2.0>
        """
        return -self + other

    @VectorDecorators.check_for_veclike
    def __isub__(self, other: "Vector.vector_like") -> "Vector":
        """`Vector()` subtraction implementation.

        Arguments:
            `other` {`Vector.vector_like`} -- second `-=` operand.

        Returns:
            `Vector` -- New vector object, difference of `self` and `other`, assigned to `self`.

        Examples:
            >>> a = Vector(2, 2, 2)
            >>> a -= Vector(1)
            >>> a
            <1.0, 2.0, 2.0>
        """
        return self - other

    @VectorDecorators.check_for_veclike
    def __mul__(self, scalar: "Vector.scalar_like") -> "Vector":
        """`Vector()` scalar multiplication implementation.

        Arguments:
            `scalar` {`Vector.scalar_like`} -- the second `*` argument.

        Raises:
            `TypeError` -- `scalar` is not scalar-like.

        Returns:
            `Vector` -- New vector object, scalar product of `self` and `other`.

        Examples:
            >>> a = Vector(1, 2, 3)
            >>> a * 5
            <5.0, 10.0, 15.0>
        """
        if not Vector.is_scalarlike(scalar):
            raise TypeError(
                Vector.messages["scalar_like_singular"]
                + Vector.messages["scalar_types"]
            )
        return Vector(*(el * float(scalar) for el in self))

    @VectorDecorators.check_for_veclike
    def __rmul__(self, scalar: "Vector.scalar_like") -> "Vector":
        """`Vector()` (reflexive) scalar multiplication implementation.

        Arguments:
            `scalar` {`Vector.scalar_like`} -- the first `*` operand.

        Returns:
            `Vector` -- New vector object, scalar product of `other` and `self`.

        Examples:
            >>> a = Vector(1, 2, 3)
            >>> 5 * a
            <5.0, 10.0, 15.0>
        """
        return self * scalar

    @VectorDecorators.check_for_veclike
    def __imul__(self, scalar: "Vector.scalar_like") -> "Vector":
        """`Vector()` scalar multiplication implementation, with assignment.

        Arguments:
            `scalar` {`Vector.scalar_like`} -- second `*=` operand.

        Returns:
            `Vector` - New vector object, scalar product of `self` and `other`, assigned to `self`.

        Examples:
            >>> a = Vector(1, 2, 3)
            >>> a *= 5
            >>> a
            <5.0, 10.0, 15.0>
        """
        return self * scalar

    @VectorDecorators.check_for_veclike
    def __truediv__(self, scalar: "Vector.scalar_like") -> "Vector":
        """`Vector()` "scalar divison" implementation.

            aka vector scalar multiplication of the form `self * (1/scalar)`

        Arguments:
            `scalar` {`Vector.scalar_like`} -- The second `/` operand.

        Raises:
            `TypeError` -- If `scalar` is not scalar-like.
            `ZeroDivisionError` -- If `scalar` is 0.

        Returns:
            `Vector` -- A new vector as the result of `self * (1/scalar)`.

        Examples:
            >>> a = Vector(4, 6, 8)
            >>> a / 2
            <2.0, 3.0, 4.0>
        """
        if not Vector.is_scalarlike(scalar):
            raise TypeError(
                Vector.messages["scalar_like_singular"]
                + Vector.messages["scalar_types"]
            )
        if scalar == 0:
            raise ZeroDivisionError("Scalar must be non-zero.")
        return (1 / scalar) * self

    @VectorDecorators.check_for_veclike
    def __itruediv__(self, scalar: "Vector.scalar_like") -> "Vector":
        """`Vector()` "scalar divison" implementation, with assignment.

            See `__truediv__`.

        Arguments:
            `scalar` {`Vector.scalar_like`} -- seoncd `/=` operand.

        Returns:
            Vector -- A new vector, the result of `self / scalar`, assigned to `self`.

        Examples:
            >>> a = Vector(4, 6, 8)
            >>> a /= 2
            >>> a
            <2.0, 3.0, 4.0>
        """
        return self / scalar

    def __bool__(self) -> bool:
        """`Vector()` `bool()` implementation.

            A vector is equal to `True` if the sum of its ...
            components is not 0. Otherwise, it is `False`.

        Returns:
            `bool` -- The boolean status of vector `self`.

        Examples:
            >>> bool(Vector(1, 0, 0))
            True
            >>> bool(Vector(0, 0, 0))
            False
        """
        return not (sum(self) == 0)

    def __neg__(self) -> "Vector":
        """Negation implementation.

        Returns:
            `Vector` -- a new vector, `== -1 * self`

        Examples:
            >>> -Vector(1, 2, 3)
            <-1.0, -2.0, -3.0>
        """
        return Vector(*(-el for el in self))

    def __pos__(self) -> "Vector":
        """Positation implementation.

        Returns:
            `Vector` -- a new vector `== 1 * self`

        Examples:
            >>> +Vector(-1, -2, -3)
            <-1.0, -2.0, -3.0>
        """
        return Vector(*(+el for el in self))

    def __abs__(self) -> "Vector":
        """`abs()` implementation.

        Returns:
            `Vector` -- a new vector whose elements are the absolute value of the elements of `self`.

        Examples:
            >>> abs(Vector(1, -2, 3, -4))
            <1.0, 2.0, 3.0, 4.0>
        """
        return Vector(*(abs(el) for el in self))

    def __floor__(self) -> "Vector":
        """`math.floor()` implementation.

        Returns
            `Vector` -- a new vector with all elements rounded down to
                        the closest integer (as floats.)

        Examples:
            >>> import math
            >>> math.floor(Vector(2.5, 3.1, 4.8))
            <2.0, 3.0, 4.0>
        """
        return Vector(*(float(floor(el)) for el in self))

    def __ceil__(self) -> "Vector":
        """`math.ceil()` implementation.

        Returns
            `Vector` -- a new vector with all elements rounded up to the closest integer (as floats.)

        Examples:
            >>> import math
            >>> math.ceil(Vector(2.5, 3.1, 4.8))
            <3.0, 4.0, 5.0>
        """
        return Vector(*(float(ceil(el)) for el in self))

    def __round__(self, n: int) -> "Vector":
        """`round()` implementation.

        Arguments:
            `n` {`int`} -- number of decimals to round components of `self` to.

        Raises:
            `TypeError` -- if `n` isn't an `int`
            `ValueError` -- if `n < 0`

        Returns:
            `Vector` -- a new vector with all elements rounded to `n` decimal places.
        """
        if not isinstance(n, int):
            raise TypeError("n must be an int.")
        if n < 0:
            raise ValueError("n must be >= 0.")
        return Vector(*(float(round(el, n)) for el in self))

    def __complex__(self) -> complex:
        """`complex()` implementation.

            Takes the first two dimensions of a vector. If there ...
            are less than two dimensions, they are padded with zeros.

        Returns:
            `complex` -- A complex-number representation of vector self.
        """
        (v,) = Vector.to_dimension(2, self)
        return complex(v.pos[0], v.pos[1])

    @staticmethod
    @VectorDecorators.check_for_veclike
    def dot(  # pylint: disable=method-hidden
        a: "Vector.vector_like", b: "Vector.vector_like"
    ) -> float:
        """Calculate the dot product between two vectors.

            If `a` and `b` are different lengths, summing ...
            stops at the end of the shorter operand.

        Arguments:
            `a` {`Vector.vector_like`} -- Vector 1
            `b` {`Vector.vector_like`} -- Vector 2

        Raises:
            `TypeError` -- at least `a` or `b` not vector-like

        Returns:
            `float` -- the dot product between `a` and `b`.
        """
        if not (Vector.is_vectorlike(a) and Vector.is_vectorlike(b)):
            raise TypeError(
                Vector.messages["vector_like_plural"] + Vector.messages["vector_types"]
            )
        a_vec, b_vec = Vector.from_vectorlike(a), Vector.from_vectorlike(b)
        return float(sum(x * y for x, y in zip(a_vec.pos, b_vec.pos)))

    def _dot(self, other: "Vector.vector_like") -> float:
        """Instance-method implementation of dot product."""
        return Vector.dot(self, other)

    @staticmethod
    @VectorDecorators.check_for_veclike
    def cross(  # pylint: disable=method-hidden
        a: "Vector.vector_like", b: "Vector.vector_like"
    ) -> "Vector":
        """Calculate the cross product between two vectors in R3.

            If `a` or `b` are of higher dimensions, they will be reduced to their first ...
            three elements If they are in smaller dimensions, they will be padded with 0s.

        Arguments:
            `a` {`Vector.vector_like`} -- Vector 1
            `b` {`Vector.vector_like`} -- Vector 2

        Raises:
            `TypeError` -- `a` or `b` not vector-like.

        Returns:
            `Vector` -- A vector orthogonal to `a` and `b`
        """
        if not (Vector.is_vectorlike(a) and Vector.is_vectorlike(b)):
            raise TypeError(
                Vector.messages["vector_like_plural"] + Vector.messages["vector_types"]
            )
        v1: Optional["Vector"] = None
        v2: Optional["Vector"] = None
        v1, v2 = Vector.to_dimension(3, a, b)
        return Vector(
            (v1.pos[1] * v2.pos[2]) - (v1.pos[2] * v2.pos[1]),
            (v1.pos[2] * v2.pos[0]) - (v1.pos[0] * v2.pos[2]),
            (v1.pos[0] * v2.pos[1]) - (v1.pos[1] * v2.pos[0]),
        )

    def _cross(self, other: "Vector.vector_like") -> "Vector":
        """Instance-method implementation of cross product."""
        return Vector.cross(self, other)

    @staticmethod
    @VectorDecorators.check_for_veclike
    def box(  # pylint: disable=method-hidden
        a: "Vector.vector_like", b: "Vector.vector_like", c: "Vector.vector_like"
    ) -> float:
        """Vector scalar triple product, or box product.

        Arguments:
            `a` {`Vector.vector_like`} -- Vector 1
            `b` {`Vector.vector_like`} -- Vector 2
            `c` {`Vector.vector_like`} -- Vector 3

        Raises:
            `TypeError` -- `a`, `b`, or `c` not vector-like.

        Returns:
            `float` -- scalar triple product of `a`, `b`, `c`.
        """
        if not (
            Vector.is_vectorlike(a)
            and Vector.is_vectorlike(b)
            and Vector.is_vectorlike(c)
        ):
            raise TypeError(
                Vector.messages["vector_like_plural"] + Vector.messages["vector_types"]
            )
        a, b, c = (
            Vector.from_vectorlike(a),
            Vector.from_vectorlike(b),
            Vector.from_vectorlike(c),
        )
        return Vector.dot(a, Vector.cross(b, c))

    def _box(self, *vecs: "Vector.vector_like") -> float:
        """Instance-method implementation of scalar triple product."""
        return Vector.box(self, vecs[0], vecs[1])

    @staticmethod
    @VectorDecorators.check_for_veclike
    def angle(  # pylint: disable=method-hidden
        a: "Vector.vector_like", b: "Vector.vector_like"
    ) -> float:
        """Determine the angle between two vectors.

        Arguments:
            `a` {`Vector.vector_like`} -- Vector 1
            `b` {`Vector.vector_like`} -- Vector 2

        Raises:
            `TypeError` -- `a` or `b` not vector-like.

        Returns:
            `float` -- The angle between `a`, `b` in [radians].
        """
        if not (Vector.is_vectorlike(a) and Vector.is_vectorlike(b)):
            raise TypeError(
                Vector.messages["vector_like_plural"] + Vector.messages["vector_types"]
            )
        a, b = Vector.from_vectorlike(a), Vector.from_vectorlike(b)
        return acos(Vector.dot(a, b) / (a.mag * b.mag))

    def _angle(self, other: "Vector.vector_like") -> float:
        """Instance-method implementation of angle-between."""
        return Vector.angle(self, other)

    @staticmethod
    @VectorDecorators.check_for_veclike
    def comp(  # pylint: disable=method-hidden
        a: "Vector.vector_like", b: "Vector.vector_like"
    ) -> float:
        """Scalar projection of vector `a` on vector `b`.

        Arguments:
            `a` {`Vector.vector_like`} -- The base vector to project.
            `b` {`Vector.vector_like`} -- The target vector to project `self` on.

        Raises:
            `TypeError` -- `a` or `b` not vector-like.

        Returns:
            `float` -- The resultant scalar.
        """
        if not (Vector.is_vectorlike(a) and Vector.is_vectorlike(b)):
            raise TypeError(
                Vector.messages["vector_like_plural"] + Vector.messages["vector_types"]
            )
        a, b = Vector.from_vectorlike(a), Vector.from_vectorlike(b)
        return Vector.dot(a, a) / b.mag

    def _comp(self, other: "Vector.vector_like") -> float:
        """Instance-method implementation of scalar projection."""
        return Vector.comp(self, other)

    @staticmethod
    @VectorDecorators.check_for_veclike
    def proj(  # pylint: disable=method-hidden
        a: "Vector.vector_like", b: "Vector.vector_like"
    ) -> "Vector":
        """Projection of vector `a` on vector `b`.

        Arguments:
            `a` {`Vector.vector_like`} -- The base vector to project.
            `b` {`Vector.vector_like`} -- The target vector to project `self` on.

        Raises:
            `TypeError` -- `a` or `b` not vector-like.

        Returns:
            `Vector` -- The resultant projected vector.
        """
        if not (Vector.is_vectorlike(a) and Vector.is_vectorlike(b)):
            raise TypeError(
                Vector.messages["vector_like_plural"] + Vector.messages["vector_types"]
            )
        a, b = Vector.from_vectorlike(a), Vector.from_vectorlike(b)
        return Vector.dot(a, b.unit) * b.unit

    def _proj(self, other: "Vector.vector_like") -> "Vector":
        """Instance-method implementation of vector projection."""
        return Vector.proj(self, other)

    @staticmethod
    @VectorDecorators.check_for_veclike
    def reject(  # pylint: disable=method-hidden
        a: "Vector.vector_like", b: "Vector.vector_like"
    ) -> "Vector":
        """Rejection of vector a on vector b.

        Arguments:
            `a` {`Vector.vector_like`} -- The base vector to reject.
            `b` {`Vector.vector_like`} -- The target vector to reject `self` on.

        Raises:
            `TypeError` -- `a` or `b` not vector-like.

        Returns:
            `Vector` -- The resultant reject vector
        """
        if not (Vector.is_vectorlike(a) and Vector.is_vectorlike(b)):
            raise TypeError(
                Vector.messages["vector_like_plural"] + Vector.messages["vector_types"]
            )
        a, b = Vector.from_vectorlike(a), Vector.from_vectorlike(b)
        return a - ((Vector.dot(a, b) / Vector.dot(b, b)) * b)

    def _reject(self, other: "Vector.vector_like") -> "Vector":
        """Instance-method implementation of vector rejection.

        Examples:
            >>> Vector(1, 2, 3).reject([2, 4, 6])
            <0.0, 0.0, 0.0>
        """
        return Vector.reject(self, other)

    @staticmethod
    def is_vectorlike(*potential: Any) -> bool:
        """Test for objects being vector-like. Return True iff all objects are vector-like.

        Arguments:
            `*potential` {`Any`} -- Object to test for vector-likeness.

        Returns:
            `bool` -- Whether `*potential` contains only vector-like.

        Examples:
            >>> Vector.is_vectorlike(Vector(1), [1, 2])
            True
            >>> Vector.is_vectorlike(Vector(1), "a")
            False
        """
        for obj in potential:
            if not (
                hasattr(obj, "__iter__")
                and all(Vector.is_scalarlike(element) for element in obj)
            ):
                return False
        return True

    @staticmethod
    def is_scalarlike(potential: Any) -> bool:
        """Test for object being scalar-like.

        Arguments:
            `potential` {`Any`} -- Object to test for scalar-likeness.

        Returns:
            `bool` -- Whether potential is scalar-like or not.

        Examples:
            >>> a, b, c = 5, 1.1, [1.0]
            >>> Vector.is_scalarlike(a), Vector.is_scalarlike(b), Vector.is_scalarlike(c)
            (True, True, False)
        """
        # using type() here to avoid bools being valid scalars.
        return type(potential) in (int, float)

    @staticmethod
    @VectorDecorators.check_for_veclike
    def to_dimension(n: int, *vecs: "Vector.vector_like") -> Tuple["Vector", ...]:
        """Convert vectors to a specific dimension.

        If vectors exceed `n` components, extras are truncated.
        Missing components are filled with zeros.

        Arguments:
            `*vecs` {`Vector.vector_like`} -- vectors to convert.
            `n` {`int`} -- dimension to convert vectors to.

        Raises:
            `TypeError` -- `n` is not `int`, or `vec in vecs` is not vector-like.
            `ValueError` -- `n` is not positive.

        Returns:
            `vecs` {`Tuple[Vector, ...]`} -- converted vectors.

        Examples:
            >>> a, b, c = Vector(1), Vector(1, 2, 3), Vector()
            >>> Vector.to_dimension(2, a, b, c)
            (<1.0, 0.0>, <1.0, 2.0>, <0.0, 0.0>)
        """
        if not isinstance(n, int):
            raise TypeError("n must be int.")
        if n < 0:
            raise ValueError("n must be positive.")
        if not all(Vector.is_vectorlike(vec) for vec in vecs):
            raise TypeError(
                Vector.messages["vector_like_plural"] + Vector.messages["vector_types"]
            )
        lvecs = [list(vec) for vec in vecs]
        vlists = [[0 for _ in range(n)] for _ in lvecs]
        for i, vlist in enumerate(vlists):
            vlist[0 : len(vecs[i])] = vecs[i][0:n]  # type: ignore  # TODO: fix typing error
        return tuple([Vector.from_vectorlike(vlist) for vlist in vlists])
