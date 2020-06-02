# vector.py

This is a single-class pure-Python implementation of a vector type with no external dependencies.

Vectors are created by instantiating the `Vector()` class with any number of desired coordinates.

## goals

- To write a native-feeling, and well polished class for use by others.

  - Well implemented magic methods.
  - Rigorous exception handling.

- To learn `mypy` and strong type hinting in Python.

- To learn `pytest` and maintain a rigorously tested project.

- To practice maintaining useful and descriptive documentation.

## features and examples

All examples below have the implied import below, to allow for terse initialization:

    from vector import Vector as v

- Support for the subscript, unpacking, and slicing operators (`[]`, `*`, `[::]`), including with negative indices.

        >>> a = v(2, 5, 6)
        >>> a[0]
        2.0
        >>> a[0:1:1]
        (2.0,)
        >>> a_list = [*a]
        [2.0, 5.0, 6.0]

- Native integration with the Iterator Protocol, allowing for Pythonic `for` loops.

        >>> a = v(2, 5, 6)
        >>> for coord in a:
        ...     print(coord)
        ...
        2.0
        5.0
        6.0

- Integration with `len()`, `str()`, and `tuple()`, (probably others too). Also, nice representation.

        >>> a = v(2, 5, 6)
        >>> a
        Vector: (2.0, 5.0, 6.0)
        >>> len(a)
        3
        >>> str(a)
        '(2.0, 5.0, 6.0)'

- Operation overloading to allow for addition, subtraction, and scalar multiplication with `+`, `-`, and `*`.

        >>> a = v(2, 5, 6)
        >>> b = v(4, 2, 3)
        >>> b_list = [4, 2, 3]
        >>> a + b
        Vector: (6.0, 7.0, 9.0)
        >>> a + b_list
        Vector: (6.0, 7.0, 9.0)
        >>> a -= b
        >>> a
        Vector: (-2.0, 3.0, 3.0)
        >>> 4 * b
        Vector: (16.0, 8.0, 12.0)

- Vector units and magnitudes

        >>> a = v(2, 5, 6)
        >>> a.mag
        8.06225774829855
        >>> a.unit
        Vector: (0.24806946917841693, 0.6201736729460423, 0.7442084075352507)

- Vector cross products and dot products.

        >>> a = v(2, 5, 6)
        >>> b = v(4, 2, 3)
        >>> c = v.dot(a, b)
        >>> c
        36.0
        >>> d = v.cross(a, b)
        >>> d
        Vector: (3.0, 18.0, -16.0)

* Vector projection, scalar projection, and rejection.

        >>> a = v(2, 5, 6)
        >>> b = v(4, 2, 3)
        >>> c = a.comp(b)  # The scalar projection of a on b
        >>> c
        6.685032174373868
        >>> d = a.proj(b)  # The vector projection of a on b
        >>> d
        Vector: (4.9655172413793105, 2.4827586206896552, 3.7241379310344827)
        >>> e = a.reject(b)  # The vector reject of a on b
        >>> e
        Vector: (-2.0, 3.0, 3.0)

* Instantiation of vectors from native vector-like data types (`list`, `tuple`).

* Instantiation of vectors from a unit `Vector()` (or vector-like object) and a desired magnitude.

* Tolerant to vector operations being performed with non-`Vector()` objects.

        >>> v(1, 2, 3) + [4, 5, 6]  # works reflexively too
        Vector: (5.0, 7.0, 9.0)
        >>> v.dot([1, 2, 3], [6, 7, 8])
        44.0
