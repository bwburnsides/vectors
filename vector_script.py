"""This script is used to generate traces for Monkeytype to detect.

Eventually type hinting may be automatically generated for vector.py. vector.py is already 95%+
type hinted; however, mypy has issues with a few of my type hints. Hopefully this may clear
them up. This also serves as a way to perform quick tests until unit tests are properly built
out with pytest.
"""
from vector import Vector as v

# Create vectors by passing in coords directly
a = v(1, 1, 1)

# Or by passing vector-like types
b = v.from_vectorlike([2, 2, 2])

# Or by passing a unit and a magnitude
# here the unit is any vector-like
c = v.from_unit([1, 1], 5)
c = v.from_unit((1,), 5)
c = v.from_unit(v(1, 1), 5)

# Or by passing an angle (rel to x-axis) and a magnitude
d = v.from_angle(3.141592 / 2, 5)

# Cross and dot products can be used statically or on-instance
assert v.dot(a, b) == a.dot(b)

assert v.cross(a, b) == a.cross(b)

# Same for box products, scalar and vector projection, and rejection
assert v.box(a, b, c) == a.box(b, c)

assert v.comp(a, b) == a.comp(b)

# With the static methods, args just need to be vector-like. For the instance, only one does.
i = v.dot([1, 1, 1], [1, 1, 1])
j = a.dot([1, 1, 1])

k = v.cross([1], (1, 1, 1))  # notice here that the missing comps will be filled in.
el = a.cross((1, 1, 1, 1))  # and here the extra comp will be trimmed.

# Access unit and mag directly
amag = a.mag
aunit = a.unit

# Grab vector coordinates as tuple
apos = a.pos

# Perform basic vector-operations easily
a + b
a += a

a - b
a -= a
a = b * 5  # Needed so that a.mag not 0 later

5 * a
a *= 5

a / 5
a /= 2

# Convert vecs to complex nums
# again, extra comps are filled/trimmed
complex1 = complex(v(1, 1))
complex2 = complex(v(1))
complex3 = complex(v(1, 1, 1))

# get the angle between two vectors
assert v.angle(a, b) == a.angle(b)
