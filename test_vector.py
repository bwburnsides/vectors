# nosec
import pytest
from vector import Vector as vec

# "Fixtures"?
v = vec(1, 1)


# Test initializing vectors with varying lengths, and filling of pos
@pytest.mark.parametrize(
    "args, pos",
    [((1, 1), ((1, 1))), ((1, 1, 1), ((1, 1, 1))), ((1,), ((1,))), ((), ((0, 0, 0)))],
)
def test_init(args, pos):
    v = vec(*args)
    assert v._coords == pos
    assert v.pos == pos


# TODO: Test from_unit()
# TODO: Test from_vectorlike()
# TODO: Test from_angle()

# TODO: parameterize and include tests for slice, tuple, negative indice, etc
def test_getitem():
    v = vec(1, 2, 3)
    assert v[0] == 1
    assert v[-1] == 3


# TODO: improve test
def test_len():
    assert len(v) == 2


# TODO: improve test
def test_repr():
    assert tuple(v) == (1.0, 1.0)


# TODO: improve test
def test_str():
    assert str(v) == "(1.0, 1.0)"


# TODO: test __iter__?
# TODO: test __next__?
# TODO: test pos property, setter

# TODO: parameterize and improve test, test setter
def test_unit():
    m = 3 ** (1 / 2)
    u = vec(1 / m, 1 / m, 1 / m)
    v = vec(1, 1, 1)
    assert v.unit == u


# TODO: parameterize, test setter
def test_mag():
    v = vec(1, 1, 1)
    assert v.mag == 3 ** (1 / 2)


@pytest.mark.parametrize(
    "pot, res",
    [
        ([1, 1, 1], True),
        ([0, 0, 0], True),
        (vec(1, 1, 1), True),
        (["a"], False),
        (("a"), False),
        ([1, 1.6, True], True),
        ([1, 1.6, "True"], False),
    ],
)
def test_is_vectorlike(pot, res):
    assert vec.is_vectorlike(pot) == res


@pytest.mark.parametrize(
    "pot, res",
    [
        (0, True),
        ("a", False),
        (1.5, True),
        ([0, 0, 0], False),
        ((1, 1, 1), False),
        (True, True),
        (False, True),
    ],
)
def test_is_scalarlike(pot, res):
    assert vec.is_scalarlike(pot) == res


# TODO: test pad_vectors
# TODO: test to_dimension


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (vec(1, 1, 1), vec(2, 2, 2), vec(3, 3, 3)),
        (vec(1, 1), vec(1, 1, 1), vec(2, 2, 1)),
        (vec(1, 1, 1), vec(1, 1), vec(2, 2, 1)),
        ([1, 1], vec(1, 1), vec(2, 2)),
        ([1, 1, 1], vec(1, 1), vec(2, 2, 1)),
        ([1, 1], vec(1, 1, 1), vec(2, 2, 1)),
    ],
)
def test_add(a, b, expected):
    assert (a + b) == expected


# TODO: test __radd__
# TODO: test __iadd__


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (vec(1, 1), vec(1, 1), vec(0, 0)),
        (vec(1), vec(1, 1, 1), vec(0, -1, -1)),
        (vec(1, 1, 1), vec(1), vec(0, 1, 1)),
    ],
)
def test_sub(a, b, expected):
    assert (a - b) == expected


# TODO: test __rsub__
# TODO: test __isub__
# TODO: test __mul__
# TODO: test __rmul__
# TODO: test __imul__
# TODO: test __truediv__
# TODO: test __itruediv__


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (vec(1, 1, 1), vec(1, 1, 1), True),
        ([1, 1, 1], vec(1, 1, 1), True),
        (vec(1, 1, 1), [1, 1, 1], True),
        ((1, 1, 1), vec(1, 1, 1), True),
        (vec(1, 1, 1), (1, 1, 1), True),
        (vec(1, 1, 1), vec(1, 1, 1, 1), False),
        (vec(1, 1, 1), (1, 1, 1, 1), False),
        (vec(1, 2, 1), vec(1, 1, 1), False),
        (vec(1), True, False),
    ],
)
def test_eq(a, b, expected):
    assert (a == b) == expected


# print("Test vec.unit: ", str(v.unit))
# a = vec.from_unit((0.707, 0.707), 1.414)
# print("Test vec.from_unit, should be 1.414: ", str(a.mag))
# print("Test vec.from_unit, should be (0.707, 0.707): ", str(a.unit))
