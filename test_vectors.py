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
def test_isscalarlike(pot, res):
    assert vec.is_scalarlike(pot) == res


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
def test_isvectorlike(pot, res):
    assert vec.is_vectorlike(pot) == res


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
    ],
)
def test_eq(a, b, expected):
    assert (a == b) == expected


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (vec(1, 1, 1), vec(2, 2, 2), vec(3, 3, 3)),
        (vec(1, 1), vec(1, 1, 1), vec(2, 2, 1)),
        (vec(1, 1, 1), vec(1, 1), vec(2, 2, 1)),
    ],
)
def test_add(a, b, expected):
    assert (a + b) == expected


def test_mag():
    v = vec(1, 1, 1)
    assert v.mag == 3 ** (1 / 2)


def test_unit():
    m = 3 ** (1 / 2)
    u = vec(1 / m, 1 / m, 1 / m)
    v = vec(1, 1, 1)
    assert v.unit == u


def test_getitem():
    v = vec(1, 2, 3)
    assert v[0] == 1
    assert v[-1] == 3


# def test_getitem_exception():
#     with pytest.raises(IndexError) as e:
#         assert v[2] == None


def test_len():
    assert len(v) == 2


def test_repr():
    assert tuple(v) == (1.0, 1.0)


def test_str():
    assert str(v) == "(1.0, 1.0)"


# print("Test vec.unit: ", str(v.unit))
# a = vec.from_unit((0.707, 0.707), 1.414)
# print("Test vec.from_unit, should be 1.414: ", str(a.mag))
# print("Test vec.from_unit, should be (0.707, 0.707): ", str(a.unit))
