import pytest
from vector import Vector as vec
from math import pi, floor, ceil

# pytest --cov vector
# pytest --cov vector --cov-report html

# "Fixtures"?
v = vec(1, 1)


# Test initializing vectors with varying lengths, and filling of pos
@pytest.mark.parametrize(
    "args, pos",
    [((1, 1), ((1, 1))), ((1, 1, 1), ((1, 1, 1))), ((1,), ((1,))), ((), ((tuple())))],
)
def test_init(args, pos):
    v = vec(*args)
    assert v._coords == pos
    assert v.pos == pos


@pytest.mark.parametrize("coords", [([True, 0, 1]), ((1, [0], "a")), ({"testkey": 3})])
def test_init_typeerror(coords):
    try:
        vec(*coords)
        assert False, "Invalid vector was created."
    except TypeError:
        assert True


@pytest.mark.parametrize("u, m, result", [(vec(1, 0), 5, vec(5, 0))])
def test_fromunit(u, m, result):
    assert vec.fromunit(u, m) == result


@pytest.mark.parametrize("a", [("hello"), (["test"]), ({"k": 1})])
def test_fromunit_typeerror(a):
    try:
        vec.fromunit(a, 1)
        assert False, "Invalid vector was created."
    except TypeError:
        assert True


@pytest.mark.parametrize(
    "vl, res", [([], vec()), ([1], vec(1)), (tuple(), vec()), ((1,), vec(1))]
)
def test_fromvectorlike(vl, res):
    assert vec.fromvectorlike(vl) == res


@pytest.mark.parametrize("theta, length, res", [(pi, 1, vec(-1, 0))])
def test_fromangle(theta, length, res):
    new_vec = vec.fromangle(theta, length)
    assert new_vec._pos_rounded == res._pos_rounded


# @pytest.mark.parametrize(
#     "t, m", [("not_num", 2), (2, "not_num"), ("not_num", "not_num")]
# )
# def test_fromangle_typeerror(t, m):
#     try:
#         vec.fromangle(t, m)
#         assert False, "Invalid vector was created."
#     except TypeError:
#         assert True


# TODO: parameterize and include tests for slice, tuple, negative indice, etc
@pytest.mark.parametrize("i, x", [(0, 1), (-1, 3), (-2, 2)])
def test_getitem_int(i, x):
    v = vec(1, 2, 3)
    assert v[i] == x


@pytest.mark.parametrize("s, x", [((0, 4, 1), (1, 2, 3, 4))])
def test_getitem_slice(s, x):
    v = vec(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    assert v[s[0] : s[1] : s[2]] == x


@pytest.mark.parametrize("test_v,x", [(v, 2), (vec(), 0)])
def test_len(test_v, x):
    assert len(test_v) == x


# TODO: improve test
def test_repr():
    assert repr(v) == "<1.0, 1.0>"


# TODO: improve test
def test_str():
    v = vec(1, 1)
    assert str(v) == "<1.0, 1.0>"


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
        ([1, 1.6, True], False),
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
        (True, False),
        (False, False),
    ],
)
def test_is_scalarlike(pot, res):
    assert vec.is_scalarlike(pot) == res


# # TODO: test to_dimension
# def test_to_dimension()


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


@pytest.mark.parametrize(
    "b, a, expected",
    [
        (vec(1, 1, 1), vec(2, 2, 2), vec(3, 3, 3)),
        (vec(1, 1), vec(1, 1, 1), vec(2, 2, 1)),
        (vec(1, 1, 1), vec(1, 1), vec(2, 2, 1)),
        ([1, 1], vec(1, 1), vec(2, 2)),
        ([1, 1, 1], vec(1, 1), vec(2, 2, 1)),
        ([1, 1], vec(1, 1, 1), vec(2, 2, 1)),
    ],
)
def test_radd(b, a, expected):
    assert (a + b) == expected


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
def test_iadd(a, b, expected):
    a += b
    assert a == expected


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


@pytest.mark.parametrize(
    "a, b, expected", [([1], vec(1, 1), vec(0, -1)), ([], vec(1), vec(-1))]
)
def test_rsub(a, b, expected):
    assert (a - b) == expected


@pytest.mark.parametrize(
    "a, b, expected",
    [(vec(1, 1), vec(1, 1), vec(0, 0)), (vec(1, 1), [1, 1], vec(0, 0))],
)
def test_isub(a, b, expected):
    a -= b
    assert a == expected


@pytest.mark.parametrize(
    "a, b, expected", [(vec(1), 5, vec(5)), (vec(1, 1), 5, vec(5, 5))]
)
def test_mul(a, b, expected):
    assert a * b == expected


@pytest.mark.parametrize(
    "a, b, expected", [(5, vec(1), vec(5)), (5, vec(1, 1), vec(5, 5))]
)
def test_rmul(a, b, expected):
    assert a * b == expected


@pytest.mark.parametrize(
    "a, b, expected", [(vec(1), 5, vec(5)), (vec(1, 1), 5, vec(5, 5))]
)
def test_imul(a, b, expected):
    a *= b
    assert a == expected


@pytest.mark.parametrize("a, b, expected", [(vec(6), 2, vec(3))])
def test_truediv(a, b, expected):
    assert a / b == expected


@pytest.mark.parametrize("a, b, expected", [(vec(6), 2, vec(3))])
def test_itruediv(a, b, expected):
    a /= b
    assert a == expected


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


@pytest.mark.parametrize(
    "a, b, result",
    [
        (vec(1.0, 0), vec(1.0, 0), False),
        (vec(2.0, 0), vec(1.0, 0), True),
        (vec(1.0, 0), vec(2.0, 0), False),
    ],
)
def test_gt(a, b, result):
    assert (a > b) == result


@pytest.mark.parametrize(
    "a, b, result",
    [
        (vec(1.0, 0), vec(1.0, 0), False),
        (vec(2.0, 0), vec(1.0, 0), False),
        (vec(1.0, 0), vec(2.0, 0), True),
    ],
)
def test_lt(a, b, result):
    assert (a < b) == result


@pytest.mark.parametrize(
    "a, b, result",
    [
        (vec(1.0, 0), vec(1.0, 0), True),
        (vec(2.0, 0), vec(1.0, 0), True),
        (vec(1.0, 0), vec(2.0, 0), False),
    ],
)
def test_ge(a, b, result):
    assert (a >= b) == result


@pytest.mark.parametrize(
    "a, b, result",
    [
        (vec(1.0, 0), vec(1.0, 0), True),
        (vec(2.0, 0), vec(1.0, 0), False),
        (vec(1.0, 0), vec(2.0, 0), True),
    ],
)
def test_le(a, b, result):
    assert (a <= b) == result


@pytest.mark.parametrize(
    "a, result",
    [(vec(1, 0), True), (vec(), False), (vec(0, 0, 0), False), (vec(1, 1, 1, 1), True)],
)
def test_bool(a, result):
    assert bool(a) == result


@pytest.mark.parametrize("a, result", [(vec(1, 2, 3), vec(1, 2, 3))])
def test_pos(a, result):
    assert +a == result


@pytest.mark.parametrize(
    "a, result",
    [
        (vec(1, 2, 3), vec(1, 2, 3)),
        (vec(-1, 2, 3), vec(1, 2, 3)),
        (vec(-1, -2, -3), vec(1, 2, 3)),
    ],
)
def test_abs(a, result):
    assert abs(a) == result


@pytest.mark.parametrize(
    "a, result", [(vec(1.1, 2.6, 5.7), vec(1, 2, 5)), (vec(1, 2, 3), vec(1, 2, 3))]
)
def test_floor(a, result):
    assert floor(a) == result


@pytest.mark.parametrize(
    "a, result", [(vec(1.1, 2.6, 5.7), vec(2, 3, 6)), (vec(1, 2, 3), vec(1, 2, 3))]
)
def test_ceil(a, result):
    assert ceil(a) == result


@pytest.mark.parametrize("a, n, result", [(vec(2.456, 3.914), 2, vec(2.46, 3.91))])
def test_round(a, n, result):
    assert round(a, n) == result


@pytest.mark.parametrize(
    "a, result", [(vec(1, 0), complex(1, 0)), (vec(), complex(0, 0))]
)
def test_complex(a, result):
    assert complex(a) == result


@pytest.mark.parametrize("a, b, result", [(vec(1, 1, 1), vec(1, 1, 1), 3)])
def test_dot(a, b, result):
    assert vec.dot(a, b) == result
    assert vec.dot(b, a) == result
    assert a.dot(b) == result
    assert b.dot(a) == result


@pytest.mark.parametrize("a, b, result", [(vec(0, 0, 0), vec(0, 0, 0), vec(0, 0, 0))])
def test_cross(a, b, result):
    assert vec.cross(a, b) == result
    assert a.cross(b) == result
    assert vec.cross(b, a) == -result
    assert b.cross(a) == -result


@pytest.mark.parametrize(
    "a, b, result", [(vec(1, 2, 3), vec(0, 1, 2), vec(0, 1.6, 3.2))]
)
def test_proj(a, b, result):
    assert round(vec.proj(a, b), 1) == result


# @pytest.mark.parameterize("a, b, result", [()])
# def test_reject(a, b, result):
#     assert round(vec.reject(a, b), 1) == result


# print("Test vec.unit: ", str(v.unit))
# a = vec.fromunit((0.707, 0.707), 1.414)
# print("Test vec.fromunit, should be 1.414: ", str(a.mag))
# print("Test vec.fromunit, should be (0.707, 0.707): ", str(a.unit))
