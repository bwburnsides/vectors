import pytest
from vector import vec

# "Fixtures"?
v = vec(1, 1)


@pytest.mark.parametrize("args, pos", [((1, 1), ((1, 1))), ((1, 1, 1), ((1, 1, 1)))])
def test_init(args, pos):
    v = vec(*args)
    assert v._pos == pos


def test_getitem():
    assert v[0] == 1


def test_len():
    assert len(v) == 2


def test_repr():
    assert tuple(v) == (1.0, 1.0)


def test_str():
    assert str(v) == "(1.0, 1.0)"


def test_mag():
    assert v.mag == pytest.approx(2 ** 0.5, abs=0.001)


# print("Test vec.unit: ", str(v.unit))
# a = vec.from_unit((0.707, 0.707), 1.414)
# print("Test vec.from_unit, should be 1.414: ", str(a.mag))
# print("Test vec.from_unit, should be (0.707, 0.707): ", str(a.unit))