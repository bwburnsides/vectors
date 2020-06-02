import pytest
from vector import Vector as vec

# "Fixtures"?
v = vec(1, 1)


# Test initializing vectors with varying lengths, and filling of pos
@pytest.mark.parametrize(
    "args, pos",
    [((1, 1), ((1, 1))), ((1, 1, 1), ((1, 1, 1))), ((1,), ((1,))), ((), (()))],
)
def test_init(args, pos):
    v = vec(*args)
    assert v._coords == pos
    assert v.pos == pos


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


def test_mag():
    assert v.mag == pytest.approx(2 ** 0.5, abs=0.001)


# print("Test vec.unit: ", str(v.unit))
# a = vec.from_unit((0.707, 0.707), 1.414)
# print("Test vec.from_unit, should be 1.414: ", str(a.mag))
# print("Test vec.from_unit, should be (0.707, 0.707): ", str(a.unit))
