import archibald as arb
import pytest


def test_bounds():
    opti = arb.Opti()

    x = opti.variable(init_guess=5, lower_bound=3)
    opti.minimize(x**2)

    sol = opti.solve()

    assert sol(x) == pytest.approx(3)


if __name__ == "__main__":
    pytest.main()



