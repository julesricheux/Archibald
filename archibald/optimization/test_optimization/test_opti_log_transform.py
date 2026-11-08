import archibald as arb
import pytest


def test_bounds():
    opti = arb.Opti()
    x = opti.variable(init_guess=3, log_transform=True, lower_bound=7)
    opti.minimize(x)
    sol = opti.solve()

    assert sol(x) == pytest.approx(7)


if __name__ == "__main__":
    pytest.main()



