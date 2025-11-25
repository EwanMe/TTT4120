import numpy as np
import numpy.typing as npt
from statsmodels.regression.linear_model import yule_walker


def w(
    length: int, *, mean: float = 0, std: float = 1
) -> npt.NDArray[np.float64]:
    return np.random.normal(mean, std, size=length)


def s(N: int) -> npt.NDArray[np.float64]:
    v = w(N, std=np.sqrt(0.09))
    ss = np.zeros(N)
    ss[0] = v[0]
    for n in range(1, N):
        ss[n] = 0.9 * ss[n - 1] + v[n]
    return ss


def x(N) -> npt.NDArray[np.float64]:
    return s(N) + w(N)


def main() -> None:
    R = np.array(
        [[1.474, 0.426, 0.384], [0.426, 1.474, 0.426], [0.384, 0.426, 1.474]]
    )
    g = [0.474, 0.426, 0.384]

    h = np.linalg.inv(R) @ g
    print(h)

    rho, sigma = yule_walker(x(10000), order=3)
    print(rho, sigma)


if __name__ == "__main__":
    main()
