import numpy as np
import numpy.typing as npt
import statsmodels.api as sm


def w(size: int) -> npt.NDArray[np.float64]:
    mean = 0
    std = 3 / 4
    return np.random.normal(mean, std, size=size)


def main():
    N = 20000
    noise = w(N)

    mean = np.mean(noise)
    autocorr = sm.tsa.acf(noise)


if __name__ == "__main__":
    main()
