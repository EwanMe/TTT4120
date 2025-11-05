import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


def w(size: int):
    mean = 0
    std = 1
    return np.random.normal(mean, std, size=size + 1)


def stem(x, n: int = 10):
    r = np.arange(-n, n + 1)

    ax = plt.subplot()
    plt.title("Autocorrelation")
    ax.stem(
        r,
        x[len(x) // 2 + r[0] : len(x) // 2 + r[-1] + 1],
        label="Estimated",
    )
    ax.legend()
    plt.show()


def cnt_pad_gamma(gamma: npt.NDArray, *, order: int):
    if len(gamma) // 2 <= order:
        pad = order - len(gamma) // 2
        gamma = np.pad(gamma, (pad, pad), "constant")

    return np.concat((gamma[len(gamma) // 2 :], gamma[: len(gamma) // 2]))


def yule_walker(gamma: npt.NDArray, *, order: int):
    if len(gamma) % 2 == 0:
        raise ValueError("Autocorrelation sequence must be of odd lenght")

    centered_g = cnt_pad_gamma(gamma, order=order)
    R = np.zeros([order, order])

    for i in range(order):
        for j in range(order):
            R[i][j] = centered_g[-j + i]

    g = np.array([centered_g[i] for i in range(1, order + 1)]).T

    return -np.linalg.inv(R) @ g


def prediction_error_variance(gamma, coeff, order):
    cnt_gamma = cnt_pad_gamma(gamma, order=order)
    return cnt_gamma[0] + sum(
        [coeff[i - 1] * cnt_gamma[i] for i in range(1, order + 1)]
    )


def subtask_c():
    n = np.array(10000)
    w_n = w(n + 1)
    x = w_n[1:] - 0.5 * w_n[:-1]
    gamma = np.correlate(x, x, "full") / n

    gamma = [-1 / 2, 5 / 4, -1 / 2]

    for order in [1, 2, 3]:
        coefficients = yule_walker(gamma, order=order)
        pev = prediction_error_variance(gamma, coefficients, order=order)
        print(f"Order={order}:\n", coefficients)
        print(f"Prediction error variance={pev}")


def main():
    subtask_c()


if __name__ == "__main__":
    main()
