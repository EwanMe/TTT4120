import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy.signal


def cnt_pad_gamma(gamma: npt.NDArray, *, order: int):
    if len(gamma) // 2 <= order:
        pad = order - len(gamma) // 2
        gamma = np.pad(gamma, (pad, pad), "constant")

    return np.concat((gamma[len(gamma) // 2 :], gamma[: len(gamma) // 2]))


def yule_walker(gamma: npt.NDArray, *, order: int):
    if len(gamma) % 2 == 0:
        raise ValueError("Autocorrelation sequence must be of odd length")

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
    gamma = [-1 / 2, 5 / 4, -1 / 2]

    for order in [1, 2, 3]:
        coefficients = yule_walker(gamma, order=order)
        pev = prediction_error_variance(gamma, coefficients, order=order)
        print(f"Order={order}:\n", coefficients)
        print(f"Prediction error variance={pev}")

        a = np.concat([[1], coefficients])
        w, H = scipy.signal.freqz([1], a)
        f = w / (2 * np.pi)
        psd_estimate = 1 / (np.abs(H) ** 2)

        psd_theoretical = 5 / 4 - np.cos(2 * np.pi * f)

        ax = plt.subplot()
        plt.title("Power spectrum density")
        ax.plot(f, psd_estimate, label="Estimated")
        ax.plot(f, psd_theoretical, label="Theoretical")
        ax.legend()
        plt.show()


def main():
    subtask_c()


if __name__ == "__main__":
    main()
