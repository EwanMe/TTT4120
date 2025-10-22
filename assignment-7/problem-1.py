import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import statsmodels.api as sm


def white_binary_noise(size: int) -> npt.NDArray[np.float64]:
    return np.random.choice([-1, 1], size=size)


def white_gaussian_noise(size: int) -> npt.NDArray[np.float64]:
    mean = 0
    std = 1
    return np.random.normal(mean, std, size=size)


def white_uniform_noise(size: int) -> npt.NDArray[np.float64]:
    return np.random.uniform(-1, 1, size=size)


def plot(f, x=None, title: str = "", xlabel: str = "", ylabel: str = ""):
    ax = plt.subplot()
    if x is None:
        ax.plot(f)
    else:
        ax.plot(x, f)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()


def ac_range(x: npt.NDArray[np.float64], r: npt.NDArray):
    """Returns autocorrelated signal in the provided range

    Args:
        x: Signal to autocorrelate
        r: Range to return autocorrelation for
    """
    ac = sm.tsa.acf(x)

    # Add negative side of correlation function
    ac_neg = ac[1:][::-1]
    ac = np.concatenate([ac_neg, ac])
    center = len(ac) // 2
    return ac[center + r[0] : center + r[-1] + 1]


def subtask_a() -> None:
    plot(white_binary_noise(100), title="Problem 1a): Binary noise")
    plot(white_gaussian_noise(100), title="Problem 1a): Gaussian noise")
    plot(white_uniform_noise(100), title="Problem 1a): Uniform noise")


def subtask_c() -> None:
    length = 20000
    r = np.arange(-10, 11)  # [-10,10]

    x_bin = white_binary_noise(length)
    m_x_bin = np.mean(x_bin)
    print(f"Binary noise mean: {m_x_bin}")
    plot(
        ac_range(x_bin, r),
        r,
        title="Binary noise autocorrelation",
    )

    x_gauss = white_gaussian_noise(length)
    m_x_gauss = np.mean(x_gauss)
    print(f"Gaussian noise mean: {m_x_gauss}")
    plot(
        ac_range(x_gauss, r),
        r,
        title="Gaussian noise autocorrelation",
    )

    x_unif = white_binary_noise(length)
    m_x_unif = np.mean(x_unif)
    print(f"Uniform noise mean: {m_x_unif}")
    plot(
        ac_range(x_gauss, r),
        r,
        title="Uniform noise autocorrelation",
    )


def main() -> None:
    subtask_a()
    subtask_c()


if __name__ == "__main__":
    main()
