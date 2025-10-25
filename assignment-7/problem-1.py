import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy.fft
import scipy.signal
import statsmodels.api as sm


def white_binary_noise(size: int) -> npt.NDArray[np.float64]:
    return np.random.choice([-1, 1], size=size)


def white_gaussian_noise(
    size: int, std: float = 1.0
) -> npt.NDArray[np.float64]:
    mean = 0
    return np.random.normal(mean, std, size=size)


def white_uniform_noise(size: int) -> npt.NDArray[np.float64]:
    return np.random.uniform(-1, 1, size=size)


def plot(
    *f,
    x: npt.NDArray = None,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
):
    ax = plt.subplot()
    if x is None:
        ax.plot(np.arange(max([len(ff) for ff in f])), *f)
    else:
        ax.plot(x, *f)
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


def problem_1_subtask_a() -> None:
    plot(white_binary_noise(100), title="Problem 1a): Binary noise")
    plot(white_gaussian_noise(100), title="Problem 1a): Gaussian noise")
    plot(white_uniform_noise(100), title="Problem 1a): Uniform noise")


def problem_1_subtask_c() -> None:
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


def normalize(x: npt.NDArray) -> npt.NDArray:
    return x / np.max(x)


def y_xx(r: npt.NDArray):
    return 3 / 4 * (-1 / 2) ** np.abs(r)


def Gamma(w: npt.NDArray):
    G = 9 / 16 * 1 / ((5 / 4) + np.cos(w))
    return normalize(G)


def problem_2_subtask_c(run: int):
    N = 20000
    w = white_gaussian_noise(N, 3 / 4)
    x = scipy.signal.lfilter([1], [1, 1 / 2], w)

    r = np.arange(-10, 11)
    e_mean = np.mean(x)
    e_ac = ac_range(x, r)
    nfft = 50
    e_pds = normalize(scipy.fft.fftshift(scipy.fft.fft(e_ac, n=nfft)))
    print(f"Run {run}: Mean={e_mean}")

    ax = plt.subplot()
    plt.title(f"Run {run}: Autocorrelation")
    ax.plot(r, normalize(y_xx(r)), "C2", linewidth=3, label="Theoretical")
    ax.stem(r, normalize(e_ac), label="Estimated")
    ax.legend()
    plt.show()

    omega = np.linspace(-np.pi, np.pi, nfft, endpoint=False)
    plt.title(f"Run {run}: Power density spectrum")
    ax = plt.subplot()
    ax.plot(omega, Gamma(omega), "C2", linewidth=3, label="Theoretical")
    ax.stem(omega, np.abs(e_pds), label="Estimated")
    ax.legend()
    plt.show()


def main() -> None:
    problem_1_subtask_a()
    problem_1_subtask_c()
    for i in range(1, 4):
        problem_2_subtask_c(i)


if __name__ == "__main__":
    main()
