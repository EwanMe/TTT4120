import matplotlib.pyplot as plt
import numpy as np


def Y(omega, M):
    return np.sin(omega * (M + 1 / 2)) / np.sin(omega / 2)


def subtask_b() -> None:
    M = 10
    print(1 * (M + 1 / 2))
    plt.plot([Y(x, M) for x in np.linspace(-np.pi, np.pi, 1000)])
    plt.show()


def main() -> None:
    subtask_b()


if __name__ == "__main__":
    main()
