import matplotlib.pyplot as plt
import numpy as np
import scipy.signal


def main():
    b_u = [0.95, -0.95]
    a_u = [1, -0.9]

    b_l = [0.05, 0.05]
    a_l = a_u

    w_1, h_1 = scipy.signal.freqz(b_u, a_u)
    w_2, h_2 = scipy.signal.freqz(b_l, a_l)

    plt.title("Frequency responses")
    ax = plt.subplot()
    ax.plot(w_1, h_1, label="H_upper")
    ax.plot(w_2, h_2, label="H_lower")
    ax.set_ylabel("Magnitude")
    ax.set_xlabel("ω")
    ax.legend()
    ax.grid(True)
    plt.xticks(
        np.arange(0, np.pi + np.pi / 2, step=(np.pi / 2)),
        ["0", "π/2", "π"],
    )
    plt.show()


if __name__ == "__main__":
    main()
