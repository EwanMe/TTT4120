import scipy.signal
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd


def transfer_func(alpha, R):
    b = np.zeros(R + 1)
    b[0] = 1.0
    b[R] = alpha
    a = np.zeros(R + 1)
    a[0] = 1
    return b, a


def subtask_c():
    N = 20
    R = 10
    alpha = 0.9

    b, a = transfer_func(alpha, R)

    lti = scipy.signal.dlti(b, a)
    _, h = scipy.signal.dimpulse(lti, n=N)
    w, H = scipy.signal.freqz(b, a, worN=1024)

    plt.title("Frequency responses")
    ax = plt.subplot()
    ax.plot(w, H, label="H(z)")
    ax.set_ylabel("Magnitude")
    ax.set_xlabel("ω")
    ax.legend()
    ax.grid(True)
    plt.show()

    plt.title("Unit pulse response h[n]")
    ax = plt.subplot()
    ax.stem(np.arange(N), np.squeeze(h))
    ax.set_xlabel("n (samples)")
    ax.set_ylabel("h[n]")
    ax.grid(True)
    plt.show()


def subtask_d():
    fs, x = scipy.io.wavfile.read("piano.wav")

    b, a = transfer_func(0.7, 2000)
    y = scipy.signal.lfilter(b, a, x)

    yscaled = y / np.max(y)
    sd.play(yscaled, fs)
    sd.wait()


def transfer_func_2(alpha: int | float, R: int, N: int):
    if R == 0 or N == 0:
        raise ValueError("R and N must be >1")

    b = np.zeros(R * N + 1)
    b[0] = 1.0
    b[R * N] = -(alpha**N)
    a = np.zeros(R + 1)
    a[0] = 1.0
    a[R] = -alpha
    return b, a


def subtask_e_f():
    alpha = 0.7
    R = 10
    N = 5
    impulse_len = 200
    a, b = transfer_func_2(alpha, R, N)

    lti = scipy.signal.dlti(b, a)
    _, h = scipy.signal.dimpulse(lti, n=impulse_len)
    w, H = scipy.signal.freqz(b, a, worN=1024)

    plt.title("Frequency responses")
    ax = plt.subplot()
    ax.plot(w, H, label="H(z)")
    ax.set_ylabel("Magnitude")
    ax.set_xlabel("ω")
    ax.legend()
    ax.grid(True)
    plt.show()

    plt.title("Unit pulse response h[n]")
    ax = plt.subplot()
    ax.stem(np.arange(impulse_len), np.squeeze(h))
    ax.set_xlabel("n (samples)")
    ax.set_ylabel("h[n]")
    ax.grid(True)
    plt.show()


def subtask_g():
    alpha = 0.7
    R = 3000
    N = 5
    b, a = transfer_func_2(alpha, R, N)

    fs, x = scipy.io.wavfile.read("piano.wav")

    y = scipy.signal.lfilter(b, a, x)

    yscaled = y / np.max(y)
    sd.play(yscaled, fs)
    sd.wait()


def main():
    subtask_c()
    subtask_d()
    subtask_e_f()
    subtask_g()


if __name__ == "__main__":
    main()
