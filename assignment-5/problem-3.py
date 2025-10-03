import scipy.signal
from scipy.io.wavfile import read
from scipy.signal import lfilter, freqz
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd


def subtask_c():
    R = 1000
    alpha = 0.9
    fs = 22050
    N = 2000

    b = np.zeros(R + 1)
    b[0] = 1.0
    b[R] = alpha
    a = np.zeros(R + 1)
    a[0] = 1

    lti = scipy.signal.dlti(b, a)
    _, h = scipy.signal.dimpulse(lti, n=N)
    w, H = scipy.signal.freqz(b, a, worN=1024, fs=fs)

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

    return b, a


def subtask_d(b, a):
    fs, x = read("piano.wav")

    y = lfilter(b, a, x)

    print("Playing *original* sound clip")
    xscaled = x / np.max(x)
    sd.play(xscaled, fs)  # play original sound

    input("Press a key to continue...")

    print("Playing *filtered* sound clip")
    yscaled = y / np.max(y)
    sd.play(yscaled, fs)
    sd.wait()


def main():
    b, a = subtask_c()
    subtask_d(b, a)


if __name__ == "__main__":
    main()
