import numpy as np
import matplotlib.pyplot as plt


class DiscreteSignal:
    def __init__(self,INF) -> None:
        self.INF = INF
        self.values= np.zeros(2 * INF + 1)
    
    def set_value_at_time(self, time, value):
        self.values[self.INF + time] = value

    def shift_signal(self,shift):
        shifted_signal = np.roll(self.values, shift)
        new_signal=DiscreteSignal(self.INF)
        new_signal.values=shifted_signal
        return new_signal
    
    def add(self,other):
        new_signal=DiscreteSignal(self.INF)
        new_signal.values=self.values + other.values
        return new_signal
    
    def multiply(self,other):
        new_signal=DiscreteSignal(self.INF)
        new_signal.values=self.values * other.values
        return new_signal
    
    def multiply_const_factor(self, scaler):
        new_signal=DiscreteSignal(self.INF)
        new_signal.values=self.values * scaler
        return new_signal
    
    def plot(self):
        plt.figure(figsize=(8, 5))
        plt.xticks(np.arange(-self.INF, self.INF + 1, 1))
        y_range = (-1, max(np.max(self.values), 3) + 1)
        plt.ylim(*y_range)
        plt.stem(np.arange(-self.INF, self.INF + 1, 1), self.values)
        plt.title('Original Signal(x[n])')
        plt.xlabel('n (Time Index)')
        plt.ylabel('x[n]')
        plt.grid(True)
        plt.show()
        #plt.savefig('x[n].png')

def plot(
        signal, 
        title=None, 
        y_range=(-1, 4), 
        figsize = (8, 5),
        x_label='n (Time Index)',
        y_label='x[n]',
        saveTo=None,
        INF=8
    ):
    plt.figure(figsize=figsize)
    plt.xticks(np.arange(-INF, INF + 1, 1))
    
    y_range = (y_range[0], max(np.max(signal), y_range[1]) + 1)
    # set y range of 
    plt.ylim(*y_range)
    plt.stem(np.arange(-INF, INF + 1, 1), signal)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    if saveTo is not None:
        plt.savefig(saveTo)
    # plt.show()

def main():
    signal = DiscreteSignal(5)
    signal.set_value_at_time(0, 0.5)
    signal.set_value_at_time(1, 2)
    # signal.plot()
    signal2=signal.shift_signal(-2)
    plot(signal.values, title='Original Signal(x[n])', saveTo='x[n].png', INF=5)
    plot(signal2.values, title='x[n+2]', saveTo='x[n-2].png', INF=5)
main()

