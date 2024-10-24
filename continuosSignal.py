import numpy as np
import matplotlib.pyplot as plt

class ContinuousSignal:
    def __init__(self,func) -> None:
        self.func = func

    def shift(self,shift):
        shifted_signal = lambda x: self.func(x-shift) # lambda for anonymous function,only arg is x
        # This could also be done by defining an explicit function which takes x
        new_signal = ContinuousSignal(shifted_signal)
        return new_signal
    
    def add(self,other):
        return ContinuousSignal(lambda x: self.func(x) + other.func(x))
    
    def multiply(self,other):
        return ContinuousSignal(lambda x: self.func(x) * other.func(x))
    
    def multiply_const_factor(self, scaler):
        return ContinuousSignal(lambda x: self.func(x)*scaler)
    
    def plot(self):
        x = np.linspace(-10, 10, 1000)
        plt.figure(figsize=(8, 5))
        plt.plot(x, self.func(x))
        plt.title('Original Signal(x(t))')
        plt.xlabel('t (Time)')
        plt.ylabel('x(t)')
        plt.grid(True)
        plt.ylim(-5,5)
        plt.show()
        #plt.savefig('x[n].png')

def plot(
        signal, 
        title=None, 
        lower_limit_x=-10,
        upper_limit_x=10,
        lower_limit_y=-5,
        upper_limit_y=5, 
        figsize = (8, 5),
        x_label='t (Time)',
        y_label='x(n)',
        saveTo=None
    ):
    x= np.linspace(lower_limit_x, upper_limit_x, 1000)
    plt.figure(figsize=figsize)
    plt.plot(x, signal.func(x))
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.ylim(lower_limit_y,upper_limit_y)
    if saveTo is not None:
        plt.savefig(saveTo)
    # plt.show()
    
def main():
    img_root_path = '.'
    signal = ContinuousSignal(lambda x: np.sin(x))
    # signal.plot()
    
    plot(signal.shift(2), title='x(t-2)', saveTo=f'{img_root_path}/x(t-2).png')
    
    plot(signal.shift(-2), title='x(t+2)', saveTo=f'{img_root_path}/x(t+2).png')
    
    plot(signal.shift(0), title='x(t+0)', saveTo=f'{img_root_path}/x(t+0).png')
    
    plot(signal.multiply_const_factor(2), title='2x(t)', saveTo=f'{img_root_path}/2x(t).png')
    
    plot(signal.multiply_const_factor(-2), title='-2x(t)', saveTo=f'{img_root_path}/-2x(t).png')
    
    plot(signal.add(ContinuousSignal(lambda x: np.cos(x))), title='x(t)+other(t)', saveTo=f'{img_root_path}/add.png')
    
    plot(signal.multiply(ContinuousSignal(lambda x: 0.01*np.exp(x))), title='x(t)*other(t)', saveTo=f'{img_root_path}/mul.png',lower_limit_y=-100,upper_limit_y=100)

main()