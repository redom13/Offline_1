import numpy as np
import matplotlib.pyplot as plt
import continuosSignal as cs
import math

class LTI_Continuous:
    def __init__(self,impulse_response,INF=3):
        self.impulse_response = impulse_response
        self.INF = INF

    def linear_combination_of_impulses(self,input_signal,delta):
        undelayed_unit_impulse = cs.ContinuousSignal(lambda x: np.where((x >= 0) & (x < delta), 1.0 / delta, 0))
        n_val = math.floor(float(self.INF)/delta)
        indices = np.arange(-n_val*delta,n_val*delta,delta,dtype=float)
        coefficients = np.zeros(2*n_val)
        for i in range(2*n_val):
            coefficients[i] = input_signal.func(indices[i])
        delayed_signals = [undelayed_unit_impulse.shift(indices[i]) for i in range(len(indices))]
        # print(coefficients)
        return delayed_signals,coefficients
    
    def output_approx(self,input_signal,delta):
        delayed_signals,coefficients = self.linear_combination_of_impulses(input_signal,delta)
        n_val = math.floor(self.INF/delta)
        indices = np.arange(-n_val*delta,n_val*delta,delta,dtype=float)
        delayed_impulses = [self.impulse_response.shift(indices[i]) for i in range(len(indices))]
        output = cs.ContinuousSignal(lambda x: sum([delayed_impulses[i].func(x)*coefficients[i]*delta for i in range(len(indices))]))
        return output,delayed_impulses,coefficients

    
def plot_all_continuous(
        signals,
        INF=3,
        delta=0.5,
        # final_signal=None,
        coefficients=None,
        titles=None,
        y_range=(-0.05, 1.05),
        figsize=(15, 10),
        saveTo=None
):
    num_plots = len(signals) + 1
    num_cols = 3
    num_rows = (num_plots + num_cols - 1) // num_cols  # Calculate the number of rows needed

    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()  # Flatten the 2D array of axes to 1D for easy iteration
    x_high = INF+0.2
    final_output = np.zeros(1001)

    for i, signal in enumerate(signals):
        ax = axes[i]
        x = np.linspace(-INF, INF,1000)
        y = signal.func(x) * coefficients[i]*delta if coefficients is not None else signal.values
        # print(signal.func(x),coefficients[i])
        x = np.append(x, INF)
        y = np.append(y, 0)
        final_output+=y
        ax.plot(x, y)
        ax.set_xlim(-x_high, x_high)
        # ax.set_ylim(*y_range)
        ax.set_xticks(np.arange(-INF, INF + 1, 1))  # Set x-ticks at intervals of 1
        ax.set_yticks(np.arange(0, y_range[1], 0.5))  # Set y-ticks at intervals of 0.5
        if titles and i < len(titles):
            ax.set_title(titles[i])
        ax.set_xlabel('t (Time)')
        ax.set_ylabel('x(t)')
        ax.grid(True)

    # if final_signal is not None:
    #     ax = axes[num_plots - 1]
    #     x = np.linspace(-final_signal.INF, final_signal.INF, len(final_signal.values))
    #     y = final_signal.values
    #     ax.plot(x, y)
    #     ax.set_xlim(-final_signal.INF, final_signal.INF)
    #     ax.set_ylim(*y_range)
    #     ax.set_title('Final Signal')
    #     ax.set_xlabel('t (Time)')
    #     ax.set_ylabel('x(t)')
    #     ax.grid(True)
    ax = axes[num_plots - 1]
    x = np.linspace(-INF, INF, 1000)
    x = np.append(x, INF)
    ax.plot(x, final_output)
    ax.set_xlim(-x_high, x_high)
    # ax.set_ylim(*y_range)
    ax.set_title('Final Signal')
    ax.set_xlabel('t (Time)')
    ax.set_ylabel('x(t)')
    ax.grid(True)

    for j in range(len(signals) + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    if saveTo is not None:
        plt.savefig(saveTo)
    else:
        plt.show()

def plot_compare_input(
        input_signal,
        # actual_signal,
        deltas,
        INF=3,
        y_range=(-0.05, 1.05),
        figsize=(15, 10),
        saveTo=None
):
    num_plots = len(deltas)
    num_cols = 2
    num_rows = (num_plots + num_cols - 1) // num_cols  # Calculate the number of rows needed
    fig,axes =plt.subplots(num_rows,num_cols,figsize=figsize)
    axes = axes.flatten()
    x_high = INF+0.2
    lti = LTI_Continuous(cs.ContinuousSignal(lambda x: np.where(x>=0,1,0)))
    # print("Input Signal",input_signal.func(np.linspace(-INF,INF,1000)))
    for i,delta in enumerate(deltas):
        ax = axes[i]
        x = np.linspace(-INF, INF,1000)
        delayed_signals,coefficients = lti.linear_combination_of_impulses(input_signal,delta)
        y_approx = sum([delayed_signals[i].func(x)*coefficients[i]*delta for i in range(len(delayed_signals))])
        y_actual = input_signal.func(x)
        ax.set_xlim(-x_high, x_high)
        ax.plot(x,y_approx,label='y_approx')
        ax.plot(x,y_actual,label='y_actual')
        ax.legend()
        ax.set_xticks(np.arange(-INF, INF + 1, 1))
        ax.set_yticks(np.arange(0, y_range[1], 0.5))
        ax.set_title(f'delta={delta}')
        ax.grid(True)

    for j in range(len(deltas),len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()

    if saveTo is not None:
        plt.savefig(saveTo)
    else:
        plt.show()

def plot_compare_output(
        input_signal,
        impulse_response,
        deltas=[],
        INF=3,
        figsize=(15, 10),
        saveTo=None
):
    num_plots = len(deltas)
    num_cols = 2
    num_rows = (num_plots + num_cols - 1) // num_cols  # Calculate the number of rows needed
    fig,axes =plt.subplots(num_rows,num_cols,figsize=figsize)
    axes = axes.flatten()
    x_high = INF+0.2
    x_values = np.linspace(-INF,INF,1000)
    y_values = input_signal.func(x_values)
    output_original= np.cumsum(y_values)*(x_values[1]-x_values[0])
    lti = LTI_Continuous(impulse_response)
    for i,delta in enumerate(deltas):
        ax = axes[i]
        output,delayed_impulses,coefficients = lti.output_approx(input_signal,delta)
        y_approx = output.func(x_values)
        ax.set_xlim(-x_high, x_high)
        ax.plot(x_values,y_approx,label='y_approx')
        ax.plot(x_values,output_original,label='y_actual')
        ax.legend()
        ax.set_xticks(np.arange(-INF, INF + 1, 1))
        ax.set_title(f'delta={delta}')
        ax.grid(True)

    for j in range(len(deltas),len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    if saveTo is not None:
        plt.savefig(saveTo)
    else:
        plt.show()

# def main():
#     img_root_path = '.'
#     input_signal = cs.ContinuousSignal(lambda x: np.exp(-x)*(x>=0))
#     impulse_response = cs.ContinuousSignal(lambda x: np.where(x>=0,1,0))
#     lti = LTI_Continuous(impulse_response)
#     # input_signal = cs.ContinuousSignal(lambda x: 1 if x>=0 and x<=1 else 0)
#     delta = 0.5
#     delayed_signals,coefficients = lti.linear_combination_of_impulses(input_signal,delta)
#     plot_all_continuous(delayed_signals,delta=delta,coefficients=coefficients,saveTo=f'{img_root_path}/lti.png')
#     output,delayed_impulses,coefficients = lti.output_approx(input_signal,delta)
#     plot_all_continuous(delayed_impulses,delta=delta,coefficients=coefficients,saveTo=f'{img_root_path}/lti_output.png')
#     # output_original = cs.ContinuousSignal(lambda x: np.ones_like(x)-np.exp(-x) if x>=0 else 0)
#     # plot_compare_input(input_signal=input_signal,deltas=[0.5,0.1,0.05,0.01],saveTo=f'{img_root_path}/lti_compare.png')
#     plot_compare_output(input_signal=input_signal,impulse_response=impulse_response,deltas=[0.5,0.1,0.05,0.01],saveTo=f'{img_root_path}/lti_compare_output.png')
# main()