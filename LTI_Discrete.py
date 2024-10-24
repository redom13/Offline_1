import numpy as np
import matplotlib.pyplot as plt
import discreteSignal as ds

class LTI_Discrete:
    def __init__(self,impulse_response) -> None:
        self.impulse_response = impulse_response # h[n]
    
    def linear_combination_of_impulses(self,input_signal):
        INF = input_signal.INF
        undelayed_unit_impulse = ds.DiscreteSignal(INF) # delta[n]
        undelayed_unit_impulse.set_value_at_time(0,1)
        coefficients = input_signal.values
        unit_impulses = []
        for i in range(-INF,INF+1):
            unit_impulses.append(undelayed_unit_impulse.shift_signal(i))
        
        return unit_impulses,coefficients
    
    def output(self,input_signal):
        INF = input_signal.INF
        unit_impulses,coefficients = self.linear_combination_of_impulses(input_signal)
        output_signal = ds.DiscreteSignal(INF)
        constituents = []
        for i in range(-INF,INF+1):
            constituent = self.impulse_response.shift_signal(i).multiply_const_factor(coefficients[i+INF])
            constituents.append(constituent)
            output_signal = output_signal.add(constituent)
        
        return constituents,output_signal
    
def plot_all(
        signals,
        final_signal=None,
        coefficients=None,
        titles=None,
        y_range=(-1, 4),
        figsize=(15, 10),
        saveTo=None
):
    num_plots = len(signals) + (1 if final_signal is not None else 0)
    num_cols = 3
    num_rows = (num_plots + num_cols - 1) // num_cols  # Calculate the number of rows needed

    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()  # Flatten the 2D array of axes to 1D for easy iteration

    for i, signal in enumerate(signals):
        ax = axes[i]
        x = np.arange(-signal.INF, signal.INF + 1, 1)
        y = signal.values * coefficients[i] if coefficients is not None else signal.values
        ax.stem(x, y)
        ax.set_xticks(np.arange(-signal.INF, signal.INF + 1, 1))
        ax.set_ylim(*y_range)
        if titles and i < len(titles):
            ax.set_title(titles[i])
        ax.set_xlabel('n (Time Index)')
        ax.set_ylabel('x[n]')
        ax.grid(True)
    
    if final_signal is not None:
        print(final_signal.values)
        ax = axes[num_plots -1]
        x = np.arange(-final_signal.INF, final_signal.INF + 1, 1)
        y = final_signal.values
        ax.stem(x, y)
        ax.set_xticks(np.arange(-final_signal.INF, final_signal.INF + 1, 1))
        ax.set_ylim(*y_range)
        ax.set_title('Final Signal')
        ax.set_xlabel('n (Time Index)')
        ax.set_ylabel('x[n]')
        ax.grid(True)

    # Hide any unused subplots
    for j in range(len(signals) + (1 if final_signal is not None else 0), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    if saveTo is not None:
        plt.savefig(saveTo)
    else:
        plt.show()

def main():
    img_root_path = '.'
    input_signal = ds.DiscreteSignal(5)
    input_signal.set_value_at_time(0, 0.5)
    input_signal.set_value_at_time(1, 2)
    impulse_response = ds.DiscreteSignal(5)
    impulse_response.set_value_at_time(0, 1)
    impulse_response.set_value_at_time(1, 1)
    impulse_response.set_value_at_time(2, 1)
    lti = LTI_Discrete(impulse_response)
    unit_impulses,coefficients = lti.linear_combination_of_impulses(input_signal)
    # unit_impluse_responses,c = lti.linear_combination_of_impulses(impulse_response)
    unit_impluse_responses,output_signal = lti.output(input_signal)
    plot_all(unit_impulses,final_signal=input_signal,coefficients=coefficients,titles=[f"δ[n-({i})]x[{i}]" for i in range(-input_signal.INF,input_signal.INF+1)],saveTo=f'{img_root_path}/unit_impulses.png')
    plot_all(unit_impluse_responses,final_signal=output_signal,titles=[f"h[n-({i})]x[{i}]" for i in range(-output_signal.INF,output_signal.INF+1)],saveTo=f'{img_root_path}/unit_impulse_responses.png')
main()