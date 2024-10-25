import discreteSignal as ds
import continuosSignal as cs
from LTI_Continuous import *
from LTI_Discrete import *
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    # Discrete Signals
    img_root_path_discrete = './Discrete Signal Plots'
    os.makedirs(img_root_path_discrete, exist_ok=True)
    input_signal = ds.DiscreteSignal(5)
    input_signal.set_value_at_time(0, 0.5)
    input_signal.set_value_at_time(1, 2)
    impulse_response = ds.DiscreteSignal(5)
    impulse_response.set_value_at_time(0, 1)
    impulse_response.set_value_at_time(1, 1)
    impulse_response.set_value_at_time(2, 1)
    lti = LTI_Discrete(impulse_response)
    unit_impulses,coefficients = lti.linear_combination_of_impulses(input_signal)
    unit_impluse_responses,output_signal = lti.output(input_signal)
    plot_all_discrete(unit_impulses,final_signal=input_signal,coefficients=coefficients,titles=[f"δ[n-({i})]x[{i}]" for i in range(-input_signal.INF,input_signal.INF+1)],saveTo=f'{img_root_path_discrete}/unit_impulses.png')
    plot_all_discrete(unit_impluse_responses,final_signal=output_signal,titles=[f"h[n-({i})]x[{i}]" for i in range(-output_signal.INF,output_signal.INF+1)],saveTo=f'{img_root_path_discrete}/unit_impulse_responses.png')

    # Continuous Signals
    img_root_path_continuous = './Continuous Signal Plots'
    os.makedirs(img_root_path_continuous, exist_ok=True)
    input_signal = cs.ContinuousSignal(lambda x: np.exp(-x)*(x>=0))
    impulse_response = cs.ContinuousSignal(lambda x: np.where(x>=0,1,0))
    lti = LTI_Continuous(impulse_response)
    delta = 0.5
    delayed_signals,coefficients = lti.linear_combination_of_impulses(input_signal,delta)
    plot_all_continuous(delayed_signals,delta=delta,coefficients=coefficients,saveTo=f'{img_root_path_continuous}/lti_input.png')
    output,delayed_impulses,coefficients = lti.output_approx(input_signal,delta)
    plot_all_continuous(delayed_impulses,delta=delta,coefficients=coefficients,saveTo=f'{img_root_path_continuous}/lti_output.png')
    deltas = [0.5,0.1,0.05,0.01]
    plot_compare_input(input_signal=input_signal,deltas=deltas,saveTo=f'{img_root_path_continuous}/lti_compare_input.png')
    plot_compare_output(input_signal=input_signal,impulse_response=impulse_response,deltas=deltas,saveTo=f'{img_root_path_continuous}/lti_compare_output.png')

main()