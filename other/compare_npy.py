 
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

# Load the saved fio.estimates object
file_path = './results/fiola_result_ptr.npy'
# file_path = './results/fiola_result_ptr.npy'
fio_estimates = np.load(file_path, allow_pickle=True).item()

# Print the type of fio_estimates
print("Type of fio_estimates:", type(fio_estimates))

# Display all attributes and their values
attributes = vars(fio_estimates)
print("\nAttributes and their values:")
for attr, value in attributes.items():
    print(f"{attr}: {value}")

# Display specific attributes
if hasattr(fio_estimates, 't'):
    print("\nAttribute 't':")
    print(fio_estimates.t)

if hasattr(fio_estimates, 'index'):
    print("\nAttribute 'index':")
    print(fio_estimates.index)

if hasattr(fio_estimates, 'trace'):
    print("\nAttribute 'trace':")
    print(fio_estimates.trace)

if hasattr(fio_estimates, 'trace_deconvolved'):
    print("\nAttribute 'trace_deconvolved':")
    print(fio_estimates.trace_deconvolved)
