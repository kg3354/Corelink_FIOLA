 
# import numpy as np
# import matplotlib.pyplot as plt
# from IPython.display import clear_output

# # Load the saved fio.estimates object
# file_path = '/Users/guobuzai/Projects/Corelink_FIOLA/results/fiola_result_ptr_1.npy'
# # file_path = './results/fiola_result_ptr.npy'
# fio_estimates = np.load(file_path, allow_pickle=True).item()

# # Print the type of fio_estimates
# print("Type of fio_estimates:", type(fio_estimates))

# # Display all attributes and their values
# attributes = vars(fio_estimates)
# print("\nAttributes and their values:")
# for attr, value in attributes.items():
#     print(f"{attr}: {value}")

# # Display specific attributes
# if hasattr(fio_estimates, 't'):
#     print("\nAttribute 't':")
#     print(fio_estimates.t)

# if hasattr(fio_estimates, 'index'):
#     print("\nAttribute 'index':")
#     print(fio_estimates.index)

# if hasattr(fio_estimates, 'trace'):
#     print("\nAttribute 'trace':")
#     print(fio_estimates.trace)

# if hasattr(fio_estimates, 'trace_deconvolved'):
#     print("\nAttribute 'trace_deconvolved':")
#     print(fio_estimates.trace_deconvolved)

import numpy as np
import sys

# Mock the fiola module
class MockFiola:
    class Estimates:
        pass

sys.modules['fiola'] = MockFiola

# Load the saved fio.estimates object
file_path = '/Users/guobuzai/Projects/Corelink_FIOLA/results/fiola_result_ptr_1.npy'
fio_estimates = np.load(file_path, allow_pickle=True).item()

# Print the type of fio_estimates
print("Type of fio_estimates:", type(fio_estimates))

# Display all attributes and their values
if isinstance(fio_estimates, dict):
    print("\nAttributes and their values (as a dictionary):")
    for key, value in fio_estimates.items():
        print(f"{key}: {value}")
else:
    attributes = dir(fio_estimates)
    print("\nAttributes and their values:")
    for attr in attributes:
        if not attr.startswith('__'):
            value = getattr(fio_estimates, attr)
            print(f"{attr}: {value}")

# Display specific attributes
for attr in ['t', 'index', 'trace', 'trace_deconvolved']:
    if isinstance(fio_estimates, dict):
        if attr in fio_estimates:
            print(f"\nAttribute '{attr}':")
            print(fio_estimates[attr])
    else:
        if hasattr(fio_estimates, attr):
            print(f"\nAttribute '{attr}':")
            print(getattr(fio_estimates, attr))
