
import time

import h5py

h5_file_path = "/Users/henryh/Desktop/eai-repos/helios-repos/helios/sample_5187.h5"

total_time = 0
num_iterations = 100

for i in range(num_iterations):
    time_start = time.time()
    with h5py.File(h5_file_path, "r") as f:
        sample_dict = {k: v[()] for k, v in f.items()}
    time_end = time.time()
    total_time += (time_end - time_start)

average_time = total_time / num_iterations
print(f"Average time taken over {num_iterations} iterations: {average_time} seconds")
