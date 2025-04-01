
import time

import h5py

h5_file_path = "/Users/henryh/Desktop/eai-repos/helios-repos/helios/sample_5187.h5"
h5_file_path = "/weka/dfive-default/helios/dataset/presto/h5py_data/latlon_sentinel1_sentinel2_l2a_worldcover/98856/sample_5187.h5"

total_time = 0
num_iterations = 100

times = []
for i in range(num_iterations):
    time_start = time.time()
    with h5py.File(h5_file_path, "r") as f:
        sample_dict = {k: v[()] for k, v in f.items()}
    time_end = time.time()
    times.append(time_end - time_start)

average_time = sum(times) / num_iterations
max_time = max(times)
min_time = min(times)
print(f"Average time taken over {num_iterations} iterations: {average_time} seconds")
print(f"Max time taken over {num_iterations} iterations: {max_time} seconds")
print(f"Min time taken over {num_iterations} iterations: {min_time} seconds")
