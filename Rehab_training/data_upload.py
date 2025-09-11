import os
import numpy as np

# path
data_dir = r"C:\Users\Administrator\Desktop\Rehab_exercise\d02_processed_data"
save_dir = r"C:\Users\Administrator\Desktop\Rehab_exercise\d02_processed_data"
os.makedirs(save_dir, exist_ok=True)

# Obtain the file list and sort it to ensure that the order matches
file_list = sorted([f for f in os.listdir(data_dir) if f.endswith('.npy')])

# Check whether it can be divided into 16 groups (with two files in each group)
assert len(file_list) == 32, f"The number of files should be 32. Currently, it is {len(file_list)}"
assert len(file_list) % 2 == 0, "The number of files is not even, so they cannot be paired in pairs"

# Group by group processing
for i in range(0, len(file_list), 2):
    file1 = os.path.join(data_dir, file_list[i])
    file2 = os.path.join(data_dir, file_list[i + 1])

    data1 = np.load(file1)  # shape: (N, 880, 6)
    data2 = np.load(file2)  # shape: (N, 880, 6)

    # Check that the number of samples is consistent
    assert data1.shape[0] == data2.shape[0], f"{file_list[i]} 与 {file_list[i+1]} 样本数不一致"

    # Channel splicing => (N, 880, 12)
    combined = np.concatenate((data1, data2), axis=2)

    # Save as a single action file
    action_idx = i // 2  
    save_path = os.path.join(save_dir, f'action_{action_idx}.npy')
    np.save(save_path, combined)

    print(f"已保存: action_{action_idx}.npy, shape: {combined.shape}")

