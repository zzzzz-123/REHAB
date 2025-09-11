import os
import numpy as np

path1 = r"C:\Users\Administrator\Desktop\Rehab_exercise\d02_processed_data"
sp = r'D:\pycharm project\Rehab-120\Rehab_training\data'  # The directory for saving the processed data
os.makedirs(sp, exist_ok=True)

window_size = 200  
stride = 40        

# 遍历每个文件
for file in os.listdir(path1):
    if not file.endswith('.npy'):
        continue  

    path = os.path.join(path1, file)
    all_samples1 = []  

    # load data (250, 880, 6)
    data1 = np.load(path)

    for i in range(data1.shape[0]): 
        sample = data1[i]  # shape: (880, 6)

        # sliding window
        for start in range(0, sample.shape[0] - window_size + 1, stride):
            end = start + window_size
            window = sample[start:end, :]  # shape: (200, 6)

            # zero-mean
            window = window - np.mean(window, axis=0, keepdims=True)

            all_samples1.append(window)

    # Convert to a numpy array
    all_samples1 = np.array(all_samples1)  # shape: (N, 200, 6)
    print(f"{file} -> Final total windowed samples: {all_samples1.shape}")

    # save data
    save_path = os.path.join(sp, file)

    np.save(save_path, all_samples1, allow_pickle=True)
