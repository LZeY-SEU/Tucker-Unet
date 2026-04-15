"""
TuckerUnet: molecular generation
"""
import torch
from torch_geometric.datasets import TUDataset
import random
import numpy as np
import tensorly as tl
import matplotlib.pyplot as plt
# Load the dataset
PROTEINS = torch.load(r"C:\Users\PC\Desktop\Tensor Diffusion\Tucker-Unet\Molecular generation\PROTEINS.pt") #change the path
PROTEINS.shape
# Download the labels
dataset = 'PROTEINS'
graphs = TUDataset(root='/tmp/' + dataset, name=dataset)
y = [graph.y for graph in graphs]
# Construct the tensor list
calss_0_id = np.array(y).reshape(-1)==0
PROTEINS_0 = PROTEINS[calss_0_id, :, :, :]
PROTEINS_list = np.array([np.array(PROTEINS_0[i, :, :, :]) for i in range(PROTEINS_0.shape[0])])
print(np.where([np.max(y)>2 for y in PROTEINS_list]))
PROTEINS_list = np.array([y for y in PROTEINS_list if np.max(y) <= 2])
print(np.where([np.max(y)>2 for y in PROTEINS_list]))


def reconstruction_error(Y, R, C):
    F = R.T @ Y @ C  # (p0, q0)
    Y_recon = R @ F @ C.T
    return np.linalg.norm(Y - Y_recon, 'fro') ** 2/np.linalg.norm(Y, 'fro') ** 2

if __name__ == '__main__':
    from tensor_experiments.tensor_utils import *
    k_1=p0=8 
    k_2=q0=8
    p_1=q_1=64 
    p_11=q_11=PROTEINS_list.shape[2] 
    channels=PROTEINS_list.shape[1]  
    gpu_id = 0
    seed = 1025
    epochs = 300        
    num_samples = None 

    from sklearn.model_selection import train_test_split

    # split data
    idx_train, idx_test = train_test_split(np.arange(PROTEINS_list.shape[0]), test_size=0.2, random_state=42)

    Y_train_flat = PROTEINS_list[idx_train].reshape(-1, p_11, q_11)
    Y_train_list = [Y_train_flat[i] for i in range(Y_train_flat.shape[0])]

    R_hat_1, C_hat_1 = glarm_subspace_estimation(Y_train_list, p0=p0, q0=q0, max_iter=5)

    #data pre-processing

    import os
    Y_train_flat_1 = PROTEINS_list[idx_train]

    P = row_orthonormal_matrix(m=p_11, n=p_1, seed=seed)  

    Y_up_channels = np.empty((Y_train_flat_1.shape[0], Y_train_flat_1.shape[1], p_1, q_1), dtype=np.float32)

    for c in range(Y_train_flat_1.shape[1]):  
        Y_c = Y_train_flat_1[:, c, :, :]

        Y_c_up = np.empty((Y_c.shape[0], p_1, q_1), dtype=np.float32)
        for i in range(Y_c.shape[0]):
            Y_c_up[i] = upsample_matrix(Y_c[i], P.T)

        Y_up_channels[:, c, :, :] = Y_c_up

    data_path = "training_data_temp.npy"
    os.makedirs(os.path.dirname(data_path) if os.path.dirname(data_path) else '.', exist_ok=True)
    np.save(data_path, Y_up_channels.astype(np.float32))
    R_hat=P.T@R_hat_1
    C_hat=P.T@C_hat_1


    from tensor_experiments.TuckerUnet_train import train_model_tucker
    data_path = r"C:\Users\PC\Desktop\Tensor Diffusion\Tucker-Unet\Molecular generation\training_data_temp.npy"
    model_dir, sample_dir = train_model_tucker(
        data_path=data_path,
        seed=seed,
        num_samples=num_samples, 
        gpu_id=gpu_id,
        epochs=epochs,
        U_init=R_hat,
        V_init=C_hat,
        k1=k_1,
        H=p_1,
        W=q_1,
        k2=k_2,
        channels=channels,
        train_UV = True
    )

    print(f"🎉 Training completed!")
    print(f"   Model saved to: {model_dir}")
    print(f"   Generated samples will be saved to: {sample_dir}")

    #merge samples and report
    import os
    npy_files = sorted(
        [os.path.join(sample_dir, f) for f in os.listdir(sample_dir) if f.endswith(".npy")]
    )

    if not npy_files:
        raise FileNotFoundError(f"❌ No .npy files found in {sample_dir}")

    arrays = []
    for file in npy_files:
        arr = np.load(file)
        arrays.append(arr)

    merged_array = np.concatenate(arrays, axis=0)

    generated_flat = merged_array
    generated_flat=generated_flat.reshape(generated_flat.shape[0], 2, p_1, q_1) 
    print(f"Loaded generated data shape: {generated_flat.shape}") 
    Y_sim = generated_flat.reshape(-1, p_1, q_1)  

    Y_sim = downsample_matrix(Y_sim, P.T)
    R_sim, C_sim = glarm_subspace_estimation(Y_sim, p0=8, q0=8, max_iter=5)

    total_error = 0.0
    n_test_matrices = 0
    for i in idx_test:
        for k in range(2):  # 两个通道
            Y_test = PROTEINS_list[i, k]  # (50,50)
            err = reconstruction_error(Y_test, R_sim, C_sim)
            total_error += err
            n_test_matrices += 1

    avg_recon_error = total_error / n_test_matrices
    print(f"Avg Test Reconstruction Error: {avg_recon_error:.4f}")