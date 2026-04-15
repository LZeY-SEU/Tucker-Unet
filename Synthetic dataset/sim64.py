"""
TuckerUnet: synthetic dataset
"""
import numpy as np

def generate_synthetic_matrix_returns(N=10000, p=100, q=100, p0=4, q0=4, seed=42):
    """
    Generate synthetic matrix-valued returns using a bilinear factor model:
        Y_t = R @ F_t @ C.T + E_t
    
    Parameters:
    - N: number of samples (time periods)
    - p, q: dimensions of observed matrix (e.g., p assets × q features)
    - p0, q0: dimensions of latent factor matrix
    - seed: random seed
    
    Returns:
    - Y: (N, p, q) array of matrix returns
    - metadata: dict containing R, C, F_mean, F_sigma, sigma_E, etc.
    """
    np.random.seed(seed)
    
    # ----------------------------------------
    # 1. Latent Matrix Factor F_t ~ N(F_mean, F_Sigma)
    #    Here we assume each element has independent mean and scale
    # ----------------------------------------
    # Mean matrix of F_t
    F_mean = np.random.uniform(0, 0.1, size=(p0, q0))  # (p0, q0)
    
    # Standard deviation: scales with mean
    F_sigma = 1.5 * F_mean  # (p0, q0)
    
    # Generate F_samples: (N, p0, q0)
    # Independent across elements
    F_noise = np.random.randn(N, p0, q0) * F_sigma  # broadcast
    F_samples = F_mean + F_noise  # (N, p0, q0)

    # ----------------------------------------
    # 2. Factor Loadings: R (p x p0), C (q x q0)
    #    Generated i.i.d. from N(0,1), as in original A
    # ----------------------------------------
    R = np.random.normal(0, 1, size=(p, p0))    # (p, p0)
    C = np.random.normal(0, 1, size=(q, q0))    # (q, q0)

    # ----------------------------------------
    # 3. Idiosyncratic Noise E_t: (p, q), element-wise independent in the article
    #    sigma_eps[i,j] ~ Uniform(0, sigma)
    # ----------------------------------------
    sigma_eps =np.random.normal(0, 1) *3# for demo: setting homogeneous noise here
    
    # ----------------------------------------
    # 4. Generate Y_t = R @ F_t @ C.T + E_t
    # ----------------------------------------
    Y = np.zeros((N, p, q), dtype=np.float32)
    
    for t in range(N):
        Ft = F_samples[t]  # (p0, q0)
        
        # Bilinear mapping: Y_t = R @ F_t @ C.T
        Yt_signal = R @ Ft @ C.T  # (p, p0) @ (p0, q0) @ (q0, q) -> (p, q)
        
        # Idiosyncratic noise
        Et = np.random.randn(p, q) * sigma_eps  # (p, q)
        
        # Add signal and noise
        Y[t] = Yt_signal + Et

    # ----------------------------------------
    # 5. Metadata
    # ----------------------------------------
    metadata = {
        'F_mean': F_mean,        # (p0, q0)
        'F_sigma': F_sigma,      # (p0, q0)
        'R': R,                  # (p, p0)
        'C': C,                  # (q, q0)
        'sigma_eps': sigma_eps,  # (p, q)
        'N': N, 'p': p, 'q': q, 'p0': p0, 'q0': q0,
        'F_samples': F_samples   # (N, p0, q0), if needed
    }
    
    return Y, metadata


if __name__ == '__main__':
    #generate synthetic dataset
    from tensor_experiments.tensor_utils import *
    k_1=p0=8
    k_2=q0=8
    p_1=64
    q_1=64
    p_11=q_11=64
    channels=1  
    gpu_id = 0
    seed = 25
    epochs = 300  
    num_samples = None 


    Y, meta = generate_synthetic_matrix_returns(N=4096, p=64, q=64, p0=8, q0=8, seed=15) #seed in [5,10,15,20,25]
    Y = Y.astype('float32') / 20
    np.save('sim64.npy', Y)
    print("Saved sim data to 'sim64.npy' with shape:", Y.shape)

    Y_train_list = [Y[i].reshape(p_1,q_1) for i in range(Y.shape[0])]
    R_hat_1, C_hat_1 = glarm_subspace_estimation(Y_train_list, p0=p0, q0=q0, max_iter=5)

    #train Tucker diffusion model and sample

    from tensor_experiments.TuckerUnet_train import train_model_tucker
    data_path = r"C:\Users\PC\Desktop\Tensor Diffusion\Tucker-Unet\Synthetic dataset\sim64.npy" #change the path
    model_dir, sample_dir = train_model_tucker(
    data_path=data_path,
    seed=seed,
    num_samples=num_samples,  # 不在训练时生成
    gpu_id=gpu_id,
    epochs=epochs,
    U_init=R_hat_1,
    V_init=C_hat_1,
    k1=k_1,
    H=p_1,
    W=q_1,
    k2=k_2,
    channels=channels,
    train_UV = True
    )

    print(f"Training completed!")
    print(f"Model saved to: {model_dir}")
    print(f"Generated samples will be saved to: {sample_dir}")

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
    generated_flat=generated_flat.reshape(generated_flat.shape[0], 1, p_1, q_1) 
    print(f"Loaded generated data shape: {generated_flat.shape}") 
    Y_sim = generated_flat.reshape(-1, p_1, q_1)  

    A1=np.eye(channels)
    A2= orth(meta['R'])
    A3= orth(meta['C'])
    fd_far, mu_r2, cov_r2, mu_f2, cov_f2 = compute_frechet_distance_with_subspace(
        generated_data_set = Y_sim.reshape(-1,1,p_1,q_1),
        test_data_set  = Y.reshape(-1,1,p_1,q_1),
        A1=A1, A2=A2, A3=A3,
        eps=1e-6
    )
    print('core Frechet distance:')
    print(fd_far)