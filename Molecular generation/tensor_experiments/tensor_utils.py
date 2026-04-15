import numpy as np
from scipy.linalg import orth

def glarm_subspace_estimation(Y_list, p0, q0, max_iter=5, verbose=True):
    """
    Extract row and column loading matrices R and C using iterative GLARM-style method.
    
    Parameters:
    - Y_list: list of (p, q) matrices, length N. Here p=q=50.
    - p0: rank for row loading matrix R (<= p)
    - q0: rank for column loading matrix C (<= q)
    - max_iter: number of alternating iterations
    
    Returns:
    - R_hat: (p, p0), estimated row loading matrix
    - C_hat: (q, q0), estimated column loading matrix
    """
    N = len(Y_list)
    p, q = Y_list[0].shape
    
    assert all(Y.shape == (p, q) for Y in Y_list), "All matrices must be (p, q)"
    

    YtYt_avg = np.zeros((q, q))
    for Y in Y_list:
        YtYt_avg += Y.T @ Y
    YtYt_avg /= N

    eigvals_C, eigvecs_C = np.linalg.eigh(YtYt_avg)  
    C_hat = eigvecs_C[:, -q0:]  # (q, q0)
    C_hat = orth(C_hat) 

    if verbose:
        print("Initialized C_hat from avg(Y_t^T Y_t)")


    for iter_idx in range(max_iter):

        YCCTY_avg = np.zeros((p, p))
        for Y in Y_list:
            proj_C = C_hat @ C_hat.T  # (q, q)
            Y_proj = Y @ proj_C      # (p, q)
            YCCTY_avg += Y_proj @ Y_proj.T
        YCCTY_avg /= N

        eigvals_R, eigvecs_R = np.linalg.eigh(YCCTY_avg)
        R_hat = eigvecs_R[:, -p0:]  # (p, p0)
        R_hat = orth(R_hat)

        YTRRTY_avg = np.zeros((q, q))
        for Y in Y_list:
            proj_R = R_hat @ R_hat.T  # (p, p)
            Y_proj = proj_R @ Y       # (p, q)
            YTRRTY_avg += Y_proj.T @ Y_proj
        YTRRTY_avg /= N

        eigvals_C, eigvecs_C = np.linalg.eigh(YTRRTY_avg)
        C_hat = eigvecs_C[:, -q0:]  # (q, q0)
        C_hat = orth(C_hat)

        if verbose:
            print(f"Iter {iter_idx+1}: R_hat and C_hat updated.")

    return R_hat, C_hat  # (p, p0), (q, q0)


def upsample_matrix(Y, P):
    return P @ Y @ P.T

def downsample_matrix(Y_tgt, P):
    return P.T @ Y_tgt @ P

def row_orthonormal_matrix(m: int, n: int, seed: int = 0) -> np.ndarray:
    assert m <= n, "need m <= n"
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, m))      
    Q, _ = np.linalg.qr(A, mode='reduced') 
    P = Q.T                             

    err = np.linalg.norm(P @ P.T - np.eye(m))
    return P

def subspace_distance(A, B):
        """
        Compute normalized Frobenius distance between two subspaces.
        A, B: (n, r) matrices with full column rank.
        """
        # Orthogonalize columns
        P_A = orth(A)  # or A @ A.T if already orthonormal
        P_B = orth(B)
        
        # Projection matrices
        Proj_A = P_A @ P_A.T  # (n, n)
        Proj_B = P_B @ P_B.T  # (n, n)
        
        # Distance
        r = A.shape[1]
        dist = np.linalg.norm(Proj_A - Proj_B, 'fro') / np.sqrt(2 * r)
        return dist

def normalize_svd_preserve_subspace_batch(Y, eps=1e-8):
    if Y.ndim == 3:  # (N, H, W)
        N, H, W = Y.shape
        out = np.empty_like(Y, dtype=np.float32)
        for i in range(N):
            X = Y[i].astype(np.float64, copy=False)
            U, S, Vt = np.linalg.svd(X, full_matrices=False)
            S_min, S_max = S.min(), S.max()
            if S_max - S_min < eps:
                S_norm = np.zeros_like(S)
            else:
                S_norm = 2 * (S - S_min) / (S_max - S_min)
            out[i] = (U * S_norm) @ Vt
        return out

    elif Y.ndim == 4:  # (N, C, H, W)
        N, C, H, W = Y.shape
        out = np.empty_like(Y, dtype=np.float32)
        for i in range(N):
            for c in range(C):
                X = Y[i, c].astype(np.float64, copy=False)  # (H, W)
                U, S, Vt = np.linalg.svd(X, full_matrices=False)
                S_min, S_max = S.min(), S.max()
                if S_max - S_min < eps:
                    S_norm = np.zeros_like(S)
                else:
                    S_norm = 2 * (S - S_min) / (S_max - S_min)
                out[i, c] = (U * S_norm) @ Vt
        return out

    else:
        raise ValueError(f"Unsupported shape {Y.shape}. Expect (N,H,W) or (N,C,H,W).")


from scipy.linalg import sqrtm

def tucker_core_vectors(X, A1, A2, A3):
    """
    X:  (N, C, H, W)
    A1: (C, c)
    A2: (H, h)
    A3: (W, w)
    return: Z ∈ R^{N × (c*h*w)}
    """
    X = np.asarray(X, dtype=np.float32)
    A1 = np.asarray(A1, dtype=np.float32)
    A2 = np.asarray(A2, dtype=np.float32)
    A3 = np.asarray(A3, dtype=np.float32)

    N, C, H, W = X.shape
    c = A1.shape[1]; h = A2.shape[1]; w = A3.shape[1]

    Z = np.empty((N, c*h*w), dtype=np.float32)

    for i in range(N):
        x = X[i]                  # (C,H,W)
        # mode-1 × A1^T  等价于 tensordot(x, A1, ([0],[0]))
        z = np.tensordot(x, A1, axes=([0],[0]))   # (H, W, c)
        z = np.tensordot(z, A2, axes=([0],[0]))   # (W, c, h)
        z = np.tensordot(z, A3, axes=([0],[0]))   # (c, h, w)
        Z[i] = z.reshape(-1)
    return Z

def frechet_distance(mu1, cov1, mu2, cov2, eps=1e-6):
    mu1 = np.asarray(mu1, dtype=np.float32)
    mu2 = np.asarray(mu2, dtype=np.float32)
    cov1 = np.asarray(cov1, dtype=np.float32)
    cov2 = np.asarray(cov2, dtype=np.float32)

    cov1 = cov1 + eps * np.eye(cov1.shape[0], dtype=np.float32)
    cov2 = cov2 + eps * np.eye(cov2.shape[0], dtype=np.float32)

    diff = mu1 - mu2
    cov_prod = cov1.dot(cov2)


    covmean = sqrtm(cov_prod)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return float(diff.dot(diff) + np.trace(cov1 + cov2 - 2.0 * covmean))

def compute_frechet_distance_with_subspace(
    generated_data_set,     # (N, C, H, W)
    test_data_set,          # (n, C, H, W)
    A1, A2, A3,             # (C,c), (H,h), (W,w)
    eps=1e-6
):
    # to numpy, float64 for stability
    gen = np.asarray(generated_data_set, dtype=np.float64)
    real = np.asarray(test_data_set, dtype=np.float64)

    Z_real = tucker_core_vectors(real, A1, A2, A3)  # (n, r)
    Z_fake = tucker_core_vectors(gen,  A1, A2, A3)  # (N, r)

    mu_real = Z_real.mean(axis=0)
    mu_fake = Z_fake.mean(axis=0)

    cov_real = np.cov(Z_real, rowvar=False)
    cov_fake = np.cov(Z_fake, rowvar=False)

    fd = frechet_distance(mu_real, cov_real, mu_fake, cov_fake, eps=eps)
    return fd, mu_real, cov_real, mu_fake, cov_fake
