import torch
import torch.nn.functional as F
import numpy as np


def matmul(A, B, method='naive', **kwargs):
    """
    Multiply two matrices.
    :param A: (N, M) torch tensor.
    :param B: (M, K) torch tensor.
    :param method:
    :return:
        Output matrix with shape (N, K)
    """
    method = method.lower()
    if method in ['naive', 'pytorch', 'torch']:
        return naive(A, B)
    elif method == 'svd':
        return svd(A, B, **kwargs)
    elif method in ['log', 'logmatmul']:
        return logmatmul(A, B, **kwargs)
    else:
        raise ValueError("Invalid [method] value: %s" % method)


def naive(A, B, **kwargs):
    return A @ B


def svd(A, B, rank_A=None, rank_B=None):
    """
    Apply low-rank approximation (SVD) to both matrix A and B with rank rank_A
    and rank_B respectively.
    :param A: (N, M) pytorch tensor
    :param B: (M, K) pytorch tensor
    :param rank_A: None or int. None means use original A matrix.
    :param rank_B: None or int. None means use original B matrix.
    :return: a (N, K) pytorch tensor
    """
    A_np = A.cpu().numpy()
    B_np = B.cpu().numpy()

    # Low-rank approximations
    if rank_A is not None:
        U_A, S_A, Vh_A = np.linalg.svd(A_np, full_matrices=False)
        U_A = U_A[:, :rank_A]
        S_A = S_A[:rank_A]
        Vh_A = Vh_A[:rank_A, :]
        A_approx = (U_A * S_A) @ Vh_A
    else:
        A_approx = A_np


    if rank_B is not None:
        U_B, S_B, Vh_B = np.linalg.svd(B_np, full_matrices=False)
        U_B = U_B[:, :rank_B]
        S_B = S_B[:rank_B]
        Vh_B = Vh_B[:rank_B, :]
        B_approx = U_B @ (S_B[:, None] * Vh_B)
    else:
        B_approx = B_np

    result = np.matmul(A_approx, B_approx)
    return torch.from_numpy(result).to(A.device).type(A.dtype)


def logmatmul(A, B, **kwargs):
    """
    Matrix multiplication in the log domain, handling negative values.
    :param A: (N, M) torch tensor
    :param B: (M, K) torch tensor
    :return: (N, K) torch tensor
    """
    eps = 1e-12
    # Decompose into sign and log(abs(x))
    sign_A = torch.sign(A)
    sign_B = torch.sign(B)
    log_A = torch.log(A.abs().clamp(min=eps))
    log_B = torch.log(B.abs().clamp(min=eps))
    
    # Broadcasting
    sum_logs = log_A.unsqueeze(2) + log_B.unsqueeze(0)
    prod_signs = sign_A.unsqueeze(2) * sign_B.unsqueeze(0)
    
    result = torch.sum(prod_signs * torch.exp(sum_logs), dim=1)
    return result

# Extra Credit
def approx_logadd(logx, logy, C=0.6931):
    diff = torch.abs(logx - logy)
    return torch.max(logx, logy) + torch.relu(C - diff)

def logmatmul_approx(A, B, **kwargs):
    eps = 1e-12
    sign_A = torch.sign(A)
    sign_B = torch.sign(B)
    log_A = torch.log(A.abs().clamp(min=eps))
    log_B = torch.log(B.abs().clamp(min=eps))
    N, M = A.shape
    M2, K = B.shape
    assert M == M2

    result = torch.zeros((N, K), dtype=A.dtype, device=A.device)
    for i in range(N):
        for j in range(K):
            s = sign_A[i, 0] * sign_B[0, j]
            log_sum = log_A[i, 0] + log_B[0, j]
            for k in range(1, M):
                s_new = sign_A[i, k] * sign_B[k, j]
                log_term = log_A[i, k] + log_B[k, j]
                if s == s_new:
                    log_sum = approx_logadd(log_sum, log_term)
                else:
                    log_sum = torch.max(log_sum, log_term) 
            result[i, j] = s * torch.exp(log_sum)
    return result
