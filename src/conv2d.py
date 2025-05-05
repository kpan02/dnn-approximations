import torch
import torch.nn.functional as F


def conv2d(x, k, b, method='naive'):
    """
    Convolution of single instance and single input and output channel
    :param x:  (H, W) PyTorch Tensor
    :param k:  (Hk, Wk) PyTorch Tensor
    :param b:  (1,) PyTorch tensor or scalar
    :param method: Which method do we use to implement it. Valid choices include
                   'naive', 'torch', 'pytorch', 'im2col', 'winograd', and 'fft'
    :return:
        Output tensor should have shape (H_out, W_out)
    """
    method = method.lower()
    if method == 'naive':
        return naive(x, k, b)
    elif method in ['torch', 'pytorch']:
        return pytorch(x, k, b)
    elif method == 'im2col':
        return im2col(x, k, b)
    elif method == 'winograd':
        return winograd(x, k, b)
    elif method == 'fft':
        return fft(x, k, b)
    else:
        raise ValueError("Invalid [method] value: %s" % method)


def naive(x, k, b):
    """ Sliding window solution. """
    output_shape_0 = x.shape[0] - k.shape[0] + 1
    output_shape_1 = x.shape[1] - k.shape[1] + 1
    result = torch.zeros(output_shape_0, output_shape_1)
    for row in range(output_shape_0):
        for col in range(output_shape_1):
            window = x[row: row + k.shape[0], col: col + k.shape[1]]
            result[row, col] = torch.sum(torch.multiply(window, k))
    return result + b


def pytorch(x, k, b):
    """ PyTorch solution. """
    return F.conv2d(
        x.unsqueeze(0).unsqueeze(0),  # (1, 1, H, W)
        k.unsqueeze(0).unsqueeze(0),  # (1, 1, Hk, Wk)
        b   # (1, )
    ).squeeze(0).squeeze(0)  # (H_out, W_out)


def im2col(x, k, b):
    H, W = x.shape
    Hk, Wk = k.shape
    H_out = H - Hk + 1
    W_out = W - Wk + 1

    cols = []
    for i in range(H_out):
        for j in range(W_out):
            patch = x[i:i+Hk, j:j+Wk].reshape(-1)
            cols.append(patch)
    im2col_matrix = torch.stack(cols, dim=1)

    k_flat = k.reshape(-1)

    out = torch.matmul(k_flat, im2col_matrix)
    out = out.reshape(H_out, W_out)

    return out + b


def winograd(x, k, b):
    H, W = x.shape
    Hk, Wk = k.shape
    assert Hk == 3 and Wk == 3
    H_out = H - 2
    W_out = W - 2

    # Winograd F(2x2, 3x3) transform matrices
    B = torch.tensor([
        [1, 0, -1, 0],
        [0, 1, 1, 0], 
        [0, -1, 1, 0],
        [0, 1, 0, -1]
    ], dtype=x.dtype, device=x.device)

    G = torch.tensor([
        [1, 0, 0],
        [0.5, 0.5, 0.5],
        [0.5, -0.5, 0.5], 
        [0, 0, 1]
    ], dtype=x.dtype, device=x.device)

    A = torch.tensor([
        [1, 1, 1, 0],
        [0, 1, -1, -1]
    ], dtype=x.dtype, device=x.device)

    U = G @ k @ G.t()

    out = torch.zeros(H_out, W_out, dtype=x.dtype, device=x.device)

    for i in range(0, H_out, 2):
        for j in range(0, W_out, 2):
            patch = torch.zeros(4, 4, dtype=x.dtype, device=x.device)
            h_max = min(4, H - i)
            w_max = min(4, W - j)
            patch[:h_max, :w_max] = x[i:i+h_max, j:j+w_max]
            V = B @ patch @ B.t()
            M = U * V
            Y = A @ M @ A.t() 
            h_end = min(i+2, H_out)
            w_end = min(j+2, W_out)
            out[i:h_end, j:w_end] = Y[:h_end-i, :w_end-j]

    return out + b


def fft(x, k, b):
    H, W = x.shape
    Hk, Wk = k.shape
    H_out = H - Hk + 1
    W_out = W - Wk + 1

    k_flipped = torch.flip(k, dims=[0, 1])

    fft_shape = (H + Hk - 1, W + Wk - 1)
    x_padded = torch.zeros(fft_shape, dtype=x.dtype, device=x.device)
    k_padded = torch.zeros(fft_shape, dtype=k.dtype, device=k.device)
    x_padded[:H, :W] = x
    k_padded[:Hk, :Wk] = k_flipped

    X_fft = torch.fft.fft2(x_padded)
    K_fft = torch.fft.fft2(k_padded)

    Y_fft = X_fft * K_fft

    y_full = torch.fft.ifft2(Y_fft).real
    y = y_full[Hk-1:Hk-1+H_out, Wk-1:Wk-1+W_out]

    return y + b
