import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def logit_mse_loss(y_hat, y, eps=1e-4):
    y  = y.clamp(eps, 1-eps)
    y_hat = y_hat.clamp(eps, 1-eps)
    z  = torch.log(y / (1 - y))
    z_hat = torch.log(y_hat / (1 - y_hat))
    return F.smooth_l1_loss(z_hat, z)  # o F.mse_loss

def loss_reconstuction(x, T, U, D, temp):
    emmisivity = 0.95
    """
    Reconstruct forward radiance x from MODTRAN products (T, U, D) using:

      L(λ) = T(λ) * ε * B(λ, Tsurf) + T(λ) * (1 - ε) * D(λ) + U(λ)

    Expected shapes (broadcasting supported):
      - x: (B, K, C)         forward radiance at K surface temperatures
      - T: (B, K, C)         transmittance (0..1)
      - U: (B, K, C)         upwelling radiance
      - D: (B, 1, C) or (B,C)  downwelling radiance (shared across temps)

    Returns:
      scalar reconstruction loss (MSE) between reconstructed radiance and x.

    Notes:
      - Uses a fixed wavelength grid of length C over 8..12 µm. If you have the
        true MODTRAN wavelength grid, replace `wl_um` accordingly.
      - Planck radiance is computed in microflicks per micron (µflick/µm),
        matching the reference implementation you shared.
    """

    # ---- constants (CODATA exact) ----
    _H = 6.62607015e-34
    _C = 299792458.0
    _K = 1.380649e-23

    if x.ndim != 3:
        raise ValueError(f"x must be (B,K,C); got {tuple(x.shape)}")
    Bsz, K, C = x.shape

    # Ensure shapes
    if D.ndim == 2:
        D = D.unsqueeze(1)  # (B,1,C)
    if D.ndim != 3:
        raise ValueError(f"D must be (B,1,C) or (B,C); got {tuple(D.shape)}")
    if D.size(0) != Bsz or D.size(-1) != C:
        raise ValueError(f"D must match batch/channels of x; got {tuple(D.shape)} vs x {tuple(x.shape)}")

    # Wavelength grid (µm) matching channel count
    wl_um = [8.10495, 8.12474, 8.14452, 8.16431, 8.18409, 8.20388, 8.22367, 8.24345,
            8.26324, 8.28302, 8.30282, 8.3226, 8.34239, 8.36219, 8.38199, 8.40178,
            8.42158, 8.44138, 8.46117, 8.48098, 8.50078, 8.52058, 8.54039, 8.56019,
            8.57999, 8.5998, 8.61961, 8.63942, 8.65924, 8.67905, 8.69886, 8.71867,
            8.73849, 8.75831, 8.77813, 8.79795, 8.81777, 8.83759, 8.85741, 8.87724,
            8.89706, 8.91689, 8.93671, 8.95654, 8.97637, 8.9962, 9.01603, 9.03587,
            9.05571, 9.07553, 9.09537, 9.11521, 9.13504, 9.15488, 9.17472, 9.19456,
            9.2144, 9.23424, 9.25408, 9.27393, 9.29377, 9.31361, 9.33346, 9.35331,
            9.37316, 9.393, 9.41286, 9.4327, 9.45255, 9.4724, 9.49225, 9.51211,
            9.53196, 9.55181, 9.57166, 9.59152, 9.61138, 9.63123, 9.65109, 9.67095,
            9.6908, 9.71066, 9.73052, 9.75038, 9.77024, 9.79011, 9.80997, 9.82982,
            9.84969, 9.86954, 9.88941, 9.90927, 9.92914, 9.949, 9.96887, 9.98874,
            10.0086, 10.02846, 10.04833, 10.0682, 10.08806, 10.10793, 10.1278, 10.14766,
            10.16753, 10.1874, 10.20728, 10.22715, 10.24701, 10.26689, 10.28675, 10.30662,
            10.32649, 10.34636, 10.36624, 10.38611, 10.40598, 10.42585, 10.44573, 10.4656,
            10.48548, 10.50535, 10.52522, 10.54509, 10.56497, 10.58484, 10.60472, 10.62459,
            10.64447, 10.66434, 10.68422, 10.7041, 10.72398, 10.74385, 10.76373, 10.78361,
            10.80349, 10.82336, 10.84324, 10.86312, 10.883, 10.90287, 10.92276, 10.94263,
            10.96251, 10.98239, 11.00227, 11.02215, 11.04203, 11.06191, 11.08179, 11.10167,
            11.12155, 11.14144, 11.16132, 11.1812, 11.20108, 11.22097, 11.24085, 11.26073,
            11.28061, 11.3005, 11.32039, 11.34027, 11.36016, 11.38004, 11.39993, 11.41981,
            11.4397, 11.45958, 11.47948, 11.49936, 11.51925, 11.53913, 11.55902, 11.57891,
            11.5988, 11.61869, 11.63857, 11.65847, 11.67835, 11.69825, 11.71814, 11.73802,
            11.75792, 11.77781, 11.79771, 11.8176, 11.8375, 11.85738, 11.87729, 11.89718,
            11.91708, 11.93697, 11.95687, 11.97677, 11.99666, 12.01656, 12.03646, 12.05636,
            12.07626, 12.09616, 12.11607, 12.13597, 12.15588, 12.17578, 12.19568, 12.21558,
            12.23549, 12.2554, 12.27531, 12.29521, 12.31513, 12.33503, 12.35495, 12.37486,
            12.39477, 12.41469, 12.43461, 12.45452, 12.47444, 12.49436, 12.51428, 12.5342,
            12.55412, 12.57404, 12.59396, 12.61388, 12.63379, 12.65372, 12.67364, 12.69356,
            12.71347, 12.73339, 12.75332, 12.77324, 12.79315, 12.81307, 12.83299, 12.85291,
            12.87283, 12.89275, 12.91267, 12.93259, 12.95251, 12.97243, 12.99235, 13.01227,
            13.03219, 13.05211, 13.07203, 13.09194, 13.11186, 13.13179, 13.1517, 13.17162]
    wl_um = torch.tensor(wl_um, device=x.device, dtype=x.dtype)
    if wl_um.numel() != C:
        # Fall back to a linear grid if channel count doesn't match the hardcoded grid.
        wl_um = torch.linspace(wl_um.min(), wl_um.max(), C, device=x.device, dtype=x.dtype)

    # temperature can be:
    # - scalar ()
    # - per-sample (B,)
    # - per-sample with extra dims (B,1,...) -> we only use the first dim as batch
    temp_t = torch.as_tensor(temp, device=x.device, dtype=x.dtype).clamp_min(1e-6)
    if temp_t.ndim == 0:
        temp_t = temp_t.view(1, 1)            # (1,1)
    else:
        temp_t = temp_t.reshape(-1, 1)        # (B,1)

    # ---- Planck spectral radiance (µflick/µm) ----
    # wl_um: (C,), temp_t: (B,1) -> Bplanck: (B,C)
    lam_m = (wl_um * 1e-6).view(1, -1)  # (1,C)
    exponent = (_H * _C) / (lam_m * _K * temp_t)  # (B,C)
    denom = torch.expm1(exponent)
    B_per_m = (2.0 * _H * _C * _C) / (torch.pow(lam_m, 5) * denom)  # (B,C)
    B_micro = B_per_m * 1e-4  # (B,C)

    # Broadcast to (B,1,C) then to (B,K,C)
    B_micro = B_micro.unsqueeze(1)  # (B,1,C)
    if B_micro.size(0) != Bsz:
        # if temp was scalar (B_micro has B=1), broadcast to batch
        if B_micro.size(0) == 1:
            B_micro = B_micro.expand(Bsz, -1, -1)
        else:
            raise ValueError(f"temp must be scalar or length-B; got {tuple(torch.as_tensor(temp).shape)} for batch B={Bsz}")

    eps = torch.as_tensor(emmisivity, device=x.device, dtype=x.dtype)

    # D: (B,1,C) -> (B,K,C)
    Dk = D.expand(-1, K, -1)

    # Reconstruct x_hat: (B,K,C)
    x_hat = (T * (eps * B_micro)) + (T * ((1.0 - eps) * Dk)) + U

    return F.mse_loss(x_hat, x)
