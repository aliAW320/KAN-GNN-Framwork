"""Finite-sample transforms; all input tensors must share a device.

Scalar queries return the signal's batch shape. A vector of Q queries appends Q
as the final output dimension, independently of the original sample dimension.
Float32/64 and complex64/128 are supported, with normal PyTorch dtype promotion.
"""
import torch


def _tensor(value, name):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dtype not in {torch.float32, torch.float64, torch.complex64, torch.complex128}:
        raise ValueError(f"{name} requires float32/64 or complex64/128")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain finite values")


def _prepare(signal, query, dim):
    _tensor(signal, "signal")
    _tensor(query, "query")
    if signal.ndim == 0 or not -signal.ndim <= dim < signal.ndim:
        raise ValueError("signal needs a valid sample dimension")
    if query.ndim > 1 or query.numel() == 0:
        raise ValueError("query must be a scalar or nonempty vector")
    if signal.device != query.device:
        raise ValueError("signal and query must share a device")
    signal = signal.movedim(dim, -1)
    if signal.size(-1) == 0:
        raise ValueError("signal must contain samples")
    return signal, query.reshape(-1)


def z_transform(signal, z, *, start_index=0, dim=-1):
    """Finite Z sum: X(z)=sum_{k=0}^{N-1} signal[k]*z**(-(k+start_index)).

    start_index=0 is a truncated unilateral transform. Negative start_index
    represents a finite two-sided indexed sequence. z=0 is rejected. This does
    not infer convergence or represent an unspecified infinite sequence.
    """
    samples, points = _prepare(signal, z, dim)
    if isinstance(start_index, bool) or not isinstance(start_index, int):
        raise ValueError("start_index must be an integer")
    if (points == 0).any():
        raise ValueError("Z-transform queries must be nonzero")
    indices = torch.arange(samples.size(-1), device=samples.device) + start_index
    kernel = points.unsqueeze(-1).pow(-indices)
    result = (samples.unsqueeze(-2) * kernel).sum(-1)
    return result.squeeze(-1) if z.ndim == 0 else result


def laplace_transform(signal, times, s, *, dim=-1):
    """Trapezoidal approximation of integral signal(t)*exp(-s*t) dt.

    Integrates only over [times[0], times[-1]], not to infinity. Real, finite,
    nonnegative, strictly increasing (possibly nonuniform) times are required.
    Missing tails and discretization error remain the caller's responsibility.
    """
    samples, points = _prepare(signal, s, dim)
    _tensor(times, "times")
    if times.is_complex() or times.ndim != 1 or times.numel() != samples.size(-1):
        raise ValueError("times must be a real vector matching the sample dimension")
    if times.device != samples.device:
        raise ValueError("times and signal must share a device")
    if times.numel() < 2 or (times < 0).any() or not (times[1:] > times[:-1]).all():
        raise ValueError("Require >=2 nonnegative, strictly increasing times")
    kernel = torch.exp(-points.unsqueeze(-1) * times)
    values = samples.unsqueeze(-2) * kernel
    result = torch.trapezoid(values, x=times, dim=-1)
    return result.squeeze(-1) if s.ndim == 0 else result
