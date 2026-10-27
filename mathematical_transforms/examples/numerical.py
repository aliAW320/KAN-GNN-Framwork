import torch

from kan_transforms import laplace_transform, z_transform


def main():
    signal = torch.tensor([1., 2., 3.], dtype=torch.float64)
    z = torch.tensor([1., 2., 1j], dtype=torch.complex128)
    print("Z-transform:", z_transform(signal, z))
    times = torch.linspace(0, 10, 10001, dtype=torch.float64)
    samples = torch.exp(-2 * times)
    s = torch.tensor([0., 1., 1.+1j], dtype=torch.complex128)
    print("Sampled Laplace:", laplace_transform(samples, times, s))
    print("Infinite-domain analytical reference:", 1 / (2 + s))


if __name__ == "__main__":
    main()
