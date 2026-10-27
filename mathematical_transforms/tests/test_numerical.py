import pytest
import torch

from kan_transforms import laplace_transform, z_transform


def test_z_exact_polynomial_complex_and_scalar():
    x = torch.tensor([1., 2., 3.], dtype=torch.float64)
    z = torch.tensor([1., 2., 1j, -2.], dtype=torch.complex128)
    torch.testing.assert_close(z_transform(x, z), 1 + 2/z + 3/z.square())
    assert z_transform(x, torch.tensor(2.)).shape == ()
    torch.testing.assert_close(z_transform(x, z, start_index=-1), z + 2 + 3/z)
    torch.testing.assert_close(z_transform(x, z, start_index=2),
                               z_transform(x, z) / z.square())


def test_z_dft_identity_and_batch_dimensions():
    x = torch.arange(12).double().reshape(3, 4)
    angles = 2 * torch.pi * torch.arange(4).double() / 4
    z = torch.exp(1j * angles)
    torch.testing.assert_close(z_transform(x, z), torch.fft.fft(x))
    torch.testing.assert_close(z_transform(x.T, z, dim=0), z_transform(x, z))
    assert z_transform(x, z[0]).shape == (3,)


def test_laplace_constant_nonuniform_and_complex_analytic():
    t = torch.tensor([0., 0.1, 0.8, 2.], dtype=torch.float64)
    x = torch.ones_like(t)
    torch.testing.assert_close(laplace_transform(x, t, torch.tensor(0.)), torch.tensor(2.).double())
    t = torch.linspace(0, 8, 16001, dtype=torch.float64)
    s = torch.tensor([0., 0.5, 1.+2j], dtype=torch.complex128)
    x = torch.exp(-2 * t)
    exact_finite_integral = (1 - torch.exp(-(s + 2) * t[-1])) / (s + 2)
    torch.testing.assert_close(laplace_transform(x, t, s), exact_finite_integral,
                               atol=1e-7, rtol=1e-6)
    batched = torch.stack([x, 2*x])
    torch.testing.assert_close(laplace_transform(batched.T, t, s, dim=0),
                               torch.stack([laplace_transform(x, t, s),
                                            2*laplace_transform(x, t, s)]))


@pytest.mark.parametrize("name", ["z", "laplace"])
def test_linearity_gradcheck_and_complex_signal(name):
    x = torch.tensor([0.5, 0.7, 1.2], dtype=torch.double, requires_grad=True)
    points = torch.tensor([1.3, 1.8], dtype=torch.double, requires_grad=True)
    if name == "z":
        operation = z_transform
    else:
        t = torch.tensor([0.1, 0.3, 1.], dtype=torch.double, requires_grad=True)
        assert torch.autograd.gradcheck(laplace_transform, (x, t, points))
        def operation(a, q):
            return laplace_transform(a, t, q)
    assert torch.autograd.gradcheck(operation, (x, points))
    torch.testing.assert_close(operation((1+2j)*x, points), (1+2j)*operation(x, points))
    assert torch.autograd.gradcheck(operation, (x.to(torch.complex128),
                                               points.to(torch.complex128)))


@pytest.mark.parametrize("bad_z", [torch.tensor(0.), torch.tensor([float('nan')]),
                                    torch.ones(2, 2), torch.empty(0), torch.tensor([1])])
def test_bad_z(bad_z):
    with pytest.raises(ValueError):
        z_transform(torch.ones(3), bad_z)


def test_invalid_inputs():
    with pytest.raises(TypeError):
        z_transform([1., 2.], torch.tensor(1.))
    for signal, dim in [(torch.tensor(1.), -1), (torch.ones(2), 3), (torch.empty(0), -1)]:
        with pytest.raises(ValueError):
            z_transform(signal, torch.tensor(1.), dim=dim)
    with pytest.raises(ValueError):
        z_transform(torch.ones(2), torch.tensor(1.), start_index=0.5)
    for t in [torch.tensor([0., -1.]), torch.tensor([0., 0.]), torch.tensor([1., 0.]),
              torch.ones(3), torch.ones(2, dtype=torch.complex64)]:
        with pytest.raises(ValueError):
            laplace_transform(torch.ones(2), t, torch.tensor(1.))
    with pytest.raises(ValueError):
        laplace_transform(torch.ones(1), torch.zeros(1), torch.tensor(1.))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
def test_cuda_numerical_parity_and_backward():
    x = torch.tensor([1., 2., 3.], dtype=torch.float64)
    t = torch.tensor([0., 0.3, 1.], dtype=torch.float64)
    q = torch.tensor([1.+1j, 2.], dtype=torch.complex128)
    gpu = x.cuda().requires_grad_()
    torch.testing.assert_close(z_transform(gpu, q.cuda()).cpu(), z_transform(x, q))
    result = laplace_transform(gpu, t.cuda(), q.cuda())
    torch.testing.assert_close(result.cpu(), laplace_transform(x, t, q))
    result.abs().square().sum().backward()
    assert torch.isfinite(gpu.grad).all()
