from importlib.metadata import version

import pytest
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import ball_query, knn_points, sample_farthest_points
from pytorch3d.renderer.mesh.rasterize_meshes import rasterize_meshes
from pytorch3d.structures import Meshes
import torch


@pytest.fixture(scope="module")
def device() -> torch.device:
    assert torch.cuda.is_available(), "The tests must run on a CUDA GPU"
    return torch.device("cuda")


def test_published_cuda_wheel(device: torch.device) -> None:
    assert version("pytorch3d") == "0.7.9+cu.12.8.torch.2.10"
    assert torch.__version__ == "2.10.0+cu128"
    assert torch.version.cuda == "12.8"
    assert torch.cuda.get_device_name(device)


@pytest.mark.parametrize("norm", [1, 2])
def test_nearest_neighbors(device: torch.device, norm: int) -> None:
    source = torch.tensor(
        [[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0]]],
        device=device,
    )
    target = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.75, 0.0, 0.0], [3.5, 0.0, 0.0], [0.0, 2.5, 0.0]]],
        device=device,
    )

    actual = knn_points(source, target, K=2, norm=norm)
    distances = torch.cdist(source, target, p=norm)
    if norm == 2:
        distances = distances.square()
    expected_distances, expected_indices = distances.topk(2, largest=False)

    torch.testing.assert_close(actual.dists, expected_distances)
    torch.testing.assert_close(actual.idx, expected_indices)


def test_radius_query(device: torch.device) -> None:
    source = torch.tensor(
        [[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]],
        device=device,
    )
    target = torch.tensor(
        [[[0.1, 0.0, 0.0], [0.4, 0.0, 0.0], [3.1, 0.0, 0.0]]],
        device=device,
    )

    actual = ball_query(source, target, K=2, radius=0.5)

    torch.testing.assert_close(
        actual.idx,
        torch.tensor([[[0, 1], [2, -1]]], device=device),
    )
    torch.testing.assert_close(
        actual.dists,
        torch.tensor([[[0.01, 0.16], [0.01, 0.0]]], device=device),
    )


def test_farthest_point_sampling(device: torch.device) -> None:
    points = torch.tensor(
        [
            [
                [-1.0, -1.0],
                [-1.3, 1.1],
                [0.2, -1.1],
                [0.0, 0.0],
                [1.3, 1.3],
                [1.0, 0.5],
                [-1.3, 0.2],
                [1.5, -0.5],
            ],
            [
                [-2.2, -2.4],
                [-2.1, 2.0],
                [2.2, 2.1],
                [2.1, -2.4],
                [0.4, -1.0],
                [0.3, 0.3],
                [1.2, 0.5],
                [4.5, 4.5],
            ],
        ],
        device=device,
    )

    actual, indices = sample_farthest_points(points, K=2)

    expected_indices = torch.tensor([[0, 4], [0, 7]], device=device)
    expected_points = points.gather(
        dim=1,
        index=expected_indices[..., None].expand(-1, -1, points.shape[-1]),
    )
    torch.testing.assert_close(indices, expected_indices)
    torch.testing.assert_close(actual, expected_points)


@pytest.mark.parametrize("translation", [0.0, 1.0])
def test_chamfer_distance(device: torch.device, translation: float) -> None:
    source = torch.tensor(
        [[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]],
        device=device,
    )
    target = source + torch.tensor([translation, 0.0, 0.0], device=device)

    actual, _ = chamfer_distance(source, target)

    torch.testing.assert_close(
        actual,
        torch.tensor(2.0 * translation**2, device=device),
    )


def test_chamfer_distance_backward(device: torch.device) -> None:
    torch.manual_seed(0)
    source = torch.randn((2, 8, 3), device=device, requires_grad=True)
    target = torch.randn((2, 6, 3), device=device, requires_grad=True)
    expected_source = source.detach().clone().requires_grad_()
    expected_target = target.detach().clone().requires_grad_()

    actual, _ = chamfer_distance(source, target)
    distances = torch.cdist(expected_source, expected_target).square()
    expected = distances.min(dim=2).values.mean() + distances.min(dim=1).values.mean()
    actual.backward()
    expected.backward()

    torch.testing.assert_close(actual, expected)
    assert source.grad is not None
    assert target.grad is not None
    assert expected_source.grad is not None
    assert expected_target.grad is not None
    torch.testing.assert_close(source.grad, expected_source.grad)
    torch.testing.assert_close(target.grad, expected_target.grad)


def make_triangle(device: torch.device, *, requires_grad: bool = False) -> Meshes:
    vertices = torch.tensor(
        [[[-0.7, -0.7, 1.0], [0.7, -0.7, 1.0], [0.0, 0.7, 1.0]]],
        device=device,
        requires_grad=requires_grad,
    )
    faces = torch.tensor([[[0, 1, 2]]], device=device)
    return Meshes(verts=vertices, faces=faces)


def test_mesh_rasterization(device: torch.device) -> None:
    face_indices, depth, barycentric, _ = rasterize_meshes(
        make_triangle(device),
        image_size=16,
        faces_per_pixel=1,
        bin_size=0,
    )
    covered_pixels = face_indices >= 0

    assert covered_pixels.any()
    torch.testing.assert_close(
        face_indices[covered_pixels],
        torch.zeros_like(face_indices[covered_pixels]),
    )
    torch.testing.assert_close(
        depth[covered_pixels],
        torch.ones_like(depth[covered_pixels]),
    )
    torch.testing.assert_close(
        barycentric[covered_pixels].sum(dim=-1),
        torch.ones_like(depth[covered_pixels]),
    )


def test_mesh_rasterization_backward(device: torch.device) -> None:
    mesh = make_triangle(device, requires_grad=True)
    face_indices, depth, _, _ = rasterize_meshes(
        mesh,
        image_size=16,
        faces_per_pixel=1,
        bin_size=0,
    )

    depth[face_indices >= 0].sum().backward()

    gradient = mesh.verts_padded().grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0
