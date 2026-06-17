# build-pytorch3d

Pre-built Linux wheels for [PyTorch3D](https://github.com/facebookresearch/pytorch3d), a library of
reusable components for deep learning with 3D data, across Python, PyTorch, CUDA, and CPU
architectures.

## Installation

Following the PyTorch convention, artifacts are published to a separate index for each CUDA
version. Each wheel has a local version suffix that identifies the CUDA, PyTorch, and C++ ABI it
was built against, such as `pytorch3d==0.7.9+cu12.8torch2.10.0cxx11abiTRUE`, and requires the
matching PyTorch minor release.

Pre-built wheels are available on [Astral's GPU indexes](https://wheels.astralhosted.com/index.html).
For example, to install a CUDA 12.8 build:

```console
$ uv add pytorch3d --index astral-cu128=https://wheels.astralhosted.com/simple/cu128/
```

This configures the index and uses it as the source for `pytorch3d`:

```toml
[tool.uv.sources]
pytorch3d = { index = "astral-cu128" }

[[tool.uv.index]]
name = "astral-cu128"
url = "https://wheels.astralhosted.com/simple/cu128/"
```

Or, with `uv pip`:

```console
$ uv pip install --index https://wheels.astralhosted.com/simple/cu128/ pytorch3d
```

## Supported versions

Wheels are available for the following `pytorch3d` versions:

- [`0.7.9`](https://github.com/astral-sh-build/build-pytorch3d/releases/tag/v0.7.9-r1)
- [`0.7.8`](https://github.com/astral-sh-build/build-pytorch3d/releases/tag/0.7.8)

The latest release, PyTorch3D 0.7.9, supports the following combinations:

| PyTorch | Python    | `x86_64` CUDA    | `aarch64` CUDA   |
| ------- | --------- | ---------------- | ---------------- |
| 2.4.1   | 3.9–3.12  | 12.1, 12.4       | —                |
| 2.5.1   | 3.9–3.12  | 12.1, 12.4       | —                |
| 2.6.0   | 3.9–3.12  | 12.4, 12.6       | —                |
| 2.7.1   | 3.9–3.13  | 12.6, 12.8       | 12.8             |
| 2.8.0   | 3.9–3.13  | 12.6, 12.8, 12.9 | 12.9             |
| 2.9.0   | 3.10–3.14 | 12.6, 12.8, 12.9 | 12.6, 12.8, 12.9 |
| 2.10.0  | 3.10–3.14 | 12.6, 12.8, 12.9 | 12.6, 12.8, 12.9 |

## License

build-pytorch3d is licensed under the [Apache License, Version 2.0](LICENSE).

<div align="center">
  <a target="_blank" href="https://astral.sh" style="background:none">
    <img src="https://raw.githubusercontent.com/astral-sh/ruff/main/assets/svg/Astral.svg" alt="Made by Astral">
  </a>
</div>
