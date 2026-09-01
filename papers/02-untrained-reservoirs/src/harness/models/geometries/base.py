"""The geometry interface every lattice venue implements.

A geometry is a *seating chart*, not a different model: all nine reuse the same
[C, G, G] kernel and state storage, so the parameter budget is matched by
construction and a geometry comparison is a genuine single-factor experiment.
What changes is only how the lattice's edges are glued, which shows up in
exactly four places:

    embed_kernel     open axes need the kernel at signed offsets in a padded
                     buffer, so a linear convolution replaces a circular one
    kernel_spectrum  the transform the venue's topology calls for (1-D ring,
                     2-D grid, 3-D prism, or a bipartite pair of 3-D blocks)
    apply            how a field is prepared, convolved, and cropped back
    circulant_index  the same operator as a dense gather, for the matmul path

Both paths are the same operator; `tests/test_geometries.py` asserts they agree
to floating-point rounding on every venue, which is what lets the runner pick
whichever is faster on the device it finds.

Geometries are plain objects rather than modules: they hold no parameters, and
the constant tensors they need (sphere weights, gather indices) are cached per
device so the hot loop never re-materializes them.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch

TWO_PI = 2 * math.pi


class Geometry(ABC):
    """One lattice venue. Stateless apart from device-local constant caches."""

    #: the `--boundary` value that selects this venue
    name: str
    #: the declared frequency axis, for the tonotopy record in the docs
    frequency_axis: str = "grid rows"
    #: kernel-support masking is defined on the 2-D row/column offset grid;
    #: venues that index offsets differently opt out rather than mask wrongly
    supports_kernel_support: bool = True

    def __init__(self, grid: int):
        self.grid = grid
        self._cache: dict = {}
        self.validate()

    def validate(self) -> None:  # noqa: B027 - optional hook, not a contract
        """Raise if this venue cannot be built at `self.grid`.

        Most venues work at any grid; the ones with derived cell dimensions
        (cube, diamond) override this to reject grids their layout cannot
        express, so the failure lands at construction rather than mid-sweep.
        """

    @property
    def clamp_factor(self) -> float:
        """Divisor applied to the spectral clamp so the cap stays a TRUE
        operator-norm bound. 1.0 wherever max |K-hat| already bounds the
        operator; the twisted venues override it because their mirrored
        double-cover extension carries norm sqrt(2)."""
        return 1.0

    # --- kernel side ------------------------------------------------------

    def embed_kernel(self, kernel: torch.Tensor) -> torch.Tensor:
        """[C, G, G] -> the buffer the transform runs on. Identity for venues
        whose axes are all periodic."""
        return kernel

    @abstractmethod
    def kernel_spectrum(self, kernel: torch.Tensor) -> torch.Tensor:
        """Embedded kernel -> the complex operator consumed by `apply`."""

    @abstractmethod
    def spectrum_to_taps(self, kfft: torch.Tensor, embedded_shape: tuple) -> torch.Tensor:
        """The clamped spectrum back to flat spatial taps, for the dense path."""

    # --- operator side ----------------------------------------------------

    @abstractmethod
    def circulant_index(self) -> torch.Tensor:
        """[N, N] (or [2, N, N] for a double cover) gather into the flat taps."""

    def dense_operator(self, taps: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        """Flat taps + gather index -> the dense [C, N, N] operator."""
        return taps[:, index]

    @abstractmethod
    def apply(self, field: torch.Tensor, coup: torch.Tensor) -> torch.Tensor:
        """Convolve [B, F, C, G, G] with the spectral operator, same shape out."""

    # --- helpers ----------------------------------------------------------

    def _constant(self, key: str, build, like: torch.Tensor) -> torch.Tensor:
        """Device- and dtype-local cache for a derived constant tensor."""
        ck = (key, like.device, like.dtype)
        if ck not in self._cache:
            self._cache[ck] = build().to(device=like.device, dtype=like.dtype)
        return self._cache[ck]

    def _signed_row_embedding(self, kernel: torch.Tensor) -> torch.Tensor:
        """Kernel rows at their SIGNED offsets in a 2G-row buffer: offsets
        0..G/2-1 stay put, negative offsets (rows G/2..G-1, i.e. r-G) land at
        2G+(r-G) = G+r, and the middle G rows stay zero. Zero-padding is what
        turns the row axis's circular convolution into a linear one, so energy
        at the top band cannot couple around to the bottom band."""
        g = self.grid
        out = kernel.new_zeros(kernel.shape[0], 2 * g, g)
        out[:, : g // 2] = kernel[:, : g // 2]
        out[:, g + g // 2 :] = kernel[:, g // 2 :]
        return out

    def _signed_both_embedding(self, kernel: torch.Tensor) -> torch.Tensor:
        """The same signed-offset embedding on BOTH axes: a [2G, 2G] buffer
        whose four corner blocks hold the kernel's offset quadrants."""
        g, h = self.grid, self.grid // 2
        out = kernel.new_zeros(kernel.shape[0], 2 * g, 2 * g)
        out[:, :h, :h] = kernel[:, :h, :h]
        out[:, :h, g + h :] = kernel[:, :h, h:]
        out[:, g + h :, :h] = kernel[:, h:, :h]
        out[:, g + h :, g + h :] = kernel[:, h:, h:]
        return out

    def _offset_index(self, rows_mod: int, cols_mod: int) -> torch.Tensor:
        """circ[i, j] = tap at the (row, column) offset between sites i and j."""
        g = self.grid
        pos = torch.arange(g * g)
        yi, xi = pos.div(g, rounding_mode="floor"), pos % g
        dy = (yi[:, None] - yi[None, :]) % rows_mod
        dx = (xi[:, None] - xi[None, :]) % cols_mod
        return dy * cols_mod + dx  # stride = the (padded) kernel's column count

    def __repr__(self) -> str:
        return f"{type(self).__name__}(grid={self.grid})"


class PlanarGeometry(Geometry):
    """Shared machinery for the venues that stay a 2-D row/column grid:
    torus, cylinder, sheet, sphere, and the twisted pair. They differ only in
    which axes are padded open and how the field is prepared before the
    transform, so the kernel side is identical for all of them."""

    #: (rows, cols) modulus of the embedded kernel buffer, as multiples of G
    embed_shape: tuple[int, int] = (1, 1)

    def kernel_spectrum(self, kernel: torch.Tensor) -> torch.Tensor:
        return torch.fft.rfft2(kernel)

    def spectrum_to_taps(self, kfft: torch.Tensor, embedded_shape: tuple) -> torch.Tensor:
        return torch.fft.irfft2(kfft, s=embedded_shape[-2:]).flatten(1)

    def circulant_index(self) -> torch.Tensor:
        rows, cols = self.embed_shape
        return self._offset_index(rows * self.grid, cols * self.grid)
