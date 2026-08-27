from typing import List, Optional

import numpy as np

from openeo.processes import array_contains, if_, power
from openeo.rest.datacube import DataCube


def to_scl_dilation_mask(
    data: DataCube,
    erosion_kernel_size: int = 0,
    mask1_values: Optional[List[int]] = [2, 4, 5, 6, 7],
    mask2_values: Optional[List[int]] = [3, 8, 9, 10, 11],
    kernel1_size: int = 17,
    kernel2_size: int = 201,
) -> DataCube:
    """
    Create a cloud/shadow mask by dilating Sentinel-2 SCL classification values.

    Mirrors ``SCLConvolutionFilter.createMask`` from the GeoTrellis backend,
    using only standard openEO processes (``apply``, ``apply_kernel``,
    ``merge_cubes``).

    :param data: Single-band DataCube containing only the Sentinel-2 SCL band.
    :param erosion_kernel_size:
        Diameter (pixels) of the circular erosion kernel applied before each
        dilation step.  0 (default) disables erosion.
    :param mask1_values:
        SCL class values considered *valid* pixels for the first mask pass.
        Defaults to ``[2, 4, 5, 6, 7]``
        (dark-area/shadows, vegetation, bare soil, water, unclassified).
        Pixels **not** in this list are dilated.
    :param mask2_values:
        SCL class values considered *cloud/shadow* pixels for the second pass.
        Defaults to ``[3, 8, 9, 10, 11]``
        (cloud shadow, medium/high-probability cloud, thin cirrus, snow).
        Pixels **in** this list are dilated.
    :param kernel1_size:
        Side length (pixels) of the Gaussian dilation kernel for mask 1.
        Corresponds to ``kernel1_size`` in the Scala implementation.
    :param kernel2_size:
        Side length (pixels) of the Gaussian dilation kernel for mask 2.
        Corresponds to ``kernel2_size`` in the Scala implementation.
    :return:
        Binary mask DataCube with the same dimensions as the input:
        ``1`` = pixel is masked (cloud / shadow / invalid neighbour),
        ``0`` = pixel is valid.
    """

    # ------------------------------------------------------------------
    # Kernel helpers
    # ------------------------------------------------------------------

    def _gaussian_kernel(size: int) -> np.ndarray:
        """
        Normalized Gaussian kernel.
        """
        sd = size / 6.0
        center = size // 2
        row, col = np.ogrid[:size, :size]
        g = np.exp(-((col - center) ** 2 + (row - center) ** 2) / (2 * sd ** 2))
        g /= g.sum()
        return g

    def _circle_kernel(size: int) -> np.ndarray:
        """
        Binary circular (disk) kernel.
        """
        radius = size // 2  # integer division, matching Scala's Int arithmetic
        center = size // 2
        row, col = np.ogrid[:size, :size]
        dist2 = (col - center) ** 2 + (row - center) ** 2
        return (dist2 <= radius * radius).astype(float)

    # ------------------------------------------------------------------
    # Morphological erosion
    # ------------------------------------------------------------------

    def _erode(cube: DataCube) -> DataCube:
        """
        Morphologically erode a binary (0/1) cube.
        """
        if erosion_kernel_size <= 0:
            return cube

        ek = _circle_kernel(erosion_kernel_size)

        # Invert binary mask:  0 → 1,  1 → 0   via (x − 1)²
        inverted = cube.apply(lambda x: power(x - 1, 2))

        # Convolve with the circular erosion kernel
        convolved = inverted.apply_kernel(ek)

        # Threshold at 0.5 and re-invert:
        #   a pixel whose inverted-mask neighbourhood sums > 0.5 lies next to a
        #   valid (0) pixel, so it gets eroded away (set to 0 in the original mask).
        return convolved.apply(lambda x: x <= 0.5)

    # ------------------------------------------------------------------
    # Mask 1 – dilate pixels NOT in mask1_values
    # ------------------------------------------------------------------
    # 0 = valid (in mask1_values),  1 = to-be-masked (everything else)
    binary1 = data.apply(lambda x: if_(array_contains(mask1_values, x), 0, 1))
    eroded1 = _erode(binary1)

    if kernel1_size > 0:
        # Gaussian dilation then threshold
        dilated1 = eroded1.apply_kernel(_gaussian_kernel(kernel1_size))
        mask1 = dilated1.apply(lambda x: x > 0.057)
    else:
        mask1 = eroded1

    # ------------------------------------------------------------------
    # Mask 2 – dilate pixels IN mask2_values
    # ------------------------------------------------------------------
    # 1 = cloud/shadow (in mask2_values),  0 = other
    binary2 = data.apply(lambda x: if_(array_contains(mask2_values, x), 1, 0))
    eroded2 = _erode(binary2)

    if kernel2_size > 0:
        # Gaussian dilation then threshold
        dilated2 = eroded2.apply_kernel(_gaussian_kernel(kernel2_size))
        mask2 = dilated2.apply(lambda x: x > 0.025)
    else:
        mask2 = eroded2

    # ------------------------------------------------------------------
    # Combine: logical OR of the two binary masks
    # ------------------------------------------------------------------
    return mask1.merge_cubes(
        mask2,
        overlap_resolver="or"
    )