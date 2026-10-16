"""Roy et al. (2016/2017) c-factor BRDF normalization for Sentinel-2.

Pure numpy math plus an openEO UDF entrypoint (`apply_datacube`) in one file, so it can
be (a) tested locally, (b) sent to the backend with `openeo.UDF.from_file`, and
(c) used as the reference for the ONNX version built in `brdf-predict-onnx.ipynb`.

Reference: sen2like S2L_Nbar.py
  https://github.com/senbox-org/sen2like/blob/master/sen2like/sen2like/s2l_processes/S2L_Nbar.py
"""

import numpy as np
import xarray  # used by the UDF entrypoint below; math helpers are numpy-only

# ---------------------------------------------------------------------------
# Roy et al. (2017) Table 1 / Roy et al. (2016) Table 5 / Sen2Like ATBD Table 6.
# MODIS-derived fixed spectral BRDF model parameters [f_iso, f_geo, f_vol].
# Only these 6 Sentinel-2 bands have MODIS analogues; everything else passes through.
# ---------------------------------------------------------------------------
ROY_COEF = {
    "B02": (0.0774, 0.0079, 0.0372),
    "B03": (0.1306, 0.0178, 0.0580),
    "B04": (0.1690, 0.0227, 0.0574),
    "B8A": (0.3093, 0.0330, 0.1535),
    "B11": (0.3430, 0.0453, 0.1154),
    "B12": (0.2658, 0.0387, 0.0639),
}
# Roy/ATBD name B08 == Sentinel-2 band B8A (the 20 m narrow NIR has the MODIS analogue).
ROY_COEF["B08"] = ROY_COEF["B8A"]

D2R = np.pi / 180.0
CLIP_FACTOR = 0.2  # sen2like limits the NBAR correction to +/-20%


def mean_sun_zenith(scene_center_lat_deg):
    """Sen2like `get_mean_sun_angle` (eq. 4): mean SZA [deg] from scene-center latitude."""
    lat = np.asarray(scene_center_lat_deg, dtype=np.float64)
    return (
        6.15e-11 * lat**6
        - 1.95e-09 * lat**5
        - 9.48e-07 * lat**4
        + 2.40e-05 * lat**3
        + 0.01187 * lat**2
        - 0.1272 * lat
        + 31.0076
    )


def kgeo_li_sparse(sza_deg, vza_deg, dphi_deg):
    """Li-Sparse geometric kernel Kgeo (Wanner et al. 1995; h/b = 2, b/r = 1)."""
    theta_s = np.asarray(sza_deg, dtype=np.float64) * D2R
    theta_v = np.asarray(vza_deg, dtype=np.float64) * D2R
    phi = np.asarray(dphi_deg, dtype=np.float64) * D2R

    h_sur_b, b_sur_r = 2.0, 1.0

    theta_s_p = np.arctan(b_sur_r * np.tan(theta_s))
    theta_v_p = np.arctan(b_sur_r * np.tan(theta_v))
    cos_zeta_p = np.cos(theta_s_p) * np.cos(theta_v_p) + np.sin(theta_s_p) * np.sin(
        theta_v_p
    ) * np.cos(phi)

    d = np.sqrt(
        np.tan(theta_s_p) ** 2
        + np.tan(theta_v_p) ** 2
        - 2.0 * np.tan(theta_s_p) * np.tan(theta_v_p) * np.cos(phi)
    )
    sec_s = 1.0 / np.cos(theta_s_p)
    sec_v = 1.0 / np.cos(theta_v_p)

    numerator = np.sqrt(d**2 + (np.tan(theta_s_p) * np.tan(theta_v_p) * np.sin(phi)) ** 2)
    denominator = sec_s + sec_v

    cos_t = h_sur_b * (numerator / denominator)
    # Nastiness: acos domain. sen2like clamps, so do we (and before sqrt(1-cos^2)).
    cos_t = np.clip(cos_t, -1.0, 1.0)

    sin_t = np.sqrt(1.0 - cos_t * cos_t)
    t = np.arccos(cos_t)

    overlap = (1.0 / np.pi) * (t - sin_t * cos_t) * (sec_s + sec_v)
    k_geo = overlap - sec_s - sec_v + 0.5 * (1.0 + cos_zeta_p) * sec_s * sec_v
    return np.nan_to_num(k_geo)


def kvol_ross_thick(sza_deg, vza_deg, dphi_deg):
    """Ross-Thick volumetric kernel, as implemented in sen2like.

    Uses Roujean's scaling 4/(3*pi)*(...) - 1/3, i.e. the MODIS form (Lucht et al. 2000,
    eq. 38: (...) - pi/4) times 4/(3*pi). The Roy coefficients come from MODIS, so this
    gives a ~1-2% weaker correction than the published method; kept to match sen2like.
    """
    theta_s = np.asarray(sza_deg, dtype=np.float64) * D2R
    theta_v = np.asarray(vza_deg, dtype=np.float64) * D2R
    phi = np.asarray(dphi_deg, dtype=np.float64) * D2R

    cos_zeta = np.cos(theta_s) * np.cos(theta_v) + np.sin(theta_s) * np.sin(theta_v) * np.cos(phi)
    cos_zeta = np.clip(cos_zeta, -1.0, 1.0)
    zeta = np.arccos(cos_zeta)

    numerator = (np.pi / 2.0 - zeta) * np.cos(zeta) + np.sin(zeta)
    denominator = np.cos(theta_v) + np.cos(theta_s)
    k_vol = (4.0 / (3.0 * np.pi)) * (numerator / denominator) - (1.0 / 3.0)
    return np.nan_to_num(k_vol)


def c_factor(sza_deg, vza_deg, dphi_deg, coef, sza_norm=None, vza_norm=0.0, dphi_norm=0.0):
    """BRDF c-factor for one band.

    `coef` is (f_iso, f_geo, f_vol). Normalization geometry defaults to
    (sza_norm, 0, 0) where sza_norm is a scalar per scene. If sza_norm is None the
    input SZA is used (i.e. no normalization) -- c == 1.
    """
    if sza_norm is None:
        sza_norm = sza_deg
    kgeo_in = kgeo_li_sparse(sza_deg, vza_deg, dphi_deg)
    kvol_in = kvol_ross_thick(sza_deg, vza_deg, dphi_deg)
    kgeo_norm = kgeo_li_sparse(sza_norm, vza_norm, dphi_norm)
    kvol_norm = kvol_ross_thick(sza_norm, vza_norm, dphi_norm)
    f_iso, f_geo, f_vol = coef
    numerator = f_iso + f_geo * kgeo_norm + f_vol * kvol_norm
    denominator = f_iso + f_geo * kgeo_in + f_vol * kvol_in
    return np.nan_to_num(numerator / denominator)


def nbar(rho, sza_deg, vza_deg, dphi_deg, coef, sza_norm=None):
    """Nadir BRDF-Adjusted Reflectance for one band array.

    Multiplicative c-factor, clamped to +/-20% (sen2like behaviour) and
    forced to 0 where input reflectance is non-positive (nodata guard).
    """
    rho = np.asarray(rho)
    c = c_factor(sza_deg, vza_deg, dphi_deg, coef, sza_norm=sza_norm)
    out = c * rho
    out = np.clip(out, (1.0 - CLIP_FACTOR) * rho, (1.0 + CLIP_FACTOR) * rho)
    out = np.where(rho <= 0, 0.0, out)
    return out


def apply_nbar_bands(bands, angles, scene_center_lat):
    """Apply the correction across bands.

    `bands`: {name: 2D array}; `angles`: dict with keys sza, vza, saa, vaa.
    Bands without Roy coefficients come back untouched.
    """
    sza, vza = angles["sza"], angles["vza"]
    dphi = angles["saa"] - angles["vaa"]
    theta_s = mean_sun_zenith(scene_center_lat)
    out = {}
    for name, rho in bands.items():
        coef = ROY_COEF.get(name)
        out[name] = nbar(rho, sza, vza, dphi, coef, sza_norm=theta_s) if coef else rho
    return out


# ---------------------------------------------------------------------------
# openEO Python UDF entrypoint.
#
# Kept in the same file as the pure math so that:
#   - `openeo.UDF.from_file("roy_brdf.py")` ships math + entrypoint as one unit,
#   - the notebook can `import roy_brdf` for local validation,
#   - the math above stays directly transliterable to ONNX.
# Use with `cube.apply_dimension(process=udf, dimension="bands")` so that
# reflectance and angle bands arrive in the same chunk.
# ---------------------------------------------------------------------------
def apply_datacube(cube: xarray.DataArray, context: dict) -> xarray.DataArray:
    import numpy as np

    def band(name):
        return cube.sel(bands=name).drop_vars("bands")

    angles = {
        "sza": band("sunZenithAngles").values,
        "vza": band("viewZenithMean").values,
        "saa": band("sunAzimuthAngles").values,
        "vaa": band("viewAzimuthMean").values,
    }
    scene_center_lat = context.get("scene_center_lat")
    if scene_center_lat is None:
        # Deliberately fatal: silently defaulting to 0 deg yields a plausible but
        # wrong c-factor (theta_s ~31 instead of ~53 at Dutch latitudes).
        raise ValueError("UDF context must provide 'scene_center_lat' (degrees).")
    scene_center_lat = float(scene_center_lat)

    band_names = [str(b) for b in cube.coords["bands"].values]
    corrected = apply_nbar_bands(
        {name: band(name).values for name in band_names}, angles, scene_center_lat
    )

    axis = list(cube.dims).index("bands")
    stacked = np.stack([corrected[name] for name in band_names], axis=axis)
    return xarray.DataArray(stacked, dims=cube.dims, coords=cube.coords)


demo_ran = False


def demo():
    """Runnable self-check: fails if the kernel/c-factor logic breaks."""
    global demo_ran
    if demo_ran:
        return
    demo_ran = True

    # Identity geometry: vza=0, dphi=0, sza == sza_norm -> c must be exactly 1.
    for band, coef in ROY_COEF.items():
        c = c_factor(np.full(3, 41.0), np.zeros(3), np.zeros(3), coef, sza_norm=41.0)
        assert np.allclose(c, 1.0, atol=1e-9), (band, c)

    # Non-trivial geometry stays finite and in a sane range.
    c = c_factor(np.full(3, 41.0), np.full(3, 12.0), np.full(3, 30.0), ROY_COEF["B04"], sza_norm=41.0)
    assert np.all(np.isfinite(c)) and np.all(c > 0), c

    # acos-domain guard: extreme angles must not produce NaN.
    for sza in (0.0, 80.0), (85.0, 5.0):
        k = kgeo_li_sparse(np.array(list(sza)), np.array([5.0, 0.0]), np.array([0.0, 180.0]))
        assert np.all(np.isfinite(k)), k

    # sign/period of dphi must not matter (kernels use cos).
    a = kgeo_li_sparse(40.0, 10.0, 30.0)
    b = kgeo_li_sparse(40.0, 10.0, -30.0)
    assert np.isclose(a, b), (a, b)

    # Constant-offset geometry: c == 1.0 exactly (ratio cancels).
    rho = np.array([[0.1, 0.2], [0.3, 0.0]])
    out = nbar(rho, 41.0, 0.0, 0.0, ROY_COEF["B02"], sza_norm=41.0)
    assert np.allclose(out, rho), out

    # Non-Roy band passes through untouched.
    res = apply_nbar_bands({"B02": rho, "B05": rho}, dict(sza=41.0, vza=0.0, saa=0.0, vaa=0.0), 0.0)
    assert np.array_equal(res["B05"], rho)
    print("roy_brdf demo: OK")


if __name__ == "__main__":
    demo()
