# TESSERA v2 pixel embeddings from openEO-loaded Sentinel-1/2

Computes [TESSERA v2](https://github.com/ucam-eo/tessera) pixel embeddings for a custom
AOI/year, directly from openEO. TESSERA embeddings are usually only available as
[precomputed global tiles](https://github.com/ucam-eo/geotessera) - this example runs the actual
TESSERA v2 "student" encoder (vendored PyTorch code) inside an openEO UDF, so it works for any
area and any year, not just the tiles Cambridge has already published.

## What this demonstrates

- `merge_cubes` with mismatched dimension labels: Sentinel-2 and the two Sentinel-1 orbits
  (ascending/descending) are loaded as separate cubes with different acquisition dates, then
  merged into one cube whose `t` dimension is the union of all three sources.
- `apply_dimension` collapsing a `t` (time) dimension into a `bands` dimension of a different
  size - the UDF here turns a variable-length time series into a fixed 128-band embedding.
- Running a real PyTorch model in a UDF via the `udf-dependency-archives` job option, including
  downloading model weights from a URL at runtime.
- `SENTINEL2_L2A`, `SENTINEL1_GRD`.

## Requirements

- An openEO backend with `SENTINEL2_L2A` and `SENTINEL1_GRD` collections (this example targets
  the [Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu/) openEO endpoint).
- A backend that supports the `udf-dependency-archives` job option, and a PyTorch dependency
  archive for it (see below) - check with your backend provider.

## Architecture

```
SENTINEL2_L2A (B04,B02,B03,B08,B8A,B05,B06,B07,B11,B12,SCL)  ─┐
SENTINEL1_GRD, orbit=ASCENDING  (VV_ASC, VH_ASC)              ├─ merge_cubes ─► apply_dimension(dimension="t",
SENTINEL1_GRD, orbit=DESCENDING (VV_DESC, VH_DESC)           ─┘                 target_dimension="bands", process=UDF)
                                                                                          │
                                                                                          ▼
                                                                          128-band TESSERA v2 embedding cube
```

`merge_cubes` is used here for its "union of labels" behaviour: S2 and the two S1 orbits have
different acquisition dates, so the merged cube's `t` dimension is the union of all three, with
`NaN` wherever a given band's source has no observation on that date. The UDF splits everything
back apart by band name + non-`NaN` dates, reconstructing the exact `(T, H, W, C)` + DOY + mask
arrays that `encode_tile()` expects, then runs the TESSERA student forward pass and collapses `t`
into a `bands` dimension of size 128.

## Running PyTorch in the UDF sandbox

Most openEO UDF sandboxes don't ship `torch` by default, but some backends let you install extra
dependencies into the sandbox via the `udf-dependency-archives` job option - a zip that gets
extracted server-side before the UDF runs. This example uses a pre-built PyTorch dependency
archive (Python 3.11):


The zip is extracted into a `feature_deps` folder next to the UDF; `udf_tessera_embedding.py` adds
that folder to `sys.path` before importing `torch`. Check with your backend provider whether an
equivalent archive is available and what folder name it uses (adjust the `sys.path.insert(...)`
call in the UDF accordingly if it differs).

`udf_tessera_embedding.py` vendors the real `tessera_infer_v2/student/model.py` and `infer.py`
source (model definition + bin-padding/inference helpers) verbatim, rather than a reimplementation
- so it faithfully reproduces the real forward pass, including the QK-norm variant if a checkpoint
uses it. At runtime the UDF downloads a `student_*.pt` checkpoint directly from a URL (passed via
`context={"weights_url": ...}`) and loads it with `torch.load`, so no local pre-processing step is
needed - just point at any HTTPS-reachable checkpoint, e.g. a `resolve/main/...` URL from one of
the `geotessera/TESSERA-V-2.0-2B-*` repos on the Hugging Face Hub.

## Files

- `udf_tessera_embedding.py` - the openEO UDF (vendored TESSERA v2 student model + input assembly).
- `TesseraEmbedding.ipynb` - loads S1/S2 via openEO, runs the UDF, visualizes the embedding (PCA
  to RGB).

## Caveats

- Load raw bands, unmodified: no compositing (max/median/mosaic), no manual rescaling
  (`0.0001 * x` etc.). The model consumes the full per-date time series with real DOY values, and
  its normalization constants are fit on raw DN/backscatter - collapsing dates or rescaling breaks
  standardization silently.
- Band harmonization (BOA offset, RTC normalization) differs from TESSERA's original MPC/AWS
  preprocessing, and the cloud mask here is SCL-based rather than the original's own mask -
  expect a reasonable but not pixel-perfect approximation of the published embeddings.
- Use the smallest student (`nano`, ~4 MB) for a quick test; `medium` (recommended default) gives
  better embeddings at the cost of a slower checkpoint download.
