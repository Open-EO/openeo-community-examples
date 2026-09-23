import numpy as np
import rasterio
from rasterio.plot import show
from pathlib import Path
import matplotlib.pyplot as plt
import geopandas as gpd
from rasterio import features
from rasterio.plot import show as rio_show
import pandas as pd

def plot_point_features_stats(csv_file, groundtruth_train, lc1_col):

    feat = pd.read_csv(csv_file)

    # join coordinates + class label via feature_index → groundtruth_train row position
    feat = feat.merge(
        groundtruth_train[["point_id", lc1_col, "geometry", "target"]]
            .reset_index()
            .rename(columns={"index": "feature_index"}),
        on="feature_index",
        how="left",
    )
    feat_gdf = gpd.GeoDataFrame(feat, geometry="geometry", crs=groundtruth_train.crs)

    classes = sorted(feat_gdf[lc1_col].dropna().unique())
    cmap = plt.colormaps["tab10"].resampled(len(classes))
    color_map = {c: cmap(i) for i, c in enumerate(classes)}

    fig, (ax_map, ax_box) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: geographic scatter coloured by class
    for cls, grp in feat_gdf.groupby(lc1_col):
        grp.plot(ax=ax_map, color=color_map[cls], markersize=40,
                label=cls, legend=False)
    ax_map.set_title("Sample points by land-cover class")
    ax_map.set_xlabel("Longitude"); ax_map.set_ylabel("Latitude")
    ax_map.ticklabel_format(useOffset=False, style="plain")
    ax_map.legend(title="LC1", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)

    # Right: per-class boxplots for NDVI / EVI / NDWI 
    index_cols = [c for c in ["NDVI", "EVI", "NDWI"] if c in feat.columns]
    n_idx      = len(index_cols)
    positions  = range(len(classes))

    for i, idx_col in enumerate(index_cols):
        offset = (i - n_idx / 2 + 0.5) * 0.25
        data   = [feat_gdf.loc[feat_gdf[lc1_col] == c, idx_col].dropna() for c in classes]
        ax_box.boxplot(data,
                    positions=[p + offset for p in positions],
                    widths=0.2,
                    patch_artist=True,
                    boxprops=dict(facecolor=f"C{i}", alpha=0.6),
                    medianprops=dict(color="black", linewidth=1.5),
                    whiskerprops=dict(linewidth=1),
                    capprops=dict(linewidth=1),
                    flierprops=dict(marker=".", markersize=3),
                    label=idx_col)

    ax_box.set_xticks(list(positions))
    ax_box.set_xticklabels(classes, rotation=30, ha="right", fontsize=8)
    ax_box.set_title("Spectral index distribution by class")
    ax_box.set_ylabel("Index value")
    ax_box.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax_box.legend(title="Index", fontsize=8)

    plt.tight_layout()
    plt.show()
    print(f"Feature table shape: {feat.shape}  |  classes: {feat[lc1_col].nunique()}")


def plot_single_patch_as_rgb(filename):
    src = rasterio.open(filename)
    img = src.read()  # shape: (bands, y, x)
    # plot RGB composite of the first patch (bands 3,2,1 = B04,B03,B02)
    fig, ax = plt.subplots(1, figsize=(6, 6))
    rgb = np.moveaxis(img[:3], 0, -1).astype(float)          # (H,W,3)
    p2, p98 = np.percentile(rgb[rgb > 0], (2, 98))
    rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-6), 0, 1)
    show(np.moveaxis(rgb, -1, 0), transform=src.transform, ax=ax)
    ax.set_title(f"RGB — {Path(src.name).stem}", fontsize=9)
    ax.ticklabel_format(useOffset=False, style="plain")
    plt.tight_layout()
    plt.show()

def plot_patches_with_polygons(output_dir, patches, df_lucas, lc1_col):


    patch_files = sorted(Path(output_dir).glob("*.tif"))
    n_show = min(9, len(patch_files))
    ncols = 3
    nrows = (n_show + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes_flat = np.array(axes).flatten()

    patches_4326 = patches.set_crs(4326, allow_override=True)
    # original (pre-centroid) LUCAS geometries keyed by point_id
    lucas_orig = df_lucas[["point_id", "geometry"]].set_index("point_id")

    for ax, pf in zip(axes_flat, patch_files[:n_show]):
        with rasterio.open(pf) as src_p:
            img_p = src_p.read([3, 2, 1]).astype(float)        # B04, B03, B02
            p2, p98 = np.percentile(img_p[img_p > 0], (2, 98))
            img_p = np.clip((img_p - p2) / (p98 - p2 + 1e-6), 0, 1)
            rio_show(img_p, transform=src_p.transform, ax=ax)

            # resolve patch_id from filename
            pid = pf.stem
            match = patches_4326[patches_4326["patch_id"] == pid]
            if match.empty:
                pid_num = pf.stem.split("_")[-1]
                match = patches_4326[patches_4326["patch_id"] == pid_num]

            # red: buffered patch square
            if not match.empty:
                match.to_crs(src_p.crs).boundary.plot(
                    ax=ax, edgecolor="red", linewidth=1.5, label="patch extent"
                )

                # blue: original LUCAS polygon (before centroid + buffer)
                pt_id = match["point_id"].values[0]
                if pt_id in lucas_orig.index:
                    orig_geom = gpd.GeoDataFrame(
                        geometry=[lucas_orig.loc[pt_id, "geometry"]],
                        crs=df_lucas.crs,
                    ).to_crs(src_p.crs)
                    orig_geom.boundary.plot(
                        ax=ax, edgecolor="dodgerblue", linewidth=1.5, label="original polygon"
                    )

            lc_label = ""
            if not match.empty and lc1_col in match.columns:
                lc_label = f" | {match[lc1_col].values[0]}"
            ax.set_title(f"{pf.stem[:22]}{lc_label}", fontsize=7)
            ax.ticklabel_format(useOffset=False, style="plain")

    for ax in axes_flat[n_show:]:
        ax.set_visible(False)

    # shared legend on the last visible axis
    handles = [
        plt.Line2D([0], [0], color="red",        linewidth=1.5, label="patch extent"),
        plt.Line2D([0], [0], color="dodgerblue", linewidth=1.5, label="original polygon"),
    ]
    axes_flat[n_show - 1].legend(handles=handles, fontsize=7, loc="lower right")

    plt.suptitle("Patch RGB composites — red: patch extent  |  blue: original polygon", fontsize=11, y=1.01)
    plt.tight_layout()
    plt.show()
