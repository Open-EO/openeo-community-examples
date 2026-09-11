import glob
import re

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling


def load_interferogram_pairs(path_pattern):
	"""Load interferogram GeoTIFFs matching a glob pattern into pairs sorted by acquisition date."""
	tif_paths = sorted(glob.glob(path_pattern))
	pair_re = re.compile(r"phase_coh_(\d{8}T\d{6})_\d{8}T\d{6}\.tif$")

	pairs = []
	crs = None
	for path in tif_paths:
		with rasterio.open(path) as src:
			desc = [d.lower() for d in src.descriptions]
			wrapped_idx = next(i for i, d in enumerate(desc) if "phase" in d and "unw" not in d)
			unwrapped_idx = next(i for i, d in enumerate(desc) if "unw" in d)
			coh_idx = next(i for i, d in enumerate(desc) if "coh" in d)

			data = src.read()
			transform, shape = src.transform, src.shape
			crs = src.crs
			print(f"Loaded {path} of shape: {shape}")
		pairs.append({
			"date": pd.Timestamp(pair_re.search(path).group(1)),
			"wrapped": data[wrapped_idx],
			"unwrapped": data[unwrapped_idx],
			"coherence": data[coh_idx],
			"transform": transform,
			"shape": shape,
		})
	pairs.sort(key=lambda p: p["date"])
	return pairs, crs, tif_paths


def mask_unreliable_displacement(ds, displacement, coherence_threshold, gradient_percentile):
	"""Mask displacement pixels with low coherence or steep local phase gradients."""
	res_y = float(abs(ds.y[1] - ds.y[0]))
	res_x = float(abs(ds.x[1] - ds.x[0]))

	dy, dx = np.gradient(ds["unwrapped_phase"].values, res_y, res_x, axis=(1, 2))
	gradient_mag = np.hypot(dx, dy)
	gradient_cutoff = np.nanpercentile(gradient_mag, gradient_percentile, axis=(1, 2), keepdims=True)

	unreliable = (ds["coherence"].values < coherence_threshold) | (gradient_mag >= gradient_cutoff)
	return displacement.where(~unreliable), gradient_cutoff


def fit_linear_deformation_trend(displacement_stack, t_days):
	"""Fit a per-pixel linear displacement-vs-time trend and derive total displacement."""
	t = t_days[:, None, None]
	valid = ~np.isnan(displacement_stack)
	y = np.where(valid, displacement_stack, 0.0)

	n_valid = valid.sum(axis=0)
	n_valid_f = n_valid.astype("float64")
	enough_data = n_valid >= 2

	sum_t = np.where(valid, t, 0.0).sum(axis=0)
	sum_tt = np.where(valid, t ** 2, 0.0).sum(axis=0)
	sum_y = y.sum(axis=0)
	sum_ty = (t * y).sum(axis=0)

	A = np.stack([
		np.stack([sum_tt, sum_t], axis=-1),
		np.stack([sum_t, n_valid_f], axis=-1),
	], axis=-2)
	b = np.stack([sum_ty, sum_y], axis=-1)

	det = sum_tt * n_valid_f - sum_t ** 2
	solvable = enough_data & (det != 0)

	velocity = np.full(n_valid.shape, np.nan)
	intercept = np.full(n_valid.shape, np.nan)
	coeffs = np.linalg.solve(A[solvable], b[solvable][..., None])[..., 0]
	velocity[solvable] = coeffs[:, 0]
	intercept[solvable] = coeffs[:, 1]

	# normal equations for y = velocity*t + intercept; NaN propagates automatically where unsolved
	total_displacement = velocity * t[-1] + intercept
	return velocity, intercept, total_displacement, enough_data, solvable, det


def find_peak_pixel(total_displacement, displacement_stack, velocity, intercept):
	"""Locate the pixel with the largest absolute fitted displacement and its per-pair series."""
	peak_y, peak_x = np.unravel_index(np.nanargmax(np.abs(total_displacement)), total_displacement.shape)
	pixel_displacement = displacement_stack[:, peak_y, peak_x]
	pixel_velocity = velocity[peak_y, peak_x]
	pixel_intercept = intercept[peak_y, peak_x]
	return peak_y, peak_x, pixel_displacement, pixel_velocity, pixel_intercept


def align_interferogram_pairs(pairs, crs):
	"""Reproject a list of interferogram pairs onto one shared raster grid and stack them into a dataset."""
	res_x, res_y = abs(pairs[0]["transform"].a), abs(pairs[0]["transform"].e)
	xmin = min(p["transform"].c for p in pairs)
	ymax = max(p["transform"].f for p in pairs)
	xmax = max(p["transform"].c + p["shape"][1] * p["transform"].a for p in pairs)
	ymin = min(p["transform"].f + p["shape"][0] * p["transform"].e for p in pairs)

	width = int(round((xmax - xmin) / res_x))
	height = int(round((ymax - ymin) / res_y))
	dst_transform = from_origin(xmin, ymax, res_x, res_y)

	def align(band, transform):
		dst = np.full((height, width), np.nan, dtype="float32")
		reproject(source=band, destination=dst, src_transform=transform, src_crs=crs,
				  dst_transform=dst_transform, dst_crs=crs, src_nodata=np.nan, dst_nodata=np.nan,
				  resampling=Resampling.nearest)  # nearest avoids blending across the phase wrap
		return dst

	ds = xr.Dataset(
		{
			"wrapped_phase": (("time", "y", "x"), np.stack([align(p["wrapped"], p["transform"]) for p in pairs])),
			"unwrapped_phase": (("time", "y", "x"), np.stack([align(p["unwrapped"], p["transform"]) for p in pairs])),
			"coherence": (("time", "y", "x"), np.stack([align(p["coherence"], p["transform"]) for p in pairs])),
		},
		coords={
			"time": pd.DatetimeIndex([p["date"] for p in pairs]),
			"y": ymax - res_y * (np.arange(height) + 0.5),
			"x": xmin + res_x * (np.arange(width) + 0.5),
		},
	).sortby("time")
	return ds


def set_plot_defaults():
	"""Apply consistent defaults for the local analysis figures."""
	plt.rcParams.update({
		"figure.facecolor": "white",
		"axes.facecolor": "white",
		"axes.edgecolor": "#c3c2b7",
		"axes.grid": True,
		"grid.color": "#e1e0d9",
		"grid.linewidth": 0.6,
		"axes.spines.top": False,
		"axes.spines.right": False,
		"font.size": 12,
	})


def plot_harvest_timing_map(harvest_doy, extent, start_day, end_day, month_days, month_labels):
	"""Plot the first detected harvest day for each spatial pixel."""
	harvest_doy_map = np.ma.masked_invalid(harvest_doy.values)

	fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
	ax.grid(False)
	image = ax.imshow(
		harvest_doy_map,
		extent=extent,
		origin="lower",
		vmin=start_day,
		vmax=end_day,
		cmap="viridis",
	)
	ax.set_title("Detected Harvest Timing", fontsize=15, fontweight="bold")
	ax.set_xlabel("X (m)")
	ax.set_ylabel("Y (m)")
	colorbar = fig.colorbar(image, ax=ax, ticks=month_days, shrink=0.85)
	colorbar.ax.set_yticklabels(month_labels)
	colorbar.set_label("Harvest month")
	plt.show()
	return fig, ax


def plot_coherence_samples(coherence, harvest_doy, sample_idx, palette=None):
	"""Plot coherence time series and detected harvest dates for sample pixels."""
	palette = palette or ["#2a78d6", "#eb6834", "#1baf7a"]
	year = pd.Timestamp(coherence.t.values[0]).year

	fig, ax = plt.subplots(figsize=(9, 5))
	for color, (y, x) in zip(palette, sample_idx):
		series = coherence.isel(y=int(y), x=int(x))
		ax.plot(
			series["t"].values,
			series.values,
			marker="o",
			markersize=5,
			linewidth=2,
			color=color,
			label=f"pixel (y={y}, x={x})",
		)

		detected_doy = harvest_doy.isel(y=int(y), x=int(x)).item()
		if np.isfinite(detected_doy):
			harvest_date = pd.Timestamp(f"{year}-01-01") + pd.Timedelta(days=detected_doy - 1)
			ax.axvline(harvest_date, color=color, linestyle="--", alpha=0.6)

	ax.set_ylabel("VH coherence")
	ax.set_xlabel("Date")
	ax.set_title("VH Coherence Time Series", fontsize=15, fontweight="bold")
	ax.legend(frameon=False)
	fig.autofmt_xdate()
	plt.show()
	return fig, ax


def plot_interferogram_quicklook(ds, time_idx=1):
	"""Plot wrapped phase, unwrapped phase and coherence for one interferogram pair."""
	fig, axs = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
	im0 = axs[0].imshow(ds["wrapped_phase"].isel(time=time_idx), cmap="hsv")
	axs[0].set_title("Wrapped phase")
	fig.colorbar(im0, ax=axs[0], label="radians", shrink=0.8)
	im1 = axs[1].imshow(ds["unwrapped_phase"].isel(time=time_idx), cmap="hsv")
	axs[1].set_title("Unwrapped phase")
	fig.colorbar(im1, ax=axs[1], label="radians", shrink=0.8)
	im2 = axs[2].imshow(ds["coherence"].isel(time=time_idx), cmap="gray", vmin=0, vmax=1)
	axs[2].set_title("Coherence")
	fig.colorbar(im2, ax=axs[2], shrink=0.8)
	for ax in axs:
		ax.grid(False)
	fig.suptitle(
		f"Interferogram pair: {pd.Timestamp(ds['wrapped_phase'].time.values[time_idx]).date()}",
		fontweight="bold",
	)
	plt.show()
	return fig, axs


def plot_displacement_velocity_maps(total_displacement, velocity_mm_yr, extent, dates):
	"""Plot the total LOS displacement and the fitted LOS velocity maps side by side."""
	fig, axs = plt.subplots(2, 1, figsize=(11, 13), constrained_layout=True)
	for ax in axs:
		ax.grid(False)
		ax.set_xlabel("Longitude [°]")
		ax.set_ylabel("Latitude [°]")

	im0 = axs[0].imshow(total_displacement * 1000, cmap="RdBu_r", vmin=-50, vmax=50,
						 extent=extent, origin="upper", interpolation="bilinear")
	axs[0].set_title("Total displacement estimate [mm]\n" +
					  f"{dates[0].date()} to {dates[-1].date()}", fontweight="bold")
	fig.colorbar(im0, ax=axs[0], shrink=0.85,
				 label="Total displacement estimate [mm]\n(negative = away from satellite)")
	im1 = axs[1].imshow(velocity_mm_yr, cmap="RdBu_r", vmin=-50, vmax=50,
						 extent=extent, origin="upper", interpolation="bilinear")
	axs[1].set_title("Fitted LOS velocity [mm/yr]\n" +
					  f"{dates[0].date()} to {dates[-1].date()}", fontweight="bold")
	fig.colorbar(im1, ax=axs[1], shrink=0.85,
				 label="LOS velocity [mm/yr]\n(negative = away from satellite)")
	plt.show()
	return fig, axs


def plot_pixel_displacement_trend(dates, t_days, pixel_displacement, pixel_velocity, pixel_intercept, peak_y, peak_x):
	"""Plot per-pair LOS displacement for one pixel against its fitted linear trend."""
	fig, ax = plt.subplots(figsize=(9, 5))
	ax.scatter(dates, pixel_displacement, s=50, color="#2a78d6", zorder=3, label="per-pair LOS displacement")
	fit_line = pixel_velocity * t_days + pixel_intercept
	ax.plot(dates, fit_line, color="#e34948", linestyle="--", linewidth=2, label="linear fit")
	ax.axhline(0, color="grey", linewidth=1, linestyle="--", alpha=0.7)

	ax.set_ylabel("LOS displacement [m]")
	ax.set_xlabel("Date")
	ax.set_title(f"Displacement trend at the peak-deformation pixel (row {peak_y}, col {peak_x})", fontweight="bold")
	ax.legend(frameon=False)

	fig.autofmt_xdate()  # rotate date labels
	plt.show()
	return fig, ax


def plot_interactive_harvest_map(harvest_doy, start_day, end_day):
	"""Create an interactive harvest-timing map with an OpenStreetMap layer."""
	import cartopy.crs as ccrs
	import geoviews as gv
	import holoviews as hv

	hv.extension("bokeh")
	gv.extension("bokeh")

	raster = gv.Image(
		harvest_doy,
		kdims=["x", "y"],
		vdims="harvest_doy",
		crs=ccrs.UTM(zone=32, southern_hemisphere=False),
	).opts(
		cmap="viridis",
		alpha=0.8,
		colorbar=True,
		clim=(start_day, end_day),
		width=900,
		height=600,
		tools=["hover"],
		title="Detected Harvest Timing",
	)
	return raster * gv.tile_sources.OSM
