"""
Gradia - Color grading transfer tool.

Applies the tonal/color style of a reference image onto one or more target images.

Methods available via --method:
  reinhard      Statistical Transfer. Mean/stddev shift in LAB space.
  kantorovich   Linear Transport. Gaussian optimal transport mapping.
  forgy         Palette Matching. K-means cluster-based color matching.
  wasserstein   Optimal Transport. Sliced Wasserstein distribution matching.

Other features:
  - --output-dir, --output-suffix
  - Before/after histogram visualization (--visualize)
  - tqdm progress bar
  - 16-bit / HDR image support
  - Logging with --verbose / --quiet
  - Graceful size mismatch warnings
"""

import cv2
import numpy as np
import argparse
import logging
from pathlib import Path

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import ot  # Python Optimal Transport (POT)
    OT_AVAILABLE = True
except ImportError:
    OT_AVAILABLE = False

from sklearn.cluster import KMeans


# ---------------------------------------------------------------------------
# Global config
# ---------------------------------------------------------------------------

_MAX_VAL_8BIT  = 255
_MAX_VAL_16BIT = 65535


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

log = logging.getLogger("Gradia")


def setup_logging(verbose: bool = False, quiet: bool = False):
    level = logging.WARNING if quiet else (logging.DEBUG if verbose else logging.INFO)
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )


# ---------------------------------------------------------------------------
# Image I/O helpers
# ---------------------------------------------------------------------------

def load_image(path: str) -> tuple:
    """Load image preserving bit depth. Returns (array, bit_depth)."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    if img.dtype == np.uint16:
        bit_depth = 16
    else:
        img = img.astype(np.uint8)
        bit_depth = 8
    log.debug(f"Loaded {path} - shape={img.shape}, dtype={img.dtype}")
    return img, bit_depth


def to_8bit(img: np.ndarray) -> np.ndarray:
    """Safely convert any image to uint8 for display or colorspace ops."""
    if img.dtype == np.uint16:
        return (img / 256).astype(np.uint8)
    return img.astype(np.uint8)


def warn_size_mismatch(reference: np.ndarray, target: np.ndarray, target_path: str):
    rh, rw = reference.shape[:2]
    th, tw = target.shape[:2]
    ratio_h = max(rh, th) / min(rh, th)
    ratio_w = max(rw, tw) / min(rw, tw)
    if ratio_h > 2.0 or ratio_w > 2.0:
        log.warning(
            f"Large resolution mismatch: reference={rw}x{rh}, target={tw}x{th} "
            f"({Path(target_path).name}). Results may look unexpected."
        )
    elif rh != th or rw != tw:
        log.info(
            f"Resolution mismatch: reference={rw}x{rh}, target={tw}x{th} "
            f"({Path(target_path).name}) - this is fine."
        )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _blend(result: np.ndarray, original: np.ndarray, intensity: float, bit_depth: int) -> np.ndarray:
    """Linear blend between graded result and original target."""
    if intensity >= 1.0:
        return result
    val_max = _MAX_VAL_8BIT if bit_depth == 8 else _MAX_VAL_16BIT
    dtype   = np.uint8      if bit_depth == 8 else np.uint16
    blended = result.astype(np.float32) * intensity + original.astype(np.float32) * (1.0 - intensity)
    return np.clip(blended, 0, val_max).astype(dtype)



# ---------------------------------------------------------------------------
# Gradia - main class
# ---------------------------------------------------------------------------

class Gradia:
    """
    Color grading transfer engine.

    Load a reference image once, then call any grade_* method to apply
    its tonal/color style onto a target image.

    Example:
        grader = Gradia("reference.jpg", intensity=0.8)
        result = grader.grade_reinhard(target_array)
        cv2.imwrite("output.jpg", result)
    """

    def __init__(self, reference_path: str, intensity: float = 0.8):
        self.reference, self.ref_bit_depth = load_image(reference_path)
        self.intensity = intensity
        log.info(f"Gradia initialized with reference: {reference_path}")

    # -----------------------------------------------------------------------
    # METHOD 1 - Reinhard (Statistical Transfer)
    # -----------------------------------------------------------------------

    def grade_reinhard(self, target: np.ndarray, bit_depth: int = 8) -> np.ndarray:
        """
        Reinhard et al. (2001) color transfer in LAB space.

        For each channel:
            result = (channel - mean_target) * (std_reference / std_target) + mean_reference

        Shifts the target distribution to match the reference mean and spread
        without touching inter-channel relationships.
        """
        ref8 = to_8bit(self.reference)
        tgt8 = to_8bit(target)

        ref_lab = cv2.cvtColor(ref8, cv2.COLOR_BGR2LAB).astype(np.float32)
        tgt_lab = cv2.cvtColor(tgt8, cv2.COLOR_BGR2LAB).astype(np.float32)

        result_lab = tgt_lab.copy()

        for i in range(3):
            ref_mean, ref_std = ref_lab[:, :, i].mean(), ref_lab[:, :, i].std()
            tgt_mean, tgt_std = tgt_lab[:, :, i].mean(), tgt_lab[:, :, i].std()

            if tgt_std < 1e-6:
                result_lab[:, :, i] = tgt_lab[:, :, i] - tgt_mean + ref_mean
            else:
                result_lab[:, :, i] = (
                    (tgt_lab[:, :, i] - tgt_mean) * (ref_std / tgt_std) + ref_mean
                )

        result_lab[:, :, 0] = np.clip(result_lab[:, :, 0], 0, 255)
        result_lab[:, :, 1] = np.clip(result_lab[:, :, 1], 0, 255)
        result_lab[:, :, 2] = np.clip(result_lab[:, :, 2], 0, 255)

        result8 = cv2.cvtColor(result_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        result  = result8.astype(np.uint16) * 256 if bit_depth == 16 else result8
        return _blend(result, target, self.intensity, bit_depth)

    # -----------------------------------------------------------------------
    # METHOD 2 - Kantorovich (Linear Transport)
    # -----------------------------------------------------------------------

    def grade_kantorovich(self, target: np.ndarray, bit_depth: int = 8,
                           sample_size: int = 50000) -> np.ndarray:
        """
        Monge-Kantorovich optimal transport under a Gaussian assumption.

        Treats both color distributions as multivariate Gaussians and finds
        the optimal linear affine map (Monge map) that moves the target
        distribution to match the reference. Because it assumes Gaussianity,
        the solution is closed-form and efficient. Works well when color
        distributions are roughly unimodal. May underperform on images with
        complex, multimodal color distributions - use wasserstein in that case.

        Args:
            sample_size: Max pixels sampled for transport computation.

        Requires: pip install POT
        """
        if not OT_AVAILABLE:
            log.error(
                "Kantorovich method requires the POT library. "
                "Install with: pip install POT"
            )
            log.warning("Falling back to Reinhard method.")
            return self.grade_reinhard(target, bit_depth=bit_depth)

        ref8 = to_8bit(self.reference)
        tgt8 = to_8bit(target)

        ref_lab = cv2.cvtColor(ref8, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0
        tgt_lab = cv2.cvtColor(tgt8, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0

        h, w = tgt_lab.shape[:2]

        ref_flat = ref_lab.reshape(-1, 3)
        tgt_flat = tgt_lab.reshape(-1, 3)

        rng = np.random.default_rng(42)
        ref_sample = ref_flat[rng.choice(len(ref_flat), min(sample_size, len(ref_flat)), replace=False)]
        tgt_sample = tgt_flat[rng.choice(len(tgt_flat), min(sample_size, len(tgt_flat)), replace=False)]

        log.debug("Running Kantorovich (Gaussian optimal transport)...")

        ot_map = ot.da.LinearTransport()
        ot_map.fit(Xs=tgt_sample, Xt=ref_sample)

        result_flat = ot_map.transform(Xs=tgt_flat)
        result_flat = np.clip(result_flat, 0, 1)

        result_lab = (result_flat.reshape(h, w, 3) * 255.0).astype(np.uint8)
        result8    = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)

        result = result8.astype(np.uint16) * 256 if bit_depth == 16 else result8
        return _blend(result, target, self.intensity, bit_depth)

    # -----------------------------------------------------------------------
    # METHOD 3 - Forgy (Palette Matching)
    # -----------------------------------------------------------------------

    def grade_forgy(self, target: np.ndarray, bit_depth: int = 8,
                    n_colors: int = 8) -> np.ndarray:
        """
        Palette-based color grading using K-means clustering (Forgy, 1965).

        Finds dominant colors in both reference and target, maps target
        palette entries to their nearest reference counterparts, then shifts
        each target pixel by the delta of its nearest palette match with a
        Gaussian-weighted falloff.

        Args:
            n_colors: Number of palette colors to extract. Range: 4-16.
        """
        ref8 = to_8bit(self.reference)
        tgt8 = to_8bit(target)

        ref_lab = cv2.cvtColor(ref8, cv2.COLOR_BGR2LAB).astype(np.float32)
        tgt_lab = cv2.cvtColor(tgt8, cv2.COLOR_BGR2LAB).astype(np.float32)

        h, w = tgt_lab.shape[:2]

        def get_samples(lab_img, max_samples=50000):
            flat = lab_img.reshape(-1, 3)
            if len(flat) > max_samples:
                idx = np.random.choice(len(flat), max_samples, replace=False)
                return flat[idx]
            return flat

        log.debug(f"Fitting K-means with {n_colors} clusters...")

        ref_samples = get_samples(ref_lab)
        tgt_samples = get_samples(tgt_lab)

        km_ref = KMeans(n_clusters=n_colors, random_state=42, n_init="auto").fit(ref_samples)
        km_tgt = KMeans(n_clusters=n_colors, random_state=42, n_init="auto").fit(tgt_samples)

        ref_centers = km_ref.cluster_centers_
        tgt_centers = km_tgt.cluster_centers_

        tgt_flat    = tgt_lab.reshape(-1, 3)
        result_flat = tgt_flat.copy()

        # Precompute per-cluster deltas: the color offset from each target
        # cluster center to its nearest reference cluster center.
        deltas = np.zeros((n_colors, 3), dtype=np.float32)
        for ci in range(n_colors):
            tgt_color = tgt_centers[ci]
            dists     = np.linalg.norm(ref_centers - tgt_color, axis=1)
            ref_match = ref_centers[np.argmin(dists)]
            deltas[ci] = ref_match - tgt_color

        # Soft clustering. Instead of hard-assigning each pixel to exactly
        # one cluster (which produces visible seams at cluster boundaries),
        # compute a weight for every pixel against every cluster based on
        # distance. The final correction is a weighted average of all
        # cluster deltas, which blends smoothly across boundaries.
        #
        # Sigma is set from the median distance between all cluster
        # centers - this auto-scales the falloff to the palette geometry,
        # so tightly-packed palettes get sharper weights and widely-spaced
        # palettes get softer ones.
        center_dists = np.linalg.norm(
            tgt_centers[:, None, :] - tgt_centers[None, :, :], axis=2
        )
        # Use the upper triangle (excluding the diagonal of zeros) for the median
        upper = center_dists[np.triu_indices(n_colors, k=1)]
        sigma = (np.median(upper) + 1e-6) * 0.75

        # Distance from every pixel to every cluster center.
        # Shape: (n_pixels, n_colors)
        pixel_to_centers = np.linalg.norm(
            tgt_flat[:, None, :] - tgt_centers[None, :, :], axis=2
        )

        # Gaussian weights, normalized so each pixel's weights sum to 1.
        # Shape: (n_pixels, n_colors)
        weights = np.exp(-0.5 * (pixel_to_centers / sigma) ** 2)
        weights /= weights.sum(axis=1, keepdims=True) + 1e-12

        # Weighted sum of deltas: each pixel gets a smooth blend of every
        # cluster's correction, proportional to how close it is to each.
        # Shape: (n_pixels, 3)
        blended_delta = weights @ deltas

        result_flat += blended_delta

        result_lab = result_flat.reshape(h, w, 3)
        result_lab[:, :, 0] = np.clip(result_lab[:, :, 0], 0, 255)
        result_lab[:, :, 1] = np.clip(result_lab[:, :, 1], 0, 255)
        result_lab[:, :, 2] = np.clip(result_lab[:, :, 2], 0, 255)

        result8 = cv2.cvtColor(result_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        result  = result8.astype(np.uint16) * 256 if bit_depth == 16 else result8
        return _blend(result, target, self.intensity, bit_depth)

    # -----------------------------------------------------------------------
    # METHOD 4 - Wasserstein (Optimal Transport)
    # -----------------------------------------------------------------------

    def grade_wasserstein(self, target: np.ndarray, bit_depth: int = 8,
                           n_slices: int = 20, sample_size: int = 50000) -> np.ndarray:
        """
        Sliced Wasserstein optimal transport color grading via iterative advection.

        Projects both color distributions onto a random 1D direction in LAB
        space, solves exact 1D optimal transport on that slice via quantile
        matching, and applies the displacement immediately before sampling
        the next random direction. Over many iterations the source
        distribution converges onto the reference distribution.

        Handles complex, multimodal color distributions that confuse linear
        methods. Does not require the POT library.

        Args:
            n_slices:    Number of iterations. Each iteration samples one
                         random direction and advects the source. 20 is a
                         good starting point. More is not always better -
                         quantile-matching noise can push pixels past the
                         converged reference at higher counts, so going
                         well beyond 50-100 often makes results worse
                         rather than better.
            sample_size: Max reference pixels sampled for quantile estimation.
                         Higher = more accurate quantile matching.
        """
        ref8 = to_8bit(self.reference)
        tgt8 = to_8bit(target)

        ref_lab = cv2.cvtColor(ref8, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0
        tgt_lab = cv2.cvtColor(tgt8, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0

        h, w = tgt_lab.shape[:2]

        ref_flat = ref_lab.reshape(-1, 3)
        tgt_flat = tgt_lab.reshape(-1, 3)

        rng = np.random.default_rng(42)

        # Subsample reference once for quantile estimation
        ref_sample = ref_flat[
            rng.choice(len(ref_flat), min(sample_size, len(ref_flat)), replace=False)
        ]

        result_flat = tgt_flat.copy()

        log.debug(f"Running Sliced Wasserstein transport ({n_slices} iterations)...")

        # Iterative advection. Each iteration picks a random direction in 3D
        # LAB space, projects both clouds onto that line, solves the 1D
        # optimal transport via quantile matching, then APPLIES the
        # displacement immediately before the next iteration. This is the
        # critical difference from accumulate-and-average: each subsequent
        # direction sees a source that has already been partially transported
        # toward the reference, so the algorithm converges on the reference
        # distribution over successive iterations rather than averaging many
        # tiny corrections that never fully commit.
        for _ in range(n_slices):
            # Random unit direction in 3D LAB color space
            direction = rng.standard_normal(3)
            direction /= np.linalg.norm(direction)

            # Project both clouds onto this 1D direction
            ref_proj = ref_sample  @ direction     # shape: (n_ref,)
            tgt_proj = result_flat @ direction     # shape: (n_tgt,)

            # 1D OT via quantile matching - sort reference, rank target,
            # map each target point to the reference value at the same quantile
            ref_sorted  = np.sort(ref_proj)
            n_ref       = len(ref_sorted)
            n_tgt       = len(tgt_proj)

            tgt_argsort              = np.argsort(tgt_proj)
            tgt_ranks                = np.empty(n_tgt, dtype=np.int64)
            tgt_ranks[tgt_argsort]   = np.arange(n_tgt)

            ref_indices = np.clip(
                (tgt_ranks * (n_ref - 1) / max(n_tgt - 1, 1)).astype(np.int64),
                0, n_ref - 1
            )

            # Displacement: how far each target pixel needs to move along
            # this random direction to match the reference distribution.
            #
            # For each target pixel, the i-th smallest target projection
            # should become the i-th smallest reference projection (that's
            # the 1D optimal transport solution - sort-and-match by rank).
            # Subtracting the current target projection from the matched
            # reference projection gives the 1D distance the pixel needs to
            # travel. Back-projecting this scalar along the 3D direction
            # (via the outer product below) turns it into a 3D color move.
            displacement = ref_sorted[ref_indices] - tgt_proj

            # Apply the displacement with a step factor (iterative advection).
            #
            # Quantile matching is noisy because the reference is sampled
            # and target ranks map imprecisely to reference indices. Over
            # many iterations, this noise causes small residual movements
            # even after the source has converged toward the reference,
            # and some pixels end up pushed outside [0, 1]. The final clip
            # then crushes those pixels to the edges of the range, which
            # shows up as blown highlights, crushed shadows, or desaturated
            # colors.
            #
            # A step factor of 0.5 halves each iteration's move to keep
            # total accumulated motion bounded. Combined with the low
            # default iteration count (20), this gives a stable result
            # that converges without overshooting. Users who want to go
            # higher can do so via --n-slices, but note that more is not
            # always better - beyond ~50-100 iterations the method tends
            # to plateau or regress as noise dominates.
            STEP_FACTOR = 0.5
            result_flat += STEP_FACTOR * np.outer(displacement, direction)

        result_flat = np.clip(result_flat, 0, 1)

        # Convert back to uint8 LAB before the colorspace conversion.
        # cv2.cvtColor interprets LAB ranges differently based on dtype:
        # uint8 input expects L, a, b all in [0, 255] (OpenCV's internal
        # uint8 encoding), while float32 input expects L in [0, 100] and
        # a, b in [-128, 127]. Since we stored normalized uint8-encoded
        # LAB values (divided by 255 after BGR2LAB on uint8 input), we
        # MUST cast back to uint8 here - passing a float32 array of those
        # same numerical values would be silently misinterpreted and
        # produce wildly wrong output colors that mostly preserve L but
        # wreck a and b, which looks like "only exposure changed".
        result_lab = (result_flat.reshape(h, w, 3) * 255.0).astype(np.uint8)
        result8    = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)

        result = result8.astype(np.uint16) * 256 if bit_depth == 16 else result8
        return _blend(result, target, self.intensity, bit_depth)

    # -----------------------------------------------------------------------
    # Process a file path and save output
    # -----------------------------------------------------------------------

    def process(self, target_path: str, method: str = "reinhard",
                 output_dir: Path = None, output_suffix: str = None,
                 n_colors: int = 8, n_slices: int = 20, sample_size: int = 50000,
                 visualize: bool = False, preview: bool = False) -> Path:
        """
        Grade a single target image file and write the output.

        Returns the output path.
        """
        target_path = Path(target_path)
        log.info(f"Processing: {target_path.name}  [method={method}]")

        target, tgt_bit_depth = load_image(str(target_path))
        warn_size_mismatch(self.reference, target, str(target_path))

        bit_depth = max(self.ref_bit_depth, tgt_bit_depth)
        if self.ref_bit_depth != tgt_bit_depth:
            log.warning(
                f"Bit depth mismatch: reference={self.ref_bit_depth}-bit, "
                f"target={tgt_bit_depth}-bit. Operating at {bit_depth}-bit."
            )

        ref = self.reference.astype(np.uint16) * 256 \
              if (self.reference.dtype == np.uint8 and bit_depth == 16) else self.reference
        tgt = target.astype(np.uint16) * 256 \
              if (target.dtype == np.uint8 and bit_depth == 16) else target

        if method == "reinhard":
            result = self.grade_reinhard(tgt, bit_depth=bit_depth)
        elif method == "kantorovich":
            result = self.grade_kantorovich(tgt, bit_depth=bit_depth,
                                             sample_size=sample_size)
        elif method == "forgy":
            result = self.grade_forgy(tgt, bit_depth=bit_depth, n_colors=n_colors)
        else:  # wasserstein
            result = self.grade_wasserstein(tgt, bit_depth=bit_depth,
                                             n_slices=n_slices, sample_size=sample_size)

        out_dir  = output_dir or target_path.parent
        out_path = out_dir / (target_path.stem + output_suffix + target_path.suffix)

        if preview:
            cv2.imshow(
                f"Original | Graded [{method}] - {target_path.name}",
                np.hstack([to_8bit(tgt), to_8bit(result)])
            )
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        cv2.imwrite(str(out_path), result)
        log.info(f"  Saved -> {out_path}")

        if visualize:
            viz_path = out_path.with_suffix(".histogram.png")
            self._save_visualization(ref, tgt, result, viz_path, bit_depth)

        return out_path

    # -----------------------------------------------------------------------
    # Visualization
    # -----------------------------------------------------------------------

    def _save_visualization(self, reference, target, result, out_path, bit_depth):
        if not MATPLOTLIB_AVAILABLE:
            log.warning("matplotlib not installed - skipping visualization.")
            return

        is_color   = len(reference.shape) == 3
        n_channels = reference.shape[2] if is_color else 1
        ch_names   = ["Blue", "Green", "Red"] if n_channels == 3 else ["Gray"]
        colors     = ["#4477CC", "#44BB99", "#EE6677"] if n_channels == 3 else ["#888888"]

        val_max = _MAX_VAL_8BIT if bit_depth == 8 else _MAX_VAL_16BIT
        n_bins  = min(256, val_max + 1)

        fig, axes = plt.subplots(n_channels, 1, figsize=(10, 3.5 * n_channels), tight_layout=True)
        if n_channels == 1:
            axes = [axes]

        fig.suptitle(f"Gradia - {out_path.stem}", fontsize=13, fontweight="bold")

        for i in range(n_channels):
            ref_ch = reference[:, :, i] if is_color else reference
            tgt_ch = target[:, :, i]    if is_color else target
            res_ch = result[:, :, i]    if is_color else result

            ax = axes[i]
            ax.hist(tgt_ch.flatten(), bins=n_bins, range=(0, val_max),
                    alpha=0.5, color=colors[i], label="Target (before)", density=True)
            ax.hist(res_ch.flatten(), bins=n_bins, range=(0, val_max),
                    alpha=0.5, color="orange",   label="Target (after)",  density=True)
            ax.hist(ref_ch.flatten(), bins=n_bins, range=(0, val_max),
                    alpha=0.4, color="black",    label="Reference",        density=True,
                    histtype="step", linewidth=1.2)
            ax.set_title(f"{ch_names[i]} - Histogram")
            ax.set_xlabel("Pixel value")
            ax.set_ylabel("Density")
            ax.legend(fontsize=8)

        fig.savefig(str(out_path), dpi=150)
        plt.close(fig)
        log.info(f"  Visualization saved -> {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gradia - Apply the tonal/color style of a reference image onto target images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("reference",
        help="Reference image whose style you want to apply.")
    parser.add_argument("targets", nargs="+",
        help="Target image(s) to grade.")

    parser.add_argument("-o", "--output-dir",
        type=Path, default=None,
        help="Directory to write output files. Defaults to same directory as each target.")
    parser.add_argument("--output-suffix",
        default=None,
        help="Suffix appended before file extension. Defaults to _graded_<reference_stem>.")

    parser.add_argument("-i", "--intensity",
        type=float, default=0.8,
        help="Grade intensity: 0.0 = no change, 1.0 = full grade.")

    parser.add_argument("-m", "--method",
        choices=["reinhard", "kantorovich", "forgy", "wasserstein"],
        default="reinhard",
        help=(
            "Grading method: "
            "'reinhard' = Statistical Transfer, mean/std LAB shift (safe default); "
            "'kantorovich' = Linear Transport, Gaussian optimal transport (requires POT); "
            "'forgy' = Palette Matching, K-means cluster-based color matching; "
            "'wasserstein' = Optimal Transport, sliced Wasserstein distribution matching."
        ))

    # Forgy-specific
    parser.add_argument("--n-colors",
        type=int, default=8,
        help="(forgy method) Number of K-means palette colors. Range: 4-16.")

    # Wasserstein-specific
    parser.add_argument("--n-slices",
        type=int, default=20,
        help="(wasserstein method) Number of advection iterations. 20 is a good starting point - more is not always better, since quantile-matching noise can overshoot the reference at higher counts.")

    # Shared by kantorovich and wasserstein
    parser.add_argument("--sample-size",
        type=int, default=50000,
        help="(kantorovich, wasserstein) Max pixels sampled for transport computation.")

    parser.add_argument("--visualize", action="store_true",
        help="Save a before/after histogram PNG alongside each output.")
    parser.add_argument("--preview", action="store_true",
        help="Show each result in an OpenCV window before saving.")

    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    parser.add_argument("-q", "--quiet",   action="store_true", help="Suppress all non-error output.")

    return parser


def run():
    parser = build_parser()
    args   = parser.parse_args()
    setup_logging(verbose=args.verbose, quiet=args.quiet)

    if args.method == "kantorovich" and not OT_AVAILABLE:
        log.warning(
            "POT library not found - kantorovich method will fall back to reinhard. "
            "Install with: pip install POT"
        )

    suffix = args.output_suffix or f"_graded_{Path(args.reference).stem}"

    grader = Gradia(
        reference_path=args.reference,
        intensity=args.intensity,
    )

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    targets  = args.targets
    iterator = (
        tqdm(targets, desc="Grading", unit="img")
        if TQDM_AVAILABLE and not args.quiet else targets
    )

    for target_path in iterator:
        try:
            grader.process(
                target_path=target_path,
                method=args.method,
                output_dir=args.output_dir,
                output_suffix=suffix,
                n_colors=args.n_colors,
                n_slices=args.n_slices,
                sample_size=args.sample_size,
                visualize=args.visualize,
                preview=args.preview,
            )
        except FileNotFoundError as e:
            log.error(str(e))
            continue

    log.info("Done.")


if __name__ == "__main__":
    run()
