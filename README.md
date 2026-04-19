# Gradia

A command-line tool for transferring the color and tonal style of one photo onto another.

Built for photographers and designers who want to match the look of a reference image - film stock, color grade, lighting mood - without manually adjusting curves or color wheels.

---

## Table of Contents

- [How It Works](#how-it-works)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Methods](#methods)
  - [Reinhard (Statistical Transfer)](#reinhard-statistical-transfer-default)
  - [Kantorovich (Linear Transport)](#kantorovich-linear-transport)
  - [Forgy (Palette Matching)](#forgy-palette-matching)
  - [Wasserstein (Optimal Transport)](#wasserstein-optimal-transport)
- [All Options](#all-options)
- [Examples](#examples)
- [Output Files](#output-files)
- [Choosing the Right Method](#choosing-the-right-method)
- [Troubleshooting](#troubleshooting)

---

## How It Works

Every image has a statistical distribution of pixel values across its color channels. By mapping one image's distribution to match another's, you transfer its color grading and tonal style onto a target image.

Gradia offers four methods for doing this, ranging from a simple statistical shift to full sliced Wasserstein optimal transport. Each method has different tradeoffs between speed, accuracy, and how "safe" the result looks.

---

## Requirements

- Python 3.8 or higher
- The following Python libraries:

| Library | Purpose | Required |
|---|---|---|
| `opencv-python` | Image loading, colorspace conversion | Yes |
| `numpy` | Array math | Yes |
| `scikit-learn` | K-means clustering (forgy method) | Yes |
| `tqdm` | Progress bar for batch jobs | Recommended |
| `matplotlib` | Histogram visualization | Recommended |
| `POT` | Gaussian optimal transport (kantorovich method only) | Optional |

---

## Installation

**1. Clone or download the script**

```bash
git clone https://github.com/rajawski/gradia
cd gradia
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. (Optional) Install POT for the kantorovich method**

```bash
pip install POT
```

> If POT is not installed, the `kantorovich` method will automatically fall back to `reinhard` with a warning. All other methods work without POT.

---

## Quick Start

```bash
# Apply the style of reference.jpg onto photo.jpg
python gradia.py reference.jpg photo.jpg
```

Output will be saved as `photo_graded_reference.jpg` in the same folder as the target.

---

## Methods

The `--method` flag controls the algorithm used for color grading. Each method has different strengths.

---

### Reinhard (Statistical Transfer) - *default*

```bash
python gradia.py reference.jpg photo.jpg --method reinhard
```

Transfers the **mean and standard deviation** of each color channel in LAB color space. Instead of remapping individual pixel values, it shifts and scales the entire color distribution.

**How it works:** For each channel, it calculates how far each pixel is from the target's average, rescales that distance to match the reference's spread, then recenters around the reference's mean.

**Best for:**
- Subtle, natural-looking color grading
- Images with very different content (e.g. indoor vs outdoor)
- When other methods produce glitchy results
- A safe starting point for any image pair

**Tradeoffs:**
- Less aggressive than other methods - won't fully match extreme looks
- Only matches two statistical moments (mean, std), not the full distribution shape

---

### Kantorovich (Linear Transport)

```bash
pip install POT
python gradia.py reference.jpg photo.jpg --method kantorovich
```

Finds the **optimal linear affine mapping** between the two color distributions in 3D LAB space, under a Gaussian assumption (Monge-Kantorovich optimal transport). Because it assumes both distributions are roughly Gaussian, the solution is closed-form and efficient.

**How it works:** Models both color clouds as multivariate Gaussians and computes the Monge map - the linear transformation that moves the target distribution to match the reference at minimum total cost. A step up from Reinhard in accuracy without the complexity of full non-linear transport.

**Best for:**
- Images where Reinhard is too subtle but Wasserstein is overkill
- Color distributions that are roughly unimodal
- When you want a global, consistent color shift across the whole image

**Tradeoffs:**
- Requires the `POT` library (`pip install POT`)
- Falls back to Reinhard automatically if POT is not installed
- May underperform on images with complex, multimodal color distributions

**Flag:**

| Flag | Default | Description |
|---|---|---|
| `--sample-size` | `50000` | Max pixels sampled for transport computation |

---

### Forgy (Palette Matching)

```bash
python gradia.py reference.jpg photo.jpg --method forgy
```

Uses **K-means clustering** to extract the dominant colors from each image, matches reference palette colors to target palette colors by perceptual distance in LAB space, then shifts each pixel toward its matched color with a Gaussian-weighted blend.

**How it works:** Both images are reduced to their `n` most dominant colors. Each dominant color in the target is matched to the closest dominant color in the reference. Pixels are then shifted by the delta between matched colors, with a smooth falloff so pixels near the center of a cluster are shifted more than pixels on the edge.

**Best for:**
- Images with distinct color regions (portraits, product photos, landscapes)
- Matching a specific palette or color mood rather than overall statistics
- When you want each color region in the target treated independently

**Tradeoffs:**
- Slower than Reinhard (K-means fitting takes time)
- More `--n-colors` = finer matching but longer processing
- Can sometimes miss subtle mid-tone transitions

**Flag:**

| Flag | Default | Description |
|---|---|---|
| `--n-colors` | `8` | Number of palette colors to extract. Range: 4-16 |

---

### Wasserstein (Optimal Transport)

```bash
python gradia.py reference.jpg photo.jpg --method wasserstein
```

Uses **Sliced Wasserstein optimal transport with iterative advection** to find a non-linear mapping between the two color distributions. Each iteration picks a random direction in 3D LAB space, projects both color clouds onto that line, solves exact 1D optimal transport via rank-based quantile matching, and applies the displacement immediately before the next iteration.

**How it works:** In 1D, optimal transport has an exact closed-form solution - sort both distributions and match them by rank. Over many iterations in different random directions, the source distribution converges onto the reference. Unlike a one-shot accumulation, iterative advection lets each subsequent direction see a source that's already been partially transported, so the algorithm actually commits to matching the reference rather than averaging many tiny partial corrections. Does not require the POT library.

**Best for:**
- The most accurate and natural results
- Images where other methods produce color glitching
- Complex, multimodal color distributions (mixed lighting, high-contrast scenes)
- When channels are strongly correlated (skin tones, golden hour light, film stocks)

**Tradeoffs:**
- Slowest method - runtime scales with `--n-slices`
- More iterations = more accurate but slower

**Flags:**

| Flag | Default | Description |
|---|---|---|
| `--n-slices` | `200` | Number of advection iterations. More = more accurate, slower |
| `--sample-size` | `50000` | Max reference pixels sampled for quantile estimation |

---

## All Options

```
usage: gradia.py [-h]
                 [-o OUTPUT_DIR]
                 [--output-suffix OUTPUT_SUFFIX]
                 [-i INTENSITY]
                 [-m {reinhard,kantorovich,forgy,wasserstein}]
                 [--n-colors N_COLORS]
                 [--n-slices N_SLICES]
                 [--sample-size SAMPLE_SIZE]
                 [--visualize]
                 [--preview]
                 [-v] [-q]
                 reference targets [targets ...]
```

| Flag | Default | Description |
|---|---|---|
| `reference` | - | Reference image whose style you want to apply |
| `targets` | - | One or more images to grade |
| `-o, --output-dir` | Same as target | Directory to write output files |
| `--output-suffix` | `_graded_<reference>` | Suffix appended before file extension |
| `-i, --intensity` | `0.8` | Grade intensity: `0.0` = no change, `1.0` = full grade |
| `-m, --method` | `reinhard` | Grading method (see Methods above) |
| `--n-colors` | `8` | Palette colors for K-means (forgy method) |
| `--n-slices` | `200` | Advection iterations (wasserstein method) |
| `--sample-size` | `50000` | Max pixels sampled (kantorovich and wasserstein) |
| `--visualize` | off | Save a before/after histogram PNG alongside each output |
| `--preview` | off | Show side-by-side comparison before saving |
| `-v, --verbose` | off | Enable debug logging |
| `-q, --quiet` | off | Suppress all output except errors |

---

## Examples

**Basic grade**
```bash
python gradia.py film_reference.jpg my_photo.jpg
```

**Process a whole folder of photos**
```bash
python gradia.py reference.jpg photos/*.jpg -o ./graded/
```

**Subtle grade at lower intensity**
```bash
python gradia.py reference.jpg photo.jpg -i 0.5
```

**Linear transport grade**
```bash
python gradia.py reference.jpg photo.jpg --method kantorovich
```

**Most accurate method**
```bash
python gradia.py reference.jpg photo.jpg --method wasserstein
```

**Faster wasserstein with fewer iterations**
```bash
python gradia.py reference.jpg photo.jpg --method wasserstein --n-slices 50
```

**Match palette with more color clusters**
```bash
python gradia.py reference.jpg photo.jpg --method forgy --n-colors 12
```

**Save before/after histogram visualization alongside output**
```bash
python gradia.py reference.jpg photo.jpg --visualize
```

**Preview result before saving**
```bash
python gradia.py reference.jpg photo.jpg --preview
```

**Custom output folder and filename suffix**
```bash
python gradia.py portra400_ref.jpg shoot/*.jpg -o ./output/ --output-suffix _portra
```

**Quiet mode for scripting**
```bash
python gradia.py reference.jpg photo.jpg -q
```

**Verbose debug output**
```bash
python gradia.py reference.jpg photo.jpg -v
```

---

## Output Files

**Graded image**
Saved as `<target_name><suffix><extension>` - e.g. `photo_graded_reference.jpg`.

**Histogram visualization** (`--visualize`)
Saved as `<output_name>.histogram.png` alongside the result. Shows an overlaid histogram per color channel with three curves: reference, target before, and target after.

Useful for understanding what the grade actually did and for debugging unexpected results.

---

## Choosing the Right Method

| Situation | Recommended Method |
|---|---|
| First time, unsure what to use | `reinhard` |
| Subtle, natural-looking grade | `reinhard` |
| Images with very different content | `reinhard` |
| Global color shift, unimodal distributions | `kantorovich` |
| Portrait or product photography | `forgy` |
| Image has distinct color regions | `forgy` |
| Most accurate result, complex scene | `wasserstein` |
| Colors glitching with other methods | `wasserstein` |
| Skin tones, film stock, golden hour | `wasserstein` |

---

## Troubleshooting

**Colors look completely wrong / glitchy**
Switch to `--method wasserstein`. It handles non-linear color relationships and complex distributions that trip up simpler methods.

**Result looks too strong / washed out**
Lower the intensity: `-i 0.5` or `-i 0.3`. The default of `0.8` is intentionally strong.

**Result looks too subtle / barely any change**
Raise the intensity: `-i 1.0`. Also check that you haven't passed the images in the wrong order - the first argument is the reference, the second is the target.

**`kantorovich` method falling back to reinhard**
Install the POT library:
```bash
pip install POT
```

**Wasserstein is too slow**
Reduce iterations. Results are still good at lower counts:
```bash
python gradia.py ref.jpg photo.jpg --method wasserstein --n-slices 50
```

**Wasserstein results look blown out or washed out in places**
Too much total motion is pushing some pixels outside the valid color range, where they get clipped. Reduce the intensity with `-i 0.6` or lower, or reduce iterations with `--n-slices 100`.

**16-bit TIFF output looks correct but washed out in preview**
The `--preview` window scales 16-bit images to 8-bit for display only. The saved file is correct - check it in your photo editor rather than the preview window.

**Warning: Large resolution mismatch**
The grade will still work - the methods don't require matching resolutions. The warning is just a heads-up that results may look unexpected if one image is dramatically lower resolution than the other.

**Image not loading**
Make sure the path is correct and the file is a supported format. OpenCV supports JPEG, PNG, TIFF, BMP, and WebP. RAW files (`.CR2`, `.ARW`, etc.) are not supported directly - export to TIFF first.
