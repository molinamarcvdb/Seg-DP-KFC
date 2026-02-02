# Synthetic Data for Preconditioner Estimation

This document describes synthetic data generation strategies for estimating the AdaDPS preconditioner **without requiring real public medical data**.

---

## Table of Contents

1. [Overview](#overview)
2. [Theoretical Motivation](#theoretical-motivation)
3. [Noise Generation](#noise-generation)
4. [Synthetic Mask Strategies](#synthetic-mask-strategies)
   - [Vessel-like Structures](#vessel-like-structures)
   - [Blob-like Structures](#blob-like-structures)
   - [Cell-like Structures](#cell-like-structures)
   - [Organ-like Structures](#organ-like-structures)
   - [Generic Strategies](#generic-strategies)
5. [Recommended Configurations](#recommended-configurations)
6. [Implementation](#implementation)
7. [Experimental Design](#experimental-design)
8. [Research Questions](#research-questions)

---

## Overview

### The Problem

AdaDPS requires **public data with labels** to estimate the diagonal preconditioner E[g²]. In medical imaging:
- Public datasets may not exist for niche tasks
- Distribution shift between public and private data
- Licensing/access restrictions

### The Solution

Use **synthetic data** to estimate E[g²]:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Synthetic Preconditioning                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   SYNTHETIC DATA                   PRIVATE DATA                 │
│   (Generated)                      (Protected by DP)            │
│                                                                 │
│   ┌─────────────────┐              ┌─────────────────┐         │
│   │  Noise + Fake   │              │  Real medical   │         │
│   │  Supervision    │              │  images + masks │         │
│   └────────┬────────┘              └────────┬────────┘         │
│            │                                │                   │
│            │    Estimate E[g²]              │ Train with DP     │
│            │         │                      │                   │
│            ▼         ▼                      ▼                   │
│       ┌──────────────────────────────────────┐                 │
│       │  Preconditioner P = 1/√(E[g²] + λ)   │                 │
│       └──────────────────────────────────────┘                 │
│                                                                 │
│   Key insight: E[g²] captures gradient MAGNITUDES per layer,   │
│   which may transfer even from synthetic data!                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Why It Might Work

The preconditioner captures:
1. **Layer-wise gradient scales** - determined by architecture
2. **Input statistics** - noise can approximate natural image statistics
3. **Task structure** - synthetic masks can mimic segmentation targets

What it does NOT need:
- Exact semantic content
- Real anatomical structures
- Correct labels (just plausible supervision signal)

---

## Theoretical Motivation

### What Does E[g²] Capture?

For a neural network with parameters θ, the preconditioner diagonal is:

```
P_ii = 1 / √(E[g_i²] + λ)
```

Where g_i = ∂L/∂θ_i is the gradient for parameter i.

**Key observation**: E[g²] depends on:

| Factor | Contribution | Synthetic Approximation |
|--------|--------------|------------------------|
| Architecture | Layer sizes, activations | Exact (same model) |
| Input magnitude | Pixel value distribution | Noise with matched statistics |
| Spatial structure | Edges, textures | Filtered noise (pink, Gabor) |
| Target structure | Mask shapes, class balance | Synthetic shapes |

### Hypothesis

> If synthetic data matches the **statistical properties** of real data (not semantic content), the estimated E[g²] will transfer.

### Prior Evidence

From your previous work:
- Pink/brown noise + random supervision worked for [task]
- This suggests gradient statistics are robust to semantic content

---

## Noise Generation

### Noise Types

| Type | Spectrum | Visual Appearance | Use Case |
|------|----------|-------------------|----------|
| **White** | Flat | Grainy, random | Baseline |
| **Pink (1/f)** | 1/f | Natural textures | General medical |
| **Brown (1/f²)** | 1/f² | Smooth blobs | Soft tissue |
| **Perlin** | Coherent | Organic patterns | Anatomical structures |
| **Gabor** | Oriented | Texture-like | Fiber structures |
| **Fractal Brownian** | Multi-scale | Cloud-like | Heterogeneous tissue |

### Implementation

```python
import numpy as np
from scipy import ndimage
from noise import pnoise2  # pip install noise

def generate_white_noise(size):
    """Pure random noise - flat spectrum."""
    return np.random.randn(size, size)

def generate_pink_noise(size):
    """
    Pink noise (1/f spectrum).
    Mimics natural image statistics.
    """
    white = np.fft.fft2(np.random.randn(size, size))

    # Create 1/f filter
    freq_x = np.fft.fftfreq(size)
    freq_y = np.fft.fftfreq(size)
    fx, fy = np.meshgrid(freq_x, freq_y)
    freq = np.sqrt(fx**2 + fy**2)
    freq[0, 0] = 1  # Avoid division by zero

    # Apply 1/f filter
    pink = white / freq
    pink[0, 0] = 0  # Zero DC component

    result = np.real(np.fft.ifft2(pink))
    return (result - result.mean()) / result.std()

def generate_brown_noise(size):
    """
    Brown noise (1/f² spectrum).
    Smoother than pink, good for soft tissue.
    """
    white = np.fft.fft2(np.random.randn(size, size))

    freq_x = np.fft.fftfreq(size)
    freq_y = np.fft.fftfreq(size)
    fx, fy = np.meshgrid(freq_x, freq_y)
    freq = np.sqrt(fx**2 + fy**2)
    freq[0, 0] = 1

    brown = white / (freq ** 2)
    brown[0, 0] = 0

    result = np.real(np.fft.ifft2(brown))
    return (result - result.mean()) / result.std()

def generate_perlin_noise(size, scale=50, octaves=4):
    """
    Perlin noise - coherent, organic patterns.
    Good for anatomical variation.
    """
    noise = np.zeros((size, size))
    offset_x, offset_y = np.random.rand(2) * 1000

    for i in range(size):
        for j in range(size):
            noise[i, j] = pnoise2(
                (i + offset_x) / scale,
                (j + offset_y) / scale,
                octaves=octaves
            )

    return (noise - noise.mean()) / noise.std()

def generate_gabor_texture(size, frequency=0.1, theta=0):
    """
    Gabor-filtered noise - oriented textures.
    Good for fibrous structures.
    """
    from skimage.filters import gabor

    noise = np.random.randn(size, size)
    real, _ = gabor(noise, frequency=frequency, theta=theta)

    return (real - real.mean()) / real.std()

def generate_fractal_brownian(size, octaves=6, persistence=0.5):
    """
    Fractal Brownian motion - multi-scale patterns.
    Good for heterogeneous tissue.
    """
    noise = np.zeros((size, size))
    amplitude = 1.0
    frequency = 1.0

    for _ in range(octaves):
        noise += amplitude * generate_perlin_noise(size, scale=size/frequency)
        amplitude *= persistence
        frequency *= 2

    return (noise - noise.mean()) / noise.std()
```

### Visual Comparison

```
White Noise          Pink Noise           Brown Noise
┌──────────┐         ┌──────────┐         ┌──────────┐
│ ░▒░▓░▒▓░ │         │ ▒▒▓▓░░▒▒ │         │ ░░░▒▒▒▓▓ │
│ ▓░▒░▓▒░▓ │         │ ░░▒▒▓▓░░ │         │ ░░▒▒▒▓▓▓ │
│ ░▒▓░▒░▓▒ │         │ ▓▓░░▒▒▓▓ │         │ ▒▒▒▓▓▓░░ │
│ ▓░▒▓░▒░▓ │         │ ▒▒▓▓░░▒▒ │         │ ▓▓▓░░░▒▒ │
└──────────┘         └──────────┘         └──────────┘
High frequency       Balanced             Low frequency
(grainy)             (natural)            (smooth blobs)

Perlin Noise         Gabor Texture        Fractal Brownian
┌──────────┐         ┌──────────┐         ┌──────────┐
│ ░░▒▒▓▓▒▒ │         │ ░▒░▒░▒░▒ │         │ ░▒▓░▒▓▒░ │
│ ░▒▒▓▓▓▒░ │         │ ▒░▒░▒░▒░ │         │ ▒▓▓▒▓▓▓▒ │
│ ▒▓▓▓▓▒░░ │         │ ░▒░▒░▒░▒ │         │ ░▒▓░░▒▓░ │
│ ▓▓▓▒▒░░░ │         │ ▒░▒░▒░▒░ │         │ ▓▓▒▒▓▓▒▒ │
└──────────┘         └──────────┘         └──────────┘
Coherent blobs       Oriented stripes     Multi-scale
```

---

## Synthetic Mask Strategies

### Vessel-like Structures

For retinal vessels, angiography, airways, etc.

#### 1. Frangi Filter Response

```python
from skimage.filters import frangi, hessian
import numpy as np

def generate_frangi_vessels(size=256, noise_type='pink', threshold_percentile=85):
    """
    Generate vessel-like structures using Frangi vesselness filter.

    The Frangi filter detects tubular structures in the noise,
    creating realistic branching patterns.
    """
    # Generate base noise
    if noise_type == 'pink':
        noise = generate_pink_noise(size)
    elif noise_type == 'perlin':
        noise = generate_perlin_noise(size)
    else:
        noise = np.random.randn(size, size)

    # Apply Frangi filter (detects ridges/tubes)
    vessels = frangi(
        noise,
        sigmas=range(1, 5),      # Multi-scale
        black_ridges=False,      # Bright vessels
        alpha=0.5,               # Plate-like sensitivity
        beta=0.5,                # Blob-like sensitivity
        gamma=15                 # Background suppression
    )

    # Threshold to binary mask
    threshold = np.percentile(vessels, threshold_percentile)
    mask = (vessels > threshold).astype(np.float32)

    # Optional: morphological cleanup
    from scipy.ndimage import binary_dilation
    mask = binary_dilation(mask, iterations=1)

    # Create 3-channel image
    image = np.stack([noise] * 3, axis=-1)
    image = (image - image.min()) / (image.max() - image.min())

    return image, mask
```

#### 2. L-System Branching Trees

```python
import numpy as np
from PIL import Image, ImageDraw

def generate_lsystem_vessels(size=256, iterations=5, angle=25, length=10):
    """
    Generate vessel-like branching structures using L-systems.

    L-systems produce recursive branching patterns similar to
    vascular trees.
    """
    # L-system rules for branching
    axiom = "F"
    rules = {"F": "FF+[+F-F-F]-[-F+F+F]"}

    # Generate L-system string
    current = axiom
    for _ in range(iterations):
        next_str = ""
        for char in current:
            next_str += rules.get(char, char)
        current = next_str

    # Draw the L-system
    mask = Image.new('L', (size, size), 0)
    draw = ImageDraw.Draw(mask)

    # Turtle graphics interpretation
    x, y = size // 2, size - 20
    heading = -90  # Start pointing up
    stack = []

    for char in current:
        if char == 'F':
            # Draw forward
            rad = np.radians(heading)
            x_new = x + length * np.cos(rad)
            y_new = y + length * np.sin(rad)
            draw.line([(x, y), (x_new, y_new)], fill=255, width=2)
            x, y = x_new, y_new
        elif char == '+':
            heading += angle
        elif char == '-':
            heading -= angle
        elif char == '[':
            stack.append((x, y, heading))
        elif char == ']':
            x, y, heading = stack.pop()

    mask = np.array(mask) / 255.0

    # Generate matching noisy image
    noise = generate_pink_noise(size)
    image = np.stack([noise] * 3, axis=-1)
    image = (image - image.min()) / (image.max() - image.min())

    return image, mask
```

#### 3. Random Walk Curves

```python
def generate_random_walk_vessels(size=256, n_vessels=10, steps=200):
    """
    Generate vessel-like curves using random walks.

    Random walks create meandering paths similar to small vessels.
    """
    mask = np.zeros((size, size))

    for _ in range(n_vessels):
        # Random starting point near edge
        edge = np.random.choice(['top', 'bottom', 'left', 'right'])
        if edge == 'top':
            x, y = np.random.randint(0, size), 0
        elif edge == 'bottom':
            x, y = np.random.randint(0, size), size - 1
        elif edge == 'left':
            x, y = 0, np.random.randint(0, size)
        else:
            x, y = size - 1, np.random.randint(0, size)

        # Random walk with momentum (smoother paths)
        dx, dy = np.random.randn(2)

        for _ in range(steps):
            # Update direction with momentum
            dx = 0.9 * dx + 0.1 * np.random.randn()
            dy = 0.9 * dy + 0.1 * np.random.randn()

            # Normalize to fixed step size
            norm = np.sqrt(dx**2 + dy**2) + 1e-6
            dx, dy = dx / norm * 3, dy / norm * 3

            # Move
            x = np.clip(x + dx, 0, size - 1)
            y = np.clip(y + dy, 0, size - 1)

            # Draw with varying thickness
            thickness = np.random.randint(1, 4)
            rr, cc = disk((int(y), int(x)), thickness, shape=(size, size))
            mask[rr, cc] = 1

    noise = generate_pink_noise(size)
    image = np.stack([noise] * 3, axis=-1)
    image = (image - image.min()) / (image.max() - image.min())

    return image, mask
```

---

### Blob-like Structures

For lesions, tumors, nodules, polyps, etc.

#### 1. Gaussian Blobs

```python
def generate_gaussian_blobs(size=256, n_blobs=5, size_range=(10, 50)):
    """
    Generate soft elliptical blobs.
    Good for: lesions, nodules, round tumors.
    """
    mask = np.zeros((size, size))

    for _ in range(n_blobs):
        # Random center
        cx = np.random.randint(size_range[1], size - size_range[1])
        cy = np.random.randint(size_range[1], size - size_range[1])

        # Random ellipse parameters
        rx = np.random.randint(*size_range)
        ry = np.random.randint(*size_range)
        angle = np.random.rand() * 2 * np.pi

        # Create ellipse mask
        y, x = np.ogrid[:size, :size]
        x_rot = (x - cx) * np.cos(angle) + (y - cy) * np.sin(angle)
        y_rot = -(x - cx) * np.sin(angle) + (y - cy) * np.cos(angle)

        ellipse = (x_rot / rx) ** 2 + (y_rot / ry) ** 2 <= 1
        mask = np.maximum(mask, ellipse.astype(float))

    noise = generate_pink_noise(size)
    image = np.stack([noise] * 3, axis=-1)
    image = (image - image.min()) / (image.max() - image.min())

    return image, mask
```

#### 2. Perlin Blobs (Irregular Boundaries)

```python
def generate_perlin_blobs(size=256, n_blobs=3, base_radius=30):
    """
    Generate blobs with irregular Perlin-noise boundaries.
    Good for: tumors with irregular margins, polyps.
    """
    mask = np.zeros((size, size))

    for _ in range(n_blobs):
        # Random center
        cx = np.random.randint(base_radius + 20, size - base_radius - 20)
        cy = np.random.randint(base_radius + 20, size - base_radius - 20)

        # Create irregular boundary using polar coordinates
        n_points = 100
        angles = np.linspace(0, 2 * np.pi, n_points)

        # Perturb radius with Perlin noise
        radii = []
        offset = np.random.rand() * 1000
        for angle in angles:
            # Sample Perlin noise along circle
            perturbation = pnoise2(
                np.cos(angle) * 2 + offset,
                np.sin(angle) * 2 + offset,
                octaves=3
            )
            r = base_radius * (1 + 0.4 * perturbation)
            radii.append(r)

        # Convert to Cartesian and draw filled polygon
        points = []
        for angle, r in zip(angles, radii):
            x = cx + r * np.cos(angle)
            y = cy + r * np.sin(angle)
            points.append((x, y))

        # Fill polygon
        from skimage.draw import polygon
        rr, cc = polygon([p[1] for p in points], [p[0] for p in points], shape=(size, size))
        mask[rr, cc] = 1

    noise = generate_pink_noise(size)
    image = np.stack([noise] * 3, axis=-1)
    image = (image - image.min()) / (image.max() - image.min())

    return image, mask
```

#### 3. Metaballs (Smooth Merging Blobs)

```python
def generate_metaballs(size=256, n_balls=5, threshold=1.0):
    """
    Generate smooth merging blobs using metaball implicit surfaces.
    Good for: coalescing lesions, tumor clusters.
    """
    # Random ball centers and radii
    centers = np.random.rand(n_balls, 2) * (size - 100) + 50
    radii = np.random.rand(n_balls) * 20 + 15

    # Compute metaball field
    y, x = np.ogrid[:size, :size]
    field = np.zeros((size, size))

    for (cx, cy), r in zip(centers, radii):
        # Metaball contribution: r² / ((x-cx)² + (y-cy)²)
        dist_sq = (x - cx) ** 2 + (y - cy) ** 2 + 1e-6
        field += (r ** 2) / dist_sq

    # Threshold to get smooth merging shapes
    mask = (field > threshold).astype(float)

    noise = generate_pink_noise(size)
    image = np.stack([noise] * 3, axis=-1)
    image = (image - image.min()) / (image.max() - image.min())

    return image, mask
```

---

### Cell-like Structures

For histopathology, cell segmentation, nuclei detection.

#### 1. Voronoi Tessellation

```python
from scipy.spatial import Voronoi
from skimage.draw import polygon

def generate_voronoi_cells(size=256, n_cells=50, shrink=0.8):
    """
    Generate cell-like regions using Voronoi tessellation.
    Good for: cell segmentation, histopathology.
    """
    # Random cell centers
    points = np.random.rand(n_cells, 2) * size

    # Add boundary points to close Voronoi regions
    boundary = np.array([
        [-size, -size], [-size, size*2], [size*2, -size], [size*2, size*2]
    ])
    all_points = np.vstack([points, boundary])

    # Compute Voronoi
    vor = Voronoi(all_points)

    # Draw cells
    mask = np.zeros((size, size))

    for region_idx in vor.point_region[:n_cells]:  # Only original points
        region = vor.regions[region_idx]
        if -1 in region or len(region) == 0:
            continue

        # Get vertices
        vertices = vor.vertices[region]

        # Shrink toward centroid (creates cell boundaries)
        centroid = vertices.mean(axis=0)
        vertices = centroid + shrink * (vertices - centroid)

        # Clip to image bounds
        vertices = np.clip(vertices, 0, size - 1)

        # Draw filled polygon
        rr, cc = polygon(vertices[:, 1], vertices[:, 0], shape=(size, size))
        mask[rr, cc] = 1

    noise = generate_pink_noise(size)
    image = np.stack([noise] * 3, axis=-1)
    image = (image - image.min()) / (image.max() - image.min())

    return image, mask
```

#### 2. Packed Ellipses

```python
def generate_packed_ellipses(size=256, n_cells=30, max_attempts=1000):
    """
    Generate non-overlapping ellipses (cell nuclei).
    Good for: nuclei segmentation.
    """
    mask = np.zeros((size, size))
    cells = []  # Store (cx, cy, rx, ry) for collision detection

    for _ in range(n_cells):
        for attempt in range(max_attempts):
            # Random ellipse
            cx = np.random.randint(20, size - 20)
            cy = np.random.randint(20, size - 20)
            rx = np.random.randint(5, 15)
            ry = np.random.randint(5, 15)

            # Check collision with existing cells
            collision = False
            for (ocx, ocy, orx, ory) in cells:
                dist = np.sqrt((cx - ocx)**2 + (cy - ocy)**2)
                if dist < (rx + orx + ry + ory) / 2 + 5:
                    collision = True
                    break

            if not collision:
                cells.append((cx, cy, rx, ry))

                # Draw ellipse
                y, x = np.ogrid[:size, :size]
                ellipse = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1
                mask = np.maximum(mask, ellipse.astype(float))
                break

    noise = generate_pink_noise(size)
    image = np.stack([noise] * 3, axis=-1)
    image = (image - image.min()) / (image.max() - image.min())

    return image, mask
```

---

### Organ-like Structures

For large, smooth anatomical structures.

#### 1. Fourier Descriptor Shapes

```python
def generate_fourier_shapes(size=256, n_harmonics=5, base_radius=60):
    """
    Generate smooth random shapes using Fourier descriptors.
    Good for: organ boundaries, large smooth structures.
    """
    mask = np.zeros((size, size))

    # Center
    cx, cy = size // 2, size // 2

    # Generate shape using Fourier descriptors
    n_points = 200
    t = np.linspace(0, 2 * np.pi, n_points)

    # Random Fourier coefficients (low frequency = smooth)
    r = np.ones(n_points) * base_radius
    for k in range(1, n_harmonics + 1):
        a_k = np.random.randn() * 10 / k  # Amplitude decreases with frequency
        b_k = np.random.randn() * 10 / k
        r += a_k * np.cos(k * t) + b_k * np.sin(k * t)

    # Ensure positive radius
    r = np.maximum(r, 10)

    # Convert to Cartesian
    x = cx + r * np.cos(t)
    y = cy + r * np.sin(t)

    # Draw filled polygon
    from skimage.draw import polygon
    rr, cc = polygon(y, x, shape=(size, size))
    mask[rr, cc] = 1

    noise = generate_brown_noise(size)  # Smoother noise for organs
    image = np.stack([noise] * 3, axis=-1)
    image = (image - image.min()) / (image.max() - image.min())

    return image, mask
```

#### 2. Deformed Ellipsoids

```python
def generate_deformed_ellipse(size=256, perturbation_scale=0.2):
    """
    Generate ellipse with smooth deformations.
    Good for: simple organ approximations.
    """
    mask = np.zeros((size, size))

    # Random ellipse parameters
    cx, cy = size // 2 + np.random.randint(-30, 30), size // 2 + np.random.randint(-30, 30)
    rx, ry = np.random.randint(40, 80), np.random.randint(40, 80)

    # Create base ellipse coordinates
    n_points = 200
    t = np.linspace(0, 2 * np.pi, n_points)
    x = rx * np.cos(t)
    y = ry * np.sin(t)

    # Add smooth perturbation
    perturbation = generate_perlin_noise(1, scale=0.5)  # 1D Perlin
    # Actually, let's use simple low-freq noise
    noise_x = np.cumsum(np.random.randn(n_points)) * perturbation_scale
    noise_y = np.cumsum(np.random.randn(n_points)) * perturbation_scale

    # Smooth the noise
    from scipy.ndimage import gaussian_filter1d
    noise_x = gaussian_filter1d(noise_x, sigma=20)
    noise_y = gaussian_filter1d(noise_y, sigma=20)

    # Apply perturbation
    x = cx + x + noise_x * rx * perturbation_scale
    y = cy + y + noise_y * ry * perturbation_scale

    # Draw
    from skimage.draw import polygon
    rr, cc = polygon(y, x, shape=(size, size))
    mask[rr, cc] = 1

    noise = generate_brown_noise(size)
    image = np.stack([noise] * 3, axis=-1)
    image = (image - image.min()) / (image.max() - image.min())

    return image, mask
```

---

### Generic Strategies

Task-agnostic approaches.

#### 1. Pink Noise + Random Binary Supervision

```python
def generate_pink_random(size=256, threshold=0.5):
    """
    Simplest approach: threshold pink noise for random supervision.
    Good for: baseline experiments, any task.
    """
    noise = generate_pink_noise(size)

    # Random threshold to create mask
    mask = (noise > np.percentile(noise, threshold * 100)).astype(float)

    image = np.stack([noise] * 3, axis=-1)
    image = (image - image.min()) / (image.max() - image.min())

    return image, mask
```

#### 2. Multi-scale Blob Detection

```python
from skimage.feature import blob_dog

def generate_multiscale_blobs(size=256, n_blobs=20):
    """
    Generate blobs at multiple scales using DoG.
    Good for: unknown target morphology.
    """
    noise = generate_pink_noise(size)

    # Detect blobs at multiple scales
    blobs = blob_dog(noise, max_sigma=30, threshold=0.1)

    # Create mask from detected blobs
    mask = np.zeros((size, size))
    for y, x, r in blobs[:n_blobs]:
        rr, cc = disk((int(y), int(x)), int(r * np.sqrt(2)), shape=(size, size))
        mask[rr, cc] = 1

    image = np.stack([noise] * 3, axis=-1)
    image = (image - image.min()) / (image.max() - image.min())

    return image, mask
```

---

## Recommended Configurations

### Complete Experimental Setup

This section ties together **datasets from DATASETS.md** with recommended synthetic generators.

#### 2D Classification

| Private Dataset | Real Public Baseline | Synthetic Generators to Try | Priority |
|-----------------|---------------------|----------------------------|----------|
| **PathMNIST** (107k, histopath) | PatchCamelyon | `pink_random`, `voronoi_cells`, `packed_ellipses` | P1 |
| **DermaMNIST** (10k, skin) | Fitzpatrick17k | `pink_random`, `perlin_blobs`, `gaussian_blobs` | P2 |
| **BloodMNIST** (17k, blood) | BCCD | `pink_random`, `packed_ellipses`, `voronoi_cells` | P2 |
| **ChestMNIST** (112k, X-ray) | NIH ChestX-ray14 | `pink_random`, `fourier_shapes`, `perlin_blobs` | P1 |
| **OrganAMNIST** (58k, CT) | Medical Decathlon | `pink_random`, `fourier_shapes`, `deformed_ellipse` | P2 |

#### 2D Segmentation

| Private Dataset | Real Public Baseline | Synthetic Generators to Try | Priority |
|-----------------|---------------------|----------------------------|----------|
| **Kvasir-SEG** (1k, polyps) | CVC-ClinicDB | `perlin_blobs`, `metaballs`, `gaussian_blobs`, `pink_random` | P1 |
| **ISIC Segmentation** (2.5k, skin) | ISIC 2016 | `perlin_blobs`, `gaussian_blobs`, `metaballs` | P2 |
| **Retinal Vessels** (105, vessels) | CHASE_DB1 | `frangi`, `lsystem`, `random_walk`, `pink_random` | P1 |
| **Lung Segmentation** (800, X-ray) | JSRT | `fourier_shapes`, `deformed_ellipse`, `pink_random` | P2 |

#### 3D Classification

| Private Dataset | Real Public Baseline | Synthetic Generators to Try | Priority |
|-----------------|---------------------|----------------------------|----------|
| **OrganMNIST3D** (1.7k, CT) | Medical Decathlon | `pink_random_3d`, `fourier_shapes_3d` | P2 |
| **NoduleMNIST3D** (1.6k, nodules) | LUNA16 | `gaussian_blobs_3d`, `perlin_blobs_3d`, `pink_random_3d` | P1 |
| **VesselMNIST3D** (1.9k, MRA) | IXI MRA | `frangi_3d`, `random_walk_3d`, `pink_random_3d` | P2 |
| **ADNI** (2k, Alzheimer's) | OASIS | `pink_random_3d`, `fourier_shapes_3d` | P3 |

#### 3D Segmentation

| Private Dataset | Real Public Baseline | Synthetic Generators to Try | Priority |
|-----------------|---------------------|----------------------------|----------|
| **BraTS** (2k, brain tumor) | BraTS 2018 / IXI | `perlin_blobs_3d`, `metaballs_3d`, `pink_random_3d` | P1 |
| **AMOS** (600, abdominal) | BTCV | `fourier_shapes_3d`, `deformed_ellipsoid_3d`, `pink_random_3d` | P2 |
| **ACDC** (150, cardiac) | M&Ms | `fourier_shapes_3d`, `deformed_ellipsoid_3d` | P3 |

### Summary: Generator-Task Matching

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Synthetic Generator Selection Guide                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Target Morphology          Recommended Generators                     │
│   ─────────────────          ─────────────────────                     │
│                                                                         │
│   Tubular (vessels,          frangi, lsystem, random_walk              │
│   airways, nerves)                                                      │
│                                                                         │
│   Round blobs (lesions,      gaussian_blobs, perlin_blobs, metaballs   │
│   nodules, polyps)                                                      │
│                                                                         │
│   Irregular masses           perlin_blobs, metaballs                   │
│   (tumors)                                                              │
│                                                                         │
│   Cellular structures        voronoi_cells, packed_ellipses            │
│   (histopath, nuclei)                                                   │
│                                                                         │
│   Large smooth organs        fourier_shapes, deformed_ellipse          │
│   (liver, kidney, heart)                                                │
│                                                                         │
│   Unknown / baseline         pink_random, multiscale_blobs             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Noise Type Recommendations

| Modality | Recommended Noise | Rationale |
|----------|-------------------|-----------|
| **Histopathology** | Pink + Gabor | Tissue texture, stain patterns |
| **Dermoscopy** | Pink | Natural skin texture |
| **X-ray** | Pink / Brown | Bone + soft tissue contrast |
| **CT** | Brown | Smooth HU transitions |
| **MRI** | Perlin | Coherent intensity variations |
| **Fundus** | Pink | Retinal background texture |
| **Ultrasound** | Pink + speckle | Speckle noise characteristic |
| **Endoscopy** | Pink | Mucosal texture |

---

## Implementation

### Unified Generator Class

```python
import numpy as np
from typing import Tuple, Callable, Optional
from torch.utils.data import Dataset

class SyntheticDataset(Dataset):
    """
    PyTorch Dataset for synthetic image-mask pairs.

    Usage:
        dataset = SyntheticDataset(
            strategy='frangi',
            noise_type='pink',
            size=256,
            n_samples=1000,
            transform=get_training_augmentations()
        )
    """

    GENERATORS = {
        # Vessel-like
        'frangi': generate_frangi_vessels,
        'lsystem': generate_lsystem_vessels,
        'random_walk': generate_random_walk_vessels,

        # Blob-like
        'gaussian_blobs': generate_gaussian_blobs,
        'perlin_blobs': generate_perlin_blobs,
        'metaballs': generate_metaballs,

        # Cell-like
        'voronoi': generate_voronoi_cells,
        'packed_ellipses': generate_packed_ellipses,

        # Organ-like
        'fourier_shapes': generate_fourier_shapes,
        'deformed_ellipse': generate_deformed_ellipse,

        # Generic
        'pink_random': generate_pink_random,
        'multiscale_blobs': generate_multiscale_blobs,
    }

    def __init__(
        self,
        strategy: str = 'pink_random',
        noise_type: str = 'pink',
        size: int = 256,
        n_samples: int = 1000,
        transform: Optional[Callable] = None,
        seed: Optional[int] = None
    ):
        self.strategy = strategy
        self.noise_type = noise_type
        self.size = size
        self.n_samples = n_samples
        self.transform = transform

        if seed is not None:
            np.random.seed(seed)

        # Pre-generate all samples (or generate on-the-fly)
        self.samples = []
        for _ in range(n_samples):
            img, mask = self.GENERATORS[strategy](size=size)
            self.samples.append((img, mask))

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        image, mask = self.samples[idx]

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        return image, mask


def estimate_preconditioner_synthetic(
    model,
    strategy: str = 'pink_random',
    noise_type: str = 'pink',
    n_samples: int = 500,
    batch_size: int = 8,
    image_size: int = 256,
    device: str = 'cuda'
):
    """
    Estimate AdaDPS preconditioner using synthetic data.

    Args:
        model: Neural network
        strategy: Synthetic generation strategy
        noise_type: Type of noise for background
        n_samples: Number of synthetic samples
        batch_size: Batch size for estimation
        image_size: Image size
        device: Device to use

    Returns:
        Dictionary mapping parameter names to E[g²] estimates
    """
    from torch.utils.data import DataLoader
    import torch
    import torch.nn as nn

    # Create synthetic dataset
    dataset = SyntheticDataset(
        strategy=strategy,
        noise_type=noise_type,
        size=image_size,
        n_samples=n_samples
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize accumulators
    grad_sq_sum = {name: torch.zeros_like(param).to(device)
                   for name, param in model.named_parameters()}
    count = 0

    # Loss function
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    model.to(device)

    for images, masks in loader:
        images = images.to(device).float().permute(0, 3, 1, 2)  # NHWC -> NCHW
        masks = masks.to(device).float().unsqueeze(1)

        # Forward
        outputs = model(images)
        loss = criterion(outputs, masks)

        # Backward
        loss.backward()

        # Accumulate squared gradients
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_sq_sum[name] += param.grad ** 2

        model.zero_grad()
        count += 1

    # Compute E[g²]
    E_g_sq = {name: g_sq / count for name, g_sq in grad_sq_sum.items()}

    return E_g_sq
```

### Configuration File Addition

Add to `configs/default.yaml`:

```yaml
# Preconditioning settings
preconditioning:
  type: "adadps_synthetic"  # Options: none, adadps_public, adadps_synthetic, adadps_oracle

  # For adadps_synthetic
  synthetic:
    strategy: "frangi"       # Generator strategy
    noise_type: "pink"       # Background noise type
    n_samples: 500           # Number of synthetic samples

  estimation_steps: 50       # For adadps_public
  damping: 0.0001
```

---

## Experimental Design

### Main Experiments

| ID | Private Data | Preconditioner Source | Synthetic Strategy | Purpose |
|----|--------------|----------------------|-------------------|---------|
| E1 | PathMNIST | None | - | Baseline DP-SGD |
| E2 | PathMNIST | Real (PCam) | - | Real public baseline |
| E3 | PathMNIST | Synthetic | pink_random | Generic synthetic |
| E4 | PathMNIST | Synthetic | packed_ellipses | Task-matched synthetic |

| ID | Private Data | Preconditioner Source | Synthetic Strategy | Purpose |
|----|--------------|----------------------|-------------------|---------|
| E5 | Kvasir-SEG | None | - | Baseline DP-SGD |
| E6 | Kvasir-SEG | Real (CVC-ClinicDB) | - | Real public baseline |
| E7 | Kvasir-SEG | Synthetic | pink_random | Generic synthetic |
| E8 | Kvasir-SEG | Synthetic | perlin_blobs | Task-matched synthetic |

| ID | Private Data | Preconditioner Source | Synthetic Strategy | Purpose |
|----|--------------|----------------------|-------------------|---------|
| E9 | BraTS | None | - | Baseline DP-SGD |
| E10 | BraTS | Real (BraTS 2018) | - | Real public (exact) |
| E11 | BraTS | Real (IXI) | - | Real public (shifted) |
| E12 | BraTS | Synthetic | pink_random_3d | Generic synthetic |
| E13 | BraTS | Synthetic | perlin_blobs_3d | Task-matched synthetic |

### Ablation Studies

| ID | Study | Variables |
|----|-------|-----------|
| A1 | Synthetic strategy | {pink_random, gaussian_blobs, perlin_blobs, frangi} |
| A2 | Noise type | {white, pink, brown, perlin} |
| A3 | Number of synthetic samples | {100, 500, 1000, 5000} |
| A4 | Synthetic vs real | Synthetic best vs real public |

---

## Research Questions

| RQ | Question | Experiment IDs |
|----|----------|---------------|
| **RQ1** | Does synthetic preconditioning improve over no preconditioning? | E1 vs E3, E5 vs E7, E9 vs E12 |
| **RQ2** | Does task-matched synthetic outperform generic synthetic? | E3 vs E4, E7 vs E8, E12 vs E13 |
| **RQ3** | How does synthetic compare to real public data? | E2 vs E3/E4, E6 vs E7/E8, E10 vs E12/E13 |
| **RQ4** | Which noise type works best? | Ablation A2 |
| **RQ5** | How many synthetic samples are needed? | Ablation A3 |
| **RQ6** | Does synthetic work for 3D tasks? | E9-E13 |

---

## Expected Outcomes

### Hypothesis

1. **Synthetic > No preconditioner**: Even random supervision provides useful gradient statistics
2. **Task-matched > Generic**: Morphology-aware generators improve transfer
3. **Real ≈ Synthetic (for some tasks)**: When domain shift is high, synthetic may match real
4. **Frangi works for vessels**: Filter-based generators capture task-specific statistics

### Potential Paper Contribution

> "We demonstrate that synthetic data with task-matched morphological priors can replace real public datasets for DP preconditioner estimation, achieving comparable utility while eliminating the need for auxiliary medical data."

---

## References

1. **Pink/Brown Noise**: Voss & Clarke, "1/f noise in music and speech", Nature 1975
2. **Perlin Noise**: Perlin, "An Image Synthesizer", SIGGRAPH 1985
3. **Frangi Filter**: Frangi et al., "Multiscale vessel enhancement filtering", MICCAI 1998
4. **L-Systems**: Lindenmayer, "Mathematical models for cellular interactions", J. Theor. Biol. 1968
5. **Metaballs**: Blinn, "A Generalization of Algebraic Surface Drawing", ACM TOG 1982
6. **Voronoi Tessellation**: Aurenhammer, "Voronoi diagrams—a survey", ACM Computing Surveys 1991

---

## Changelog

| Date | Version | Changes |
|------|---------|---------|
| 2026-01-30 | 1.0 | Initial documentation |

