"""
=============================================================
  Point Source Initial Condition — Three Plots
=============================================================
  Problem Setup:
    PDE    : 2D Advection-Diffusion-Reaction
    Domain : x ∈ [-2, 2],  z ∈ [-1, 1]
    IC     : c(x,z,0) = (1/2πσ²) * exp(-(x² + (z-z0)²) / 2σ²)
    σ      = 0.025   (narrow Gaussian ≈ Dirac delta)
    z0     = 0.5     (source position in z)
    Source : at (x=0, z=0.5)

  Three Figures Produced:
    Fig 1 — Full domain + Zoomed view  (like Zong et al. Fig 1b)
    Fig 2 — 3D surface view
    Fig 3 — Cross-sections through source

  Requirements:
    pip install numpy matplotlib
=============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # needed for 3D plot

# =============================================================
#  PARAMETERS  —  change only here if needed
# =============================================================
SIGMA  = 0.025      # Gaussian std  (mimics Dirac delta)
Z0     = 0.5        # source z-location
X_SRC  = 0.0        # source x-location
X_MIN, X_MAX = -2.0,  2.0    # x domain
Z_MIN, Z_MAX = -1.0,  1.0    # z domain
PEAK   = 1.0 / (2.0 * np.pi * SIGMA**2)   # theoretical peak ≈ 254.6

print("=" * 55)
print("  Point Source IC — Key Numbers")
print("=" * 55)
print(f"  σ (sigma)       = {SIGMA}")
print(f"  Source location = (x={X_SRC}, z={Z0})")
print(f"  Peak value      = 1/(2πσ²) = {PEAK:.4f}")
print(f"  Domain          : x∈[{X_MIN},{X_MAX}], z∈[{Z_MIN},{Z_MAX}]")
print("=" * 55)


# =============================================================
#  HELPER FUNCTION
# =============================================================
def gaussian_IC(X, Z, sigma=SIGMA, x_src=X_SRC, z_src=Z0):
    """
    Gaussian approximation of Dirac delta IC.
    c(x,z,0) = (1/2πσ²) * exp(-(r²)/(2σ²))
    where r² = (x - x_src)² + (z - z_src)²
    Integrates to 1 over the full plane (unit mass).
    """
    r2 = (X - x_src)**2 + (Z - z_src)**2
    return np.exp(-r2 / (2.0 * sigma**2)) / (2.0 * np.pi * sigma**2)


# =============================================================
#  FIGURE 1 — Full domain + Zoomed panel  (like paper Fig 1b)
# =============================================================
print("\nGenerating Figure 1 — Main Panel ...")

# Fine grid to resolve the sharp spike
Nx_full = 1000
Nz_full = 500
x_full  = np.linspace(X_MIN, X_MAX, Nx_full)
z_full  = np.linspace(Z_MIN, Z_MAX, Nz_full)
X_full, Z_full = np.meshgrid(x_full, z_full)

C0_full = gaussian_IC(X_full, Z_full)

# Quick sanity checks
grid_peak  = C0_full.max()
total_mass = C0_full.sum() * (X_MAX - X_MIN) / Nx_full * (Z_MAX - Z_MIN) / Nz_full

print(f"  Grid peak value   = {grid_peak:.4f}  (theory = {PEAK:.4f})")
print(f"  Total mass ∬c dxdz ≈ {total_mass:.4f}  (should be ≈ 1.0)")

fig1, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# ── Left: full domain view ────────────────────────────────
ax = axes[0]
pcm = ax.pcolormesh(X_full, Z_full, C0_full,
                    cmap='hot',
                    shading='auto',
                    vmin=0,
                    vmax=PEAK)
cbar = plt.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label(r'$c(x,z,0)$', fontsize=12)
cbar.ax.tick_params(labelsize=10)

# Mark walls
ax.axhline(y=Z_MIN, color='cyan', lw=1.5, ls='--', alpha=0.8,
           label=r'Walls $z = \pm1$')
ax.axhline(y=Z_MAX, color='cyan', lw=1.5, ls='--', alpha=0.8)

# Mark source
ax.plot(X_SRC, Z0, 'b+', markersize=14, markeredgewidth=2.5,
        label=fr'Source $(0,\ {Z0})$')

ax.set_xlim(X_MIN, X_MAX)
ax.set_ylim(Z_MIN, Z_MAX)
ax.set_xlabel(r'$x$', fontsize=14)
ax.set_ylabel(r'$z$', fontsize=14)
ax.set_title('Full domain view', fontsize=13, fontweight='bold')
ax.legend(fontsize=10, loc='lower right')
ax.tick_params(labelsize=11)

# ── Right: zoomed around source ───────────────────────────
ax2 = axes[1]
zoom = 0.15    # show ±zoom around source
mask_x = (x_full >= X_SRC - zoom) & (x_full <= X_SRC + zoom)
mask_z = (z_full >= Z0    - zoom) & (z_full <= Z0    + zoom)
x_zoom = x_full[mask_x]
z_zoom = z_full[mask_z]
X_zoom, Z_zoom = np.meshgrid(x_zoom, z_zoom)
C0_zoom = gaussian_IC(X_zoom, Z_zoom)

pcm2 = ax2.pcolormesh(X_zoom, Z_zoom, C0_zoom,
                      cmap='hot',
                      shading='auto',
                      vmin=0,
                      vmax=PEAK)
cbar2 = plt.colorbar(pcm2, ax=ax2, fraction=0.046, pad=0.04)
cbar2.set_label(r'$c(x,z,0)$', fontsize=12)
cbar2.ax.tick_params(labelsize=10)

ax2.plot(X_SRC, Z0, 'b+', markersize=14, markeredgewidth=2.5,
         label=fr'Source $(0,\ {Z0})$')
ax2.set_xlabel(r'$x$', fontsize=14)
ax2.set_ylabel(r'$z$', fontsize=14)
ax2.set_title(fr'Zoomed: $\pm{zoom}$ around source', fontsize=13,
              fontweight='bold')
ax2.legend(fontsize=10)
ax2.tick_params(labelsize=11)

fig1.suptitle(
    r'Point Source IC:  $c(x,z,0) = \frac{1}{2\pi\sigma^2}'
    r'\exp\!\left(-\frac{x^2+(z-z_0)^2}{2\sigma^2}\right)$'
    fr'     $\sigma={SIGMA}$,  peak $\approx {PEAK:.1f}$',
    fontsize=13, y=1.02
)
plt.tight_layout()
fig1.savefig('Figure1_point_source_IC_panel.png', dpi=180, bbox_inches='tight')
plt.close(fig1)
print("  Saved: Figure1_point_source_IC_panel.png")


# =============================================================
#  FIGURE 2 — 3D Surface view
# =============================================================
print("\nGenerating Figure 2 — 3D Surface ...")

# Coarser grid is enough for 3D (finer = slower rendering)
Nx_3d = 400
Nz_3d = 200
x_3d  = np.linspace(X_MIN, X_MAX, Nx_3d)
z_3d  = np.linspace(Z_MIN, Z_MAX, Nz_3d)
X_3d, Z_3d = np.meshgrid(x_3d, z_3d)
C0_3d = gaussian_IC(X_3d, Z_3d)

fig2 = plt.figure(figsize=(12, 7))
ax3  = fig2.add_subplot(111, projection='3d')

surf = ax3.plot_surface(X_3d, Z_3d, C0_3d,
                        cmap='viridis',
                        linewidth=0,
                        antialiased=True,
                        alpha=0.93)

fig2.colorbar(surf, ax=ax3, fraction=0.025, pad=0.1,
              label=r'$c(x,z,0)$')

ax3.set_xlabel(r'$x$',         fontsize=13, labelpad=10)
ax3.set_ylabel(r'$z$',         fontsize=13, labelpad=10)
ax3.set_zlabel(r'$c(x,z,0)$',  fontsize=13, labelpad=10)
ax3.set_title(
    fr'3D view — Point Source IC   $\sigma={SIGMA}$,  source at $(0,\ {Z0})$'
    f'\nPeak $= 1/(2\pi\sigma^2) \\approx {PEAK:.1f}$',
    fontsize=12, pad=15
)
ax3.view_init(elev=35, azim=-60)

# Clean pane backgrounds
for pane in [ax3.xaxis.pane, ax3.yaxis.pane, ax3.zaxis.pane]:
    pane.fill = False

plt.tight_layout()
fig2.savefig('Figure2_point_source_IC_3D.png', dpi=180, bbox_inches='tight')
plt.close(fig2)
print("  Saved: Figure2_point_source_IC_3D.png")


# =============================================================
#  FIGURE 3 — Cross-sections through source
# =============================================================
print("\nGenerating Figure 3 — Cross-sections ...")

x_1d = np.linspace(X_MIN, X_MAX, 10000)   # fine for smooth curves
z_1d = np.linspace(Z_MIN, Z_MAX, 10000)

# x-profile at z = z0  →  only z-z0 term in r², z-z0=0
c_x_profile = gaussian_IC(x_1d, np.full_like(x_1d, Z0))

# z-profile at x = 0  →  only (z-z0)² term
c_z_profile = gaussian_IC(np.zeros_like(z_1d), z_1d)

fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5.5))

# ── Left: x-profile ──────────────────────────────────────
ax = axes3[0]
ax.plot(x_1d, c_x_profile, color='royalblue', lw=2.2,
        label=r'$c(x,\ z_0,\ 0)$')
ax.axvline(X_SRC, color='red', ls='--', lw=1.5, alpha=0.8,
           label=fr'Source $x={X_SRC}$')

# Mark 3-sigma width
ax.axvline( 3*SIGMA, color='green', ls=':', lw=1.3, alpha=0.7,
            label=fr'$\pm 3\sigma = \pm{3*SIGMA}$')
ax.axvline(-3*SIGMA, color='green', ls=':', lw=1.3, alpha=0.7)

ax.set_xlabel(r'$x$', fontsize=14)
ax.set_ylabel(r'$c(x,\ z_0,\ 0)$', fontsize=14)
ax.set_title(fr'x-profile at $z = z_0 = {Z0}$', fontsize=13,
             fontweight='bold')
ax.set_xlim(-0.25, 0.25)    # zoom to see spike
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
ax.tick_params(labelsize=11)

# Annotate peak
ax.annotate(fr'Peak $\approx {PEAK:.1f}$',
            xy=(0, PEAK),
            xytext=(0.05, PEAK * 0.85),
            fontsize=11,
            arrowprops=dict(arrowstyle='->', color='black'),
            color='darkblue')

# ── Right: z-profile ─────────────────────────────────────
ax = axes3[1]
ax.plot(c_z_profile, z_1d, color='darkorange', lw=2.2,
        label=r'$c(0,\ z,\ 0)$')

# Wall lines
ax.axhline(Z_MIN, color='gray', ls='--', lw=1.5, alpha=0.7,
           label=fr'Walls $z = \pm1$')
ax.axhline(Z_MAX, color='gray', ls='--', lw=1.5, alpha=0.7)

# Source line
ax.axhline(Z0, color='red', ls='--', lw=1.5, alpha=0.8,
           label=fr'Source $z = {Z0}$')

# Distance markers
dist_upper = Z_MAX - Z0
dist_lower = Z0 - Z_MIN
ax.annotate('', xy=(PEAK * 0.3, Z_MAX), xytext=(PEAK * 0.3, Z0),
            arrowprops=dict(arrowstyle='<->', color='purple', lw=1.5))
ax.text(PEAK * 0.33, (Z0 + Z_MAX) / 2,
        fr'd={dist_upper:.1f}', color='purple', fontsize=10, va='center')

ax.annotate('', xy=(PEAK * 0.3, Z_MIN), xytext=(PEAK * 0.3, Z0),
            arrowprops=dict(arrowstyle='<->', color='teal', lw=1.5))
ax.text(PEAK * 0.33, (Z0 + Z_MIN) / 2,
        fr'd={dist_lower:.1f}', color='teal', fontsize=10, va='center')

ax.set_xlabel(r'$c(0,\ z,\ 0)$', fontsize=14)
ax.set_ylabel(r'$z$', fontsize=14)
ax.set_title(r'z-profile at $x = 0$', fontsize=13, fontweight='bold')
ax.set_ylim(Z_MIN - 0.05, Z_MAX + 0.05)
ax.legend(fontsize=11, loc='center right')
ax.grid(alpha=0.3)
ax.tick_params(labelsize=11)

fig3.suptitle(
    r'Cross-sections of Point Source IC through $(x{=}0,\ z{=}0.5)$'
    fr'     [$\sigma={SIGMA}$]',
    fontsize=13, y=1.01
)
plt.tight_layout()
fig3.savefig('Figure3_point_source_IC_crosssections.png',
             dpi=180, bbox_inches='tight')
plt.close(fig3)
print("  Saved: Figure3_point_source_IC_crosssections.png")


# =============================================================
#  SUMMARY
# =============================================================
print("\n" + "=" * 55)
print("  ✓  All 3 figures generated successfully.")
print("=" * 55)
print(f"\n  KEY RESULTS:")
print(f"  σ                  = {SIGMA}")
print(f"  Source location    = ({X_SRC}, {Z0})")
print(f"  Peak  c(0,z0,0)    = {PEAK:.4f}")
print(f"  Total mass ∬c dxdz ≈ {total_mass:.4f}  (≈ 1 confirmed)")
print(f"  Distance to z=+1   = {Z_MAX - Z0:.1f}  (upper wall)")
print(f"  Distance to z=-1   = {Z0 - Z_MIN:.1f}  (lower wall)")
print(f"\n  Figures saved:")
print(f"  → Figure1_point_source_IC_panel.png")
print(f"  → Figure2_point_source_IC_3D.png")
print(f"  → Figure3_point_source_IC_crosssections.png")
