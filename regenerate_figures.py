#!/usr/bin/env python3
"""
regenerate_figures.py
=====================

Regenerates every simulation-derived panel in the manuscript from the
reaction-diffusion models in the STAR Methods (nondimensional, on a periodic
1D domain of length 1 representing the perimeter of a target cell, i = 1,
facing a pseudo-cell, i = 2).

Panels and the commands that produce them
-----------------------------------------
All manuscript figures use N = 800 and tmax = 2e4 (about 18 h). The default
--N is 200, which is for quick checks only, so always pass --N 800.

    Figure 2H        --panels 2H     polarity vs puncta, coloured by D_a
    Figure 2I-L      --panels 2IL    example profiles at four values of D_a
    Figure 6B-D      --panels 6BD    example profiles at low/moderate/high D_CKIe
    Figure 6E-M      --panels 6EM    example profiles, genotypes and variants
    Figure 6R,S      --panels 6NO    polarity and puncta across conditions B-M
                                     (labelled 6N,O in the script's output)
    Figure S3C-J     --sensitivity MD     one-parameter sweeps, MD model
    Figure S6A-G     --sensitivity CKI    one-parameter sweeps, CKIe model
    Figure S6H       --sigma-test         polarity vs puncta as sigma varies
    Figure S6I,J     --pk-dose            Pk dose response, CKIe model
    Figure S7A       --time-course        polarity and puncta against time
    Figure S7B,C     --mesh-test --mesh-N 800 1600   paired refinement check
    Figure S7D       --panels width       punctum width vs D_a
    Figure S7E       --width-sigma ...    punctum width vs sigma

Example:

    python regenerate_figures.py --panels all --N 800 --reps 35 --jobs 8
    python regenerate_figures.py --smoke                  # 2 min sanity check

Other modes: --ic-test (white-noise vs correlated initial conditions),
--timescale (residence half-life at puncta vs FRAP), --fractions (bound
fractions vs FRAP), --sweep NAME --sweep-values ... (any numeric entry in
BASE). Parameter overrides: --D_a, --D_CKIe, --spread-length, --k_off_W,
--k_on_W, --mu, --nu, --k_basal, --K, --hill-q, --sigma, --cki-recruit.
Runs that override a parameter get a suffix on their figure filenames, so
they cannot overwrite a production figure.

Caching
-------
Results are cached in --outdir as .npz files tagged by N, reps, tmax and a
hash of the parameter set; a rerun with the same settings replots without
recomputing. Pass --force to recompute. Any Figure 6 panel computes all
twelve conditions on first call, since they share one cache file.

Runtime and memory
------------------
One CKIe simulation takes roughly 50 s at N = 200 and several minutes at
N = 800. Figure 6R,S at 35 repeats is 420 simulations, so use --jobs. scipy
builds a sparse numerical Jacobian in each worker, and memory grows with
jobs x N: at N = 1600 use --jobs 2 or 3 on a 16 GB machine. On Windows,
"DLL load failed ... paging file is too small" at start-up means virtual
memory ran out; results remain valid, but reduce --jobs. Each process is
limited to one maths-library thread.

Parameterisation
----------------
Time unit T0 = 3.3 s, from the Dsh/Pk FRAP fast phases (18.6 and 45.9 s)
matched to their unmodulated off-rate k_off_VP = 0.07. Constrained by FRAP:

  k_off_W = 0.065   CKIe slow phase (35.5 s)
  k_on_W  = 0.0065  CKIe recruitment rate; 23% of CKIe complex-associated,
                    against about 30% by FRAP once bleaching losses are
                    allowed for (see --fractions)
  nu      = 0.005   chosen so that core complexes at CKIe-model puncta turn
                    over with the measured core slow phases (see --timescale)

Chosen: D_CKIe = 0.00218 (spread length sqrt(D_CKIe/k_off_W) = 0.183 of
the perimeter, about 3.7 um; 0.26 um^2/s), mu = 0.2, sigma_ic, the feedback
strengths tau, eta, gamma, xi, the exponents n and q, K, k_bind, k_unbind.

At the production settings (N = 800, 35 seeds, D_a = 1e-7) the CKIe model
gives median polarity 0.28 with 3.0 puncta per cell, against 0.16 with 6.0
in the mutual destabilisation (MD) model.

Notation
--------
As in the manuscript. Superscript f is the free density and superscript a
the Fmi-associated (Fz, Stbm) or complex-associated (Dsh, Pk, CKIe) density;
the trailing digit is the cell index. In code: Ff1, Fa1, Sf1, Sa1, Vf1, Va1,
Pf1, Pa1, Wf1, Wa1, and trans-complexes C1, C2. Rates are k_on, k_off_FS,
k_off_VP, k_off_W, k_basal, k_bind and k_unbind; diffusion coefficients D_f
(free), D_a (Fmi-associated intermediates) and D_CKIe (complex-associated
CKIe). Feedback strengths keep their Greek names (tau, eta, gamma, xi, nu,
mu), with feedback exponent n, recruitment exponent q and half-saturation K.

Initial conditions
------------------
Fmi-associated Fz and Stbm start as random fluctuations smoothed to a fixed
correlation length sigma_ic (0.01 of the perimeter, 200 nm). Without
smoothing, the correlation length is the grid spacing and finer grids
nucleate more puncta (--ic-test). The field is drawn on a reference grid of
N_REF points and interpolated, so a given seed gives the same physical
initial condition at every N and refinement comparisons are paired. Punctum
number and width are inherited from sigma_ic (FWHM about 3.8 sigma), not
selected by the dynamics.

Caveats
-------
1. Not a steady state. Puncta merge slowly and the system is still evolving
   at t = 1e5 (about 90 h), so every result is reported at t = 2e4. Punctum
   number depends on this choice (CKIe model: 2.8 at 18 h); polarity much
   less so (--time-course).
2. Resolution. Polarity rises by about 0.02-0.03 per doubling of N from 200
   to 1600 in both models, so absolute polarity is not converged at N = 800.
   The difference between the models is stable under refinement, as are
   punctum numbers (--mesh-test).
3. Punctum counting. A punctum is a local maximum in bound Fz (intermediate
   plus trans-complexed) with height above 0.1 and prominence above 2% of
   the mean density; the count is halved because each punctum lies on a
   junction shared with the pseudo-cell. Without the prominence test,
   numerical ripples on a uniform profile are counted as puncta.
4. Punctum width. FWHM scales weakly with D_a (as about D_a^0.16) and in
   proportion to sigma_ic, so it reflects the assumed initial condition and
   is not a model prediction.

Requires numpy, scipy, matplotlib.
"""

import argparse
import os

# One maths-library thread per process. Parallelism here comes from --jobs
# (separate processes); letting each process also start its own pool of
# MKL/OpenBLAS threads oversubscribes the cores and, on Windows, each extra
# thread pool adds to the committed memory charged against the paging file.
# Must be set before numpy/scipy are imported. setdefault leaves any value
# you have set yourself untouched.
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np  # noqa: E402
from scipy.integrate import solve_ivp  # noqa: E402
from scipy.signal import find_peaks, peak_widths  # noqa: E402
from scipy.ndimage import gaussian_filter1d  # noqa: E402
from scipy.sparse import lil_matrix  # noqa: E402

# ======================================================================
# parameters
# ======================================================================
# Reference grid on which initial conditions are generated (see
# correlated_noise). Must be at least the finest N you run.
N_REF = 1600

# CKIe parameters are set from FRAP (see Parameterisation in the docstring).
# Long-term FRAP (Figure 2F), slow- and fast-phase half-lives in seconds.
FRAP_SLOW_CORE = dict(Fmi=1381, Fz=1129, Stbm=1694, Dsh=848, Pk=1944, Dgo=1247)
FRAP_FAST_DSH_PK = (18.6, 45.9)
FRAP_SLOW_CKIE = 35.5
K_OFF_W = 0.065          # from the measured CKIe slow-phase half-life, ~36 s
CKI_SPREAD_LENGTH = 0.183  # L = sqrt(D_CKIe / k_off_W), ~3.7 um of perimeter

BASE = dict(
    D_f=0.1,          # free species
    D_a=1e-7,         # Fmi-associated intermediates (Fz, Stbm)
    D_CKIe=CKI_SPREAD_LENGTH**2 * K_OFF_W,   # = 0.00218; complex-associated CKIe
    k_on=0.07,         # forward basal rate
    k_on_W=0.0065,     # CKIe recruitment rate by Dsh; gives 23% of CKIe
                       # complex-associated, against ~30% by FRAP
    k_off_FS=0.2,       # backward basal rate, Fz and Stbm
    k_off_VP=0.07,      # backward basal rate, Dsh and Pk
    k_off_W=K_OFF_W,    # backward basal rate, CKIe (FRAP-derived)
    k_basal=0.01,       # basal recruitment
    k_bind=1.0,       # trans-complex formation
    k_unbind=0.1,       # trans-complex dissociation
    tau=0.3,         # Dsh stabilises Fz
    eta=0.3,         # Pk stabilises Stbm
    gamma=0.2,       # Pk destabilises Fz, Dsh-dependent
    xi=0.2,          # Dsh destabilises Stbm       (mutual destabilisation model)
    mu=0.2,          # CKIe destabilises Stbm      (CKIe model); set to give
                     # 2-3 puncta per cell at the FRAP-derived k_off_W
    nu=0.005,        # CKIe stabilises Dsh         (CKIe model); fitted so
                     # that core complexes at puncta turn over with the
                     # measured slow-phase half-life (see --timescale)
    n=3,             # Hill exponent, feedback terms
    K=1.0,           # half-saturation of G(y); chosen, swept in Figure S6A
    hill_q=2,        # exponent of the recruitment function G(y) for Dsh and Pk
                     # (and for CKIe when cki_recruit='hill'). q=2 is the
                     # cooperative form used throughout; q=1 is the
                     # non-cooperative Michaelis-Menten form, which tests
                     # whether cooperative recruitment is needed at all
                     # (distinct from n, the exponent of the feedback terms).
    init_free=4.0,   # initial free Fz/Stbm/Dsh/Pk
    init_W=7.0,      # initial free CKIe
    init_inter=0.1,  # intermediates ~ U[0, init_inter]
    cki_recruit='mass_action',   # 'mass_action' (default, linear in complex-
                     # associated Dsh) or 'hill' (cooperative). Dsh and Pk
                     # recruitment is always cooperative.
    uniform_W=False, # True -> CKIe recruited uniformly (distal bias removed)
    sigma_ic=0.01,   # correlation length of initial fluctuations, in units of
                     # cell perimeter (0.01 = 200 nm). 0 gives grid-dependent
                     # white noise (see --ic-test).
    n_ref=N_REF,     # reference grid for generating the initial condition, so
                     # that one seed gives the same physical field at every N.
                     # 0 reverts to drawing the field on the simulation grid.
)

# Snapshot of the production parameter set, taken at import. Any diagnostic run
# that changes a parameter (--sigma, --hill-q, --D_a ...) gets a suffix on its
# figure filenames from _fig_suffix(), so it cannot silently overwrite the
# production figure of the same panel.
_DEFAULTS = dict(BASE)

_SHORT = {'sigma_ic': 'sig', 'hill_q': 'q', 'D_a': 'Da', 'D_CKIe': 'DW',
          'k_on_W': 'konW', 'k_off_W': 'koffW', 'k_basal': 'kb',
          'cki_recruit': 'rec', 'uniform_W': 'unifW', 'init_W': 'W0'}


def _fig_suffix():
    bits = []
    for key in sorted(BASE):
        if key in _DEFAULTS and BASE[key] != _DEFAULTS[key]:
            val = BASE[key]
            val = ('%g' % val) if isinstance(val, (int, float)) else str(val)
            bits.append('%s%s' % (_SHORT.get(key, key), val))
    return ('_' + '_'.join(bits)) if bits else ''


# Superscript f = free density, a = Fmi-associated / complex-associated density;
# the trailing digit is the cell index (1 = target cell, 2 = pseudo-cell).
FIELDS_MD = ['Ff1', 'Fa1', 'Sf1', 'Sa1', 'Vf1', 'Va1', 'Pf1', 'Pa1',
             'Ff2', 'Fa2', 'Sf2', 'Sa2', 'Vf2', 'Va2', 'Pf2', 'Pa2', 'C1', 'C2']
FIELDS_CKI = FIELDS_MD + ['Wf1', 'Wa1', 'Wf2', 'Wa2']

POLARITY_THRESHOLD = 0.15
PEAK_HEIGHT = 0.1
# A punctum must also stand out from its surroundings, not merely exceed an
# absolute height: a uniform profile at density 2.7 (the n=1 case, or the
# dsh;pk double mutant) otherwise returns ~9 "puncta" that are numerical
# ripples crossing the height threshold. The prominence required is this
# fraction of the profile's mean density.
PEAK_PROMINENCE_FRAC = 0.02

GREEN, ORANGE, VIOLET = '#2ca02c', '#ff7f0e', '#7b3fa0'


# ======================================================================
# model
# ======================================================================
class Grid:
    """Holds N and dx so the right-hand sides can be resolution-agnostic."""
    N = 200
    dx = 1.0 / 200


def lap(u, D):
    """Periodic second-difference Laplacian, scaled by D."""
    if D == 0.0:
        return 0.0
    return D * (np.roll(u, 1) - 2.0 * u + np.roll(u, -1)) / Grid.dx**2


def G(y, K, q=2):
    """Recruitment function y^q / (K^q + y^q)."""
    return y**q / (K**q + y**q)


def unpack(y, fields):
    """Split the state vector into a dict of per-field arrays."""
    N = Grid.N
    return {f: y[k * N:(k + 1) * N] for k, f in enumerate(fields)}


def rhs_MD(t, y, p):
    """Mutual destabilisation model, Eqs (1)-(10)."""
    v = unpack(y, FIELDS_MD)
    n, K = p['n'], p['K']
    d = {}
    for i, j in ((1, 2), (2, 1)):
        Ff, Fa = v['Ff%d' % i], v['Fa%d' % i]
        Sf, Sa = v['Sf%d' % i], v['Sa%d' % i]
        Vf, Va = v['Vf%d' % i], v['Va%d' % i]
        Pf, Pa = v['Pf%d' % i], v['Pa%d' % i]
        Vaj, Paj = v['Va%d' % j], v['Pa%d' % j]
        Faj, Saj = v['Fa%d' % j], v['Sa%d' % j]
        Ci, Cj = v['C%d' % i], v['C%d' % j]

        # feedback-modulated off-rates
        offF = (
            p['k_off_FS'] * Fa * (1 + p['gamma'] * Va * Pa**n) / (1 + p['tau'] * Va**n)
        )
        offS = p['k_off_FS'] * Sa * (1 + p['xi'] * Va**n) / (1 + p['eta'] * Pa**n)
        offC = (
            p['k_unbind']
            * Ci
            * (1 + p['gamma'] * Va * Pa**n)
            * (1 + p['xi'] * Vaj**n)
            / ((1 + p['tau'] * Va**n) * (1 + p['eta'] * Paj**n))
        )
        offCj = (
            p['k_unbind']
            * Cj
            * (1 + p['gamma'] * Vaj * Paj**n)
            * (1 + p['xi'] * Va**n)
            / ((1 + p['tau'] * Vaj**n) * (1 + p['eta'] * Pa**n))
        )

        d['Ff%d' % i] = lap(Ff, p['D_f']) - p['k_on'] * Ff + offF
        d['Fa%d' % i] = (lap(Fa, p['D_a']) + p['k_on'] * Ff - offF
                         - p['k_bind'] * Fa * Saj + offC)
        d['Sf%d' % i] = lap(Sf, p['D_f']) - p['k_on'] * Sf + offS
        d['Sa%d' % i] = (lap(Sa, p['D_a']) + p['k_on'] * Sf - offS
                         - p['k_bind'] * Faj * Sa + offCj)
        d['C%d' % i] = p['k_bind'] * Fa * Saj - offC       # trans-complexes immobile

        recV = Vf * (p['k_basal'] + p['k_on'] * G(Fa + Ci, K, p['hill_q']))
        recP = Pf * (p['k_basal'] + p['k_on'] * G(Sa + Cj, K, p['hill_q']))
        d['Vf%d' % i] = lap(Vf, p['D_f']) - recV + p['k_off_VP'] * Va
        d['Va%d' % i] = recV - p['k_off_VP'] * Va  # complex-associated Dsh immobile
        d['Pf%d' % i] = lap(Pf, p['D_f']) - recP + p['k_off_VP'] * Pa
        d['Pa%d' % i] = recP - p['k_off_VP'] * Pa  # complex-associated Pk immobile
    return np.concatenate([d[f] for f in FIELDS_MD])


def rhs_CKI(t, y, p):
    """CKIe model, Eqs (11)-(18) with (1),(2),(9),(10) unchanged."""
    v = unpack(y, FIELDS_CKI)
    n, K = p['n'], p['K']
    d = {}
    for i, j in ((1, 2), (2, 1)):
        Ff, Fa = v['Ff%d' % i], v['Fa%d' % i]
        Sf, Sa = v['Sf%d' % i], v['Sa%d' % i]
        Vf, Va = v['Vf%d' % i], v['Va%d' % i]
        Pf, Pa = v['Pf%d' % i], v['Pa%d' % i]
        Wf, Wa = v['Wf%d' % i], v['Wa%d' % i]
        Vaj, Paj, Waj = v['Va%d' % j], v['Pa%d' % j], v['Wa%d' % j]
        Faj, Saj = v['Fa%d' % j], v['Sa%d' % j]
        Ci, Cj = v['C%d' % i], v['C%d' % j]

        # Dsh->Stbm antagonism replaced by CKIe-mediated destabilisation
        offF = (
            p['k_off_FS'] * Fa * (1 + p['gamma'] * Va * Pa**n) / (1 + p['tau'] * Va**n)
        )
        offS = p['k_off_FS'] * Sa * (1 + p['mu'] * Wa**n) / (1 + p['eta'] * Pa**n)
        offC = (
            p['k_unbind']
            * Ci
            * (1 + p['gamma'] * Va * Pa**n)
            * (1 + p['mu'] * Waj**n)
            / ((1 + p['tau'] * Va**n) * (1 + p['eta'] * Paj**n))
        )
        offCj = (
            p['k_unbind']
            * Cj
            * (1 + p['gamma'] * Vaj * Paj**n)
            * (1 + p['mu'] * Wa**n)
            / ((1 + p['tau'] * Vaj**n) * (1 + p['eta'] * Pa**n))
        )

        d['Ff%d' % i] = lap(Ff, p['D_f']) - p['k_on'] * Ff + offF
        d['Fa%d' % i] = (lap(Fa, p['D_a']) + p['k_on'] * Ff - offF
                         - p['k_bind'] * Fa * Saj + offC)
        d['Sf%d' % i] = lap(Sf, p['D_f']) - p['k_on'] * Sf + offS
        d['Sa%d' % i] = (lap(Sa, p['D_a']) + p['k_on'] * Sf - offS
                         - p['k_bind'] * Faj * Sa + offCj)
        d['C%d' % i] = p['k_bind'] * Fa * Saj - offC

        recV = Vf * (p['k_basal'] + p['k_on'] * G(Fa + Ci, K, p['hill_q']))
        offV = p['k_off_VP'] * Va / (1 + p['nu'] * Wa**n)   # CKIe stabilises Dsh
        recP = Pf * (p['k_basal'] + p['k_on'] * G(Sa + Cj, K, p['hill_q']))
        # CKIe recruited by local complex-associated Dsh, or uniformly if the
        # distal bias is removed (Figure 6M). Mass action is linear in Dsh and
        # unbounded; the Hill form saturates. K is the half-saturation constant
        # of the Hill function and has no role in mass action, so it does NOT
        # enter the mass-action term: the single mass-action rate is k_on_W.
        if p['cki_recruit'] == 'mass_action':
            gV = Va
        elif p['cki_recruit'] == 'hill':
            gV = G(Va, K, p['hill_q'])
        else:
            raise ValueError("cki_recruit must be 'mass_action' or 'hill'")
        if p['uniform_W']:
            gV = np.full_like(Va, gV.mean())
        recW = Wf * (p['k_basal'] + p['k_on_W'] * gV)

        d['Vf%d' % i] = lap(Vf, p['D_f']) - recV + offV
        d['Va%d' % i] = recV - offV
        d['Pf%d' % i] = lap(Pf, p['D_f']) - recP + p['k_off_VP'] * Pa
        d['Pa%d' % i] = recP - p['k_off_VP'] * Pa
        d['Wf%d' % i] = lap(Wf, p['D_f']) - recW + p['k_off_W'] * Wa
        d['Wa%d' % i] = lap(Wa, p['D_CKIe']) + recW - p['k_off_W'] * Wa
    return np.concatenate([d[f] for f in FIELDS_CKI])


def correlated_noise(rng, N, amplitude, sigma_perim, n_ref=None):
    """Random field with a fixed physical correlation length, generated on a
    RESOLUTION-INDEPENDENT reference grid.

    Two separate problems are being solved here.

    (1) Independent draws at each grid point are white noise whose correlation
        length is the grid spacing, so finer grids nucleate more puncta and the
        count does not converge under refinement. Smoothing with a kernel of
        fixed width in units of cell perimeter (sigma_perim) removes that.

    (2) Drawing N values and smoothing them still gives a DIFFERENT physical
        field at each N, so runs at different resolutions from the same seed
        are independent realisations. Convergence then has to be assessed
        across the full seed-to-seed spread, which needs many seeds. Drawing
        on a fixed reference grid of n_ref points and interpolating down means
        one seed gives (to interpolation error) the same physical field at
        every N, so refinement comparisons are PAIRED and need far fewer seeds.

    Set n_ref to 0 to recover the older per-N behaviour. n_ref should be at
    least as large as the finest grid you intend to run.
    """
    if sigma_perim <= 0:
        return rng.uniform(0.0, amplitude, N)      # legacy white-noise behaviour

    if n_ref is None:
        n_ref = N_REF
    if not n_ref:                                  # legacy per-N realisation
        r = gaussian_filter1d(rng.uniform(0.0, amplitude, N), sigma_perim * N,
                              mode='wrap')
    else:
        r = gaussian_filter1d(rng.uniform(0.0, amplitude, n_ref),
                              sigma_perim * n_ref, mode='wrap')
        if N != n_ref:                             # periodic linear interpolation
            xr = np.arange(n_ref) / n_ref
            x = np.arange(N) / N
            r = np.interp(x, np.append(xr, 1.0), np.append(r, r[0]))
    r = r - r.min()
    return r / max(r.max(), 1e-12) * amplitude


def initial_state(p, model, seed):
    """Initial state: uniform free densities, correlated noise in the
    Fmi-associated intermediates, zero elsewhere."""
    rng = np.random.default_rng(seed)
    fields = FIELDS_MD if model == 'MD' else FIELDS_CKI
    N = Grid.N
    out = {}
    for f in fields:
        state, species = f[1], f[0]          # e.g. 'Fa1' -> state 'a', species 'F'
        if state == 'f':                     # free densities, uniform
            if species == 'V':
                out[f] = np.full(N, p['init_free'] * p.get('dsh_dose', 1.0))
            elif species == 'P':
                out[f] = np.full(N, p['init_free'] * p.get('pk_dose', 1.0))
            elif species == 'W':
                out[f] = np.full(N, p['init_W'] * p.get('cki_dose', 1.0))
            else:                            # F, S
                out[f] = np.full(N, p['init_free'])
        elif state == 'a' and species in 'FS':
            out[f] = correlated_noise(rng, N, p['init_inter'], p['sigma_ic'],
                                      n_ref=p.get('n_ref', N_REF))
        else:                                # complex-associated V/P/W and C1, C2
            out[f] = np.zeros(N)
    return np.concatenate([out[f] for f in fields])


def jac_sparsity(nf, fields=None, nonlocal_W=False):
    """Local coupling: every field couples to every field at the same point,
    and to itself at neighbouring points.

    If nonlocal_W, CKIe recruitment depends on the spatial MEAN of Dsh, so the
    W rows couple to ALL V columns. Omitting this gives BDF a wrong Jacobian
    and the solver stalls."""
    N = Grid.N
    S = lil_matrix((nf * N, nf * N), dtype=np.int8)
    for a in range(nf):
        for b in range(nf):
            for k in range(N):
                S[a * N + k, b * N + k] = 1
        for k in range(N):
            S[a * N + k, a * N + (k - 1) % N] = 1
            S[a * N + k, a * N + (k + 1) % N] = 1
    if nonlocal_W and fields is not None:
        wrows = [fields.index(f) for f in fields if f[0] == 'W']
        vcols = [fields.index(f) for f in fields if f in ('Va1', 'Va2')]
        for a in wrows:
            for b in vcols:
                S[a * N:(a + 1) * N, b * N:(b + 1) * N] = 1
    return S.tocsr()


# ======================================================================
# quantification (STAR Methods)
# ======================================================================
def polarity_index(rho):
    """Anisotropy of the moment-of-inertia tensor of rho mapped to a circle.
    NOTE this is a second-moment (nematic) measure: one domain and two
    antipodal domains both score 1."""
    k = np.arange(len(rho))
    x, y = np.cos(2 * np.pi * k / len(rho)), np.sin(2 * np.pi * k / len(rho))
    M = rho.sum()
    if M <= 0:
        return 0.0
    inertia = np.array([[(rho * y**2).sum() / M, (rho * x * y).sum() / M],
                        [(rho * x * y).sum() / M, (rho * x**2).sum() / M]])
    l1, l2 = sorted(np.linalg.eigvalsh(inertia))[::-1]
    return (l1 - l2) / (l1 + l2) if (l1 + l2) > 0 else 0.0


def _peaks(rho, height=PEAK_HEIGHT, prominence_frac=None):
    """Indices of peaks in the middle copy of a tripled (periodic) profile."""
    if prominence_frac is None:
        prominence_frac = PEAK_PROMINENCE_FRAC
    ext = np.concatenate([rho, rho, rho])
    prom = prominence_frac * float(np.mean(rho))
    pk, _ = find_peaks(ext, height=height,
                       prominence=prom if prom > 0 else None)
    return ext, pk[(pk >= len(rho)) & (pk < 2 * len(rho))]


def count_puncta(rho, height=PEAK_HEIGHT, prominence_frac=None):
    """Puncta per cell: detected peaks halved, since each punctum lies on a
    junction shared with the pseudo-cell."""
    _, pk = _peaks(rho, height, prominence_frac)
    return len(pk) / 2.0


def punctum_width(rho, height=PEAK_HEIGHT):
    """Mean full width at half prominence of the individual puncta, in units of
    cell perimeter. Measured per peak rather than by thresholding the whole
    profile, so that peaks sitting on a broad polarised domain are not merged.
    Returns nan if no peak is resolved."""
    ext, pk = _peaks(rho, height)
    if len(pk) == 0:
        return np.nan
    widths, _, _, _ = peak_widths(ext, pk, rel_height=0.5)
    return float(np.mean(widths)) / len(rho)


def perimeter_to_nm(frac, perimeter_um=20.0):
    """Convert a fraction of the cell perimeter to nanometres."""
    return frac * perimeter_um * 1000.0


# ======================================================================
# simulation driver
# ======================================================================
def fz_residence(v, p, model):
    """Half-life of junctional Fz, from trans-complex to free, in model time.

    Fz leaves in two steps in sequence, C_i -> F_i^a (rate k_unbind) and
    F_i^a -> F_i^f (rate k_off_FS), both modulated by feedback, so the
    reciprocal rates add. Returned at the position of maximum trans-complex
    density (the punctum peak, as in the Methods) and as a trans-complex-
    weighted mean over the perimeter.
    """
    n = p['n']
    V, P = v['Va1'], v['Pa1']
    V2, P2 = v['Va2'], v['Pa2']
    fz_fac = (1 + p['gamma'] * V * P**n) / (1 + p['tau'] * V**n)
    if model == 'MD':
        stbm_fac = (1 + p['xi'] * V2**n) / (1 + p['eta'] * P2**n)
    else:
        stbm_fac = (1 + p['mu'] * v['Wa2']**n) / (1 + p['eta'] * P2**n)
    k_unbind = p['k_unbind'] * fz_fac * stbm_fac
    k_off = p['k_off_FS'] * fz_fac
    th = np.log(2) * (1.0 / k_unbind + 1.0 / k_off)
    C = v['C1']
    return dict(peak=float(th[np.argmax(C)]),
                weighted=float(np.sum(th * C) / np.sum(C)) if C.sum() > 0 else np.nan)


def time_unit_seconds(p=None):
    """T0 from the Dsh/Pk fast phases, matched to their unmodulated off-rate
    k_off_VP. This is independent of the feedback parameters and of which
    model is used."""
    p = p or BASE
    return np.mean(FRAP_FAST_DSH_PK) / (np.log(2) / p['k_off_VP'])


def simulate(model, overrides, tmax, seed, sparsity=None, from_base=True):
    """Run one simulation to tmax and return profiles and summary measures."""
    if from_base:
        p = dict(BASE)
        p.update(overrides or {})
    else:
        p = dict(overrides)          # already a complete parameter dict
    fields = FIELDS_MD if model == 'MD' else FIELDS_CKI
    rhs = rhs_MD if model == 'MD' else rhs_CKI
    if sparsity is None:
        sparsity = jac_sparsity(len(fields), fields,
                                nonlocal_W=bool(p.get('uniform_W')))
    sol = solve_ivp(rhs, (0, tmax), initial_state(p, model, seed), args=(p,),
                    method='BDF', jac_sparsity=sparsity,
                    rtol=1e-6, atol=1e-8, t_eval=[tmax])
    if not sol.success:
        raise RuntimeError('integration failed: ' + sol.message)
    v = unpack(sol.y[:, -1], fields)
    fz = v['Fa1'] + v['C1']         # complex-associated Fz in the target cell
    stbm = v['Sa1'] + v['C2']       # complex-associated Stbm in the target cell
    cki = v['Wa1'] if model == 'CKI' else None
    return dict(fz=fz, stbm=stbm, cki=cki,
                polarity=polarity_index(fz), puncta=count_puncta(fz),
                width=punctum_width(fz), fractions=state_fractions(v, model),
                residence=fz_residence(v, p, model))


def state_fractions(v, model):
    """Steady-state partition of each species in the target cell (cell 1),
    integrated over the whole perimeter.

    cki_bound   complex-associated CKIe / total CKIe
    dsh_bound   complex-associated Dsh / total Dsh
    pk_bound    complex-associated Pk / total Pk
    fz_assoc    (Fmi-associated + trans-complexed) Fz / total Fz
    fz_inter    Fmi-associated Fz / (Fmi-associated + trans-complexed) Fz,
                i.e. the share of junctional Fz NOT in trans-complexes

    For comparison with FRAP, note that FRAP fractions are fractions of the
    fluorescence recovering in a bleached junctional region, whereas these
    are fractions of each cell's total. The closest like-for-like comparison is
    cki_bound against the CKIe slow fraction, since CKIe is mostly free.
    """
    def tot(*keys):
        return float(sum(v[k].sum() for k in keys))

    def ratio(num, den):
        # null mutants (dose 0) have no protein at all; report nan, not an error
        return num / den if den > 0 else float('nan')
    f = dict(dsh_bound=ratio(tot('Va1'), tot('Vf1', 'Va1')),
             pk_bound=ratio(tot('Pa1'), tot('Pf1', 'Pa1')),
             fz_assoc=ratio(tot('Fa1', 'C1'), tot('Ff1', 'Fa1', 'C1')),
             fz_inter=ratio(tot('Fa1'), tot('Fa1', 'C1')))
    if model == 'CKI':
        f['cki_bound'] = ratio(tot('Wa1'), tot('Wf1', 'Wa1'))
    return f


def _job(args):
    # NOTE: the full parameter dict is passed explicitly, not read from BASE.
    # On Windows, multiprocessing uses 'spawn': each worker re-imports this
    # module and rebuilds BASE from its module-level defaults, so anything
    # main() wrote into BASE (--K, --sigma, --cki-recruit, --D_a, --D_CKIe)
    # would be silently lost in the workers while the parent still printed the
    # overridden value. Passing params through the task tuple avoids that.
    model, params, tmax, seed, N = args
    Grid.N, Grid.dx = N, 1.0 / N
    r = simulate(model, params, tmax, seed, from_base=False)
    return (r['polarity'], r['puncta'], r['fz'], r['stbm'], r['cki'], r['width'],
            r['fractions'], r['residence'])


def run_many(model, overrides, tmax, seeds, N, jobs):
    """Run one simulation per seed, in parallel if jobs > 1."""
    params = dict(BASE)
    params.update(overrides or {})
    tasks = [(model, params, tmax, s, N) for s in seeds]
    n_workers = min(jobs, len(tasks))
    if n_workers > 1:
        from multiprocessing import Pool
        with Pool(n_workers) as pool:
            return pool.map(_job, tasks)
    return [_job(t) for t in tasks]


# ======================================================================
# conditions for Figure 6
# ======================================================================
#  Perturbations are applied as parameter or dosage overrides. Dosages of 0
#  represent null mutants; 2.0 represents overexpression.
#  Panels B-D vary D_CKIe around whatever value is in force, rather than using
#  hardcoded numbers, so they stay meaningful when the CKIe parameters change:
#    B  immobile complex-associated CKIe (D_CKIe = 0)
#    C  the default, i.e. spread length CKI_SPREAD_LENGTH
#    D  "high", equal to the free-species diffusion coefficient D_f, matching
#       the legend's "comparable to unbound diffusion coefficient values"
#  The B-D entries below are placeholders, filled in from BASE at run time
#  by fig6_conditions().
FIG6_CONDITIONS = [
    ('B', 'low $D_{CKI\\varepsilon}$',        None),
    ('C', 'moderate $D_{CKI\\varepsilon}$',   None),
    ('D', 'high $D_{CKI\\varepsilon}$',       None),
    ('E', '$pk$ mutant',                      dict(pk_dose=0.0)),
    ('F', '$dsh$ mutant',                     dict(dsh_dose=0.0)),
    ('G', '$dsh; pk$ double mutant',          dict(pk_dose=0.0, dsh_dose=0.0)),
    ('H', 'Pk overexpression',                dict(pk_dose=2.0)),
    ('I', 'CKI$\\varepsilon$ mutant',         dict(cki_dose=0.0)),
    ('J', 'Fz destab. by Pk removed',         dict(gamma=0.0)),
    ('K', 'Stbm destab. by CKI$\\varepsilon$ removed', dict(mu=0.0)),
    ('L', 'Dsh stab. by CKI$\\varepsilon$ removed',    dict(nu=0.0)),
    ('M', 'CKI$\\varepsilon$ distal bias removed',     dict(uniform_W=True)),
]

def fig6_conditions():
    """Figure 6 conditions with B-D resolved against the current BASE."""
    rel = {'B': dict(D_CKIe=0.0),
           'C': dict(D_CKIe=BASE['D_CKIe']),
           'D': dict(D_CKIe=BASE['D_f'])}
    return [(lab, name, rel[lab] if ov is None else ov)
            for lab, name, ov in FIG6_CONDITIONS]


FIG2_DB = [1e-7, 1e-6, 1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]   # for panel H
FIG2_IL = [('I', 1e-7), ('J', 1e-6), ('K', 1e-5), ('L', 3e-4)]  # for panels I-L


# ======================================================================
# panels
# ======================================================================
def _run_tag(args):
    """Identify a run by grid, seeds, run time and the full parameter set.

    Cached panel data are only reused when all of these match, so a change of
    N, --reps, --tmax or any parameter override triggers a fresh computation
    rather than silently reusing old results.
    """
    import hashlib
    params = repr(sorted((k, repr(v)) for k, v in BASE.items()))
    h = hashlib.sha1(params.encode()).hexdigest()[:8]
    return 'N%d_r%d_t%g_%s' % (args.N, args.reps, args.tmax, h)


def _cache(args, name):
    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, '%s_%s.npz' % (name, _run_tag(args)))
    if os.path.exists(path) and not args.force:
        print('  using cached results in %s' % path, flush=True)
        return path, np.load(path, allow_pickle=True)
    return path, None


def panel_2H(args):
    """Figure 2H: polarity against punctum number across D_a."""
    import matplotlib.pyplot as plt
    path, cached = _cache(args, 'fig2H')
    if cached is None:
        pol, pun, dbs = [], [], []
        for db in FIG2_DB:
            print('  2H: D_a = %g' % db, flush=True)
            res = run_many('MD', dict(D_a=db), args.tmax,
                           range(args.reps), args.N, args.jobs)
            for r in res:
                pol.append(r[0])
                pun.append(r[1])
                dbs.append(db)
        np.savez(path, pol=pol, puncta=pun, db=dbs)
        cached = np.load(path)
    pol, pun, dbs = cached['pol'], cached['puncta'], cached['db']

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    sc = ax.scatter(pun, pol, c=np.log10(dbs), cmap='viridis', s=26,
                    edgecolors='k', linewidths=0.3, alpha=0.85)
    ax.axhline(POLARITY_THRESHOLD, ls='--', c='k', lw=1)
    ax.text(ax.get_xlim()[1], POLARITY_THRESHOLD, ' threshold 0.15',
            va='bottom', ha='right', fontsize=8)
    ax.set_xlabel('puncta number')
    ax.set_ylabel('Fz polarity')
    ax.set_title('Figure 2H', fontsize=10)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label(r'$\log_{10} D_a$')
    fig.tight_layout()
    fig.savefig(
        os.path.join(args.outdir, 'figure_2H_N%d%s.png' % (args.N, _fig_suffix())),
        dpi=200,
    )
    plt.close(fig)


def _profile_panel(ax, x, fz, stbm, cki, title):
    ax.plot(x, fz, color=GREEN, lw=1.3, label='complex-assoc. Fz')
    ax.plot(x, stbm, color=ORANGE, lw=1.3, label='complex-assoc. Stbm')
    if cki is not None:
        ax.plot(x, cki, color=VIOLET, lw=1.5, label=r'complex-assoc. CKI$\varepsilon$')
    ax.set_title(title, fontsize=9)
    ax.set_xlabel('position')


def panel_2IL(args):
    """Figures 2I-L: example profiles at four values of D_a."""
    import matplotlib.pyplot as plt
    path, cached = _cache(args, 'fig2IL')
    if cached is None:
        store = {}
        for lab, db in FIG2_IL:
            print('  2%s: D_a = %g' % (lab, db), flush=True)
            r = run_many('MD', dict(D_a=db), args.tmax, [0], args.N, 1)[0]
            store[lab] = np.array([r[2], r[3]])
        np.savez(path, **store)
        cached = np.load(path)

    x = np.linspace(0, 1, args.N, endpoint=False)
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4), sharey=True)
    for ax, (lab, db) in zip(axes, FIG2_IL):
        fz, stbm = cached[lab]
        # Panel titles deliberately carry no polarity/punctum numbers: these
        # are single example simulations, whereas the text and Figure 2H quote
        # medians over all seeds, and printing both invites the reader to
        # compare numbers that are not meant to match.
        print('  2%s (example run, seed 0): polarity %.2f, %.1f puncta'
              % (lab, polarity_index(fz), count_puncta(fz)), flush=True)
        _profile_panel(ax, x, fz, stbm, None,
                       '(%s) $D_a=%g$\nexample simulation' % (lab, db))
    axes[0].set_ylabel('density (a.u.)')
    axes[0].legend(fontsize=8)
    fig.suptitle('Figure 2I-L', fontsize=11)
    fig.tight_layout()
    fig.savefig(
        os.path.join(args.outdir, 'figure_2I-L_N%d%s.png' % (args.N, _fig_suffix())),
        dpi=200,
    )
    plt.close(fig)


def panel_6(args, which):
    """which = 'BD', 'EM' or 'NO'. All three share one cache."""
    import matplotlib.pyplot as plt
    path, cached = _cache(args, 'fig6')
    store = {} if cached is None else {k: cached[k] for k in cached.files}
    # Figure 6 is twelve conditions sharing one cache file, several hours at
    # production settings. Each condition is written as it finishes, so a
    # rerun resumes where an interrupted run stopped instead of starting over.
    conditions = fig6_conditions()
    todo = [c for c in conditions if ('pol_' + c[0]) not in store]
    if todo and len(todo) < len(conditions):
        print('  resuming: %d of %d conditions already done'
              % (len(conditions) - len(todo), len(conditions)), flush=True)
    for lab, name, ov in todo:
        print('  6%s: %s' % (lab, name), flush=True)
        res = run_many('CKI', ov, args.tmax, range(args.reps),
                       args.N, args.jobs)
        store['prof_' + lab] = np.array([res[0][2], res[0][3], res[0][4]])
        store['pol_' + lab] = np.array([r[0] for r in res])
        store['pun_' + lab] = np.array([r[1] for r in res])
        np.savez(path, **store)          # save as we go
    cached = store

    x = np.linspace(0, 1, args.N, endpoint=False)
    labels = {lab: name for lab, name, _ in FIG6_CONDITIONS}

    if which == 'BD':
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
        for ax, lab in zip(axes, ['B', 'C', 'D']):
            fz, stbm, cki = cached['prof_' + lab]
            # See the comment in panel_2IL: no numbers in profile panel titles.
            print('  6%s (example run, seed 0): polarity %.2f, %.1f puncta'
                  % (lab, polarity_index(fz), count_puncta(fz)), flush=True)
            _profile_panel(ax, x, fz, stbm, cki,
                           '(%s) %s\nexample simulation' % (lab, labels[lab]))
        axes[0].set_ylabel('density (a.u.)')
        axes[0].legend(fontsize=8)
        fig.suptitle('Figure 6B-D', fontsize=11)
        fig.tight_layout()
        fig.savefig(
            os.path.join(
                args.outdir, 'figure_6B-D_N%d%s.png' % (args.N, _fig_suffix())
            ),
            dpi=200,
        )
        plt.close(fig)

    elif which == 'EM':
        labs = [c for c, _, _ in FIG6_CONDITIONS if c in 'EFGHIJKLM']
        fig, axes = plt.subplots(3, 3, figsize=(13, 10), sharey=True)
        for ax, lab in zip(axes.ravel(), labs):
            fz, stbm, cki = cached['prof_' + lab]
            print('  6%s (example run, seed 0): polarity %.2f, %.1f puncta'
                  % (lab, polarity_index(fz), count_puncta(fz)), flush=True)
            _profile_panel(ax, x, fz, stbm, cki,
                           '(%s) %s\nexample simulation' % (lab, labels[lab]))
        axes[0, 0].set_ylabel('density (a.u.)')
        axes[0, 0].legend(fontsize=8)
        fig.suptitle('Figure 6E-M', fontsize=11)
        fig.tight_layout()
        fig.savefig(
            os.path.join(
                args.outdir, 'figure_6E-M_N%d%s.png' % (args.N, _fig_suffix())
            ),
            dpi=200,
        )
        plt.close(fig)

    elif which == 'NO':
        labs = [c for c, _, _ in FIG6_CONDITIONS]
        fig, axes = plt.subplots(2, 1, figsize=(9, 8))
        for ax, key, ylab, title in (
            (axes[0], 'pol_', 'Fz polarity', '(N) polarity'),
            (axes[1], 'pun_', 'puncta number', '(O) puncta number'),
        ):
            data = [cached[key + c] for c in labs]
            try:
                ax.boxplot(data, tick_labels=labs, showfliers=False,
                           medianprops=dict(color='k'))
            except TypeError:                      # matplotlib < 3.9
                ax.boxplot(data, labels=labs, showfliers=False,
                           medianprops=dict(color='k'))
            for k, d in enumerate(data, start=1):
                ax.plot(np.full(len(d), k) + np.random.uniform(-.13, .13, len(d)),
                        d, '.', color='#1f77b4', ms=4, alpha=0.6)
            ax.set_ylabel(ylab)
            ax.set_title(title, fontsize=10)
            if key == 'pol_':
                ax.axhline(POLARITY_THRESHOLD, ls='--', c='k', lw=1)
        axes[1].set_xlabel('condition (panels B-M)')
        fig.suptitle('Figure 6N,O  (n = %d simulations per condition)' % args.reps,
                     fontsize=11)
        fig.tight_layout()
        fig.savefig(
            os.path.join(
                args.outdir, 'figure_6N-O_N%d%s.png' % (args.N, _fig_suffix())
            ),
            dpi=200,
        )
        plt.close(fig)


def panel_width(args):
    """Punctum FWHM against D_a (Figure S7D).

    Width is measured at each D_a, including the value used throughout
    (BASE['D_a']), and the power-law exponent is fitted. The simulated
    widths do not follow the square-root scaling of linear theory.

    Each D_a is cached separately, so adding a value does not recompute the
    others.
    """
    import matplotlib.pyplot as plt
    dbs = args.width_db or [1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4]
    nseed = max(1, min(args.reps, 5))
    seeds = list(range(args.seed, args.seed + nseed))
    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, 'width2_%s.npz' % _run_tag(args))
    store = {}
    if os.path.exists(path) and not args.force:
        with np.load(path) as f:
            store = {k: f[k] for k in f.files}

    print('\nPunctum width against D_a  (N=%d, %d seeds, dx=%.0f nm)\n'
          % (args.N, len(seeds), perimeter_to_nm(1.0 / args.N)))
    print('  %10s %14s %12s %10s' % ('D_a', 'FWHM (nm)', 'SD (nm)', 'points'))
    for db in dbs:
        key, ks = 'w_%g' % db, 's_%g' % db
        if key in store and ks in store and np.array_equal(store[ks], np.array(seeds)):
            vals, tag = store[key], '  (cached)'
        else:
            res = run_many('MD', dict(D_a=db), args.tmax, seeds, args.N, args.jobs)
            vals = np.array([r[5] for r in res if np.isfinite(r[5])])
            store[key], store[ks] = vals, np.array(seeds)
            np.savez(path, **store)          # save as we go; runs are slow
            tag = ''
        if len(vals) == 0:
            print('  %10g %14s %12s %10d%s'
                  % (db, 'no peak', '-', 0, tag), flush=True)
            continue
        nm = perimeter_to_nm(vals)
        # A punctum needs several grid points to be a width rather than a
        # spike: flag any value that is not resolved at this dx.
        flag = (
            ' <- under 5 grid points'
            if nm.mean() < 5 * perimeter_to_nm(1.0 / args.N)
            else ''
        )
        print('  %10g %14.0f %12.0f %10d%s%s'
              % (db, nm.mean(), nm.std(), len(vals), tag, flag), flush=True)

    have = [(db, perimeter_to_nm(store['w_%g' % db]).mean()) for db in dbs
            if ('w_%g' % db) in store and len(store['w_%g' % db])]
    if len(have) >= 2:
        x = np.log10([h[0] for h in have])
        y = np.log10([h[1] for h in have])
        slope, intercept = np.polyfit(x, y, 1)
        print('\n  fitted FWHM ~ D_a^%.2f  (a square-root law would give 0.50)'
              % slope)
        if BASE['D_a'] in [h[0] for h in have]:
            w0 = dict(have)[BASE['D_a']]
            print('  measured width at the operating point D_a = %g: %.0f nm'
                  % (BASE['D_a'], w0))
        else:
            print('  NOTE: the operating point D_a = %g was not simulated; the '
                  'width there is unmeasured.' % BASE['D_a'])

    fig, ax = plt.subplots(figsize=(5.4, 4.3))
    if have:
        ax.loglog([h[0] for h in have], [h[1] for h in have], 'o',
                  color='#2ca02c', ms=7, label='simulated')
    if len(have) >= 2:
        grid = np.logspace(np.log10(min(h[0] for h in have)),
                           np.log10(max(h[0] for h in have)), 50)
        ax.loglog(grid, 10**intercept * grid**slope, '-', color='#d62728',
                  lw=1.4, label=r'fit: FWHM $\propto D_a^{%.2f}$' % slope)
        ref = dict(have)[have[0][0]] / np.sqrt(have[0][0])
        ax.loglog(grid, ref * np.sqrt(grid), '--', color='#888', lw=1.2,
                  label=r'$\sqrt{D_a}$ reference')
    ax.axvline(BASE['D_a'], color='#888', lw=1)
    ax.set_xlabel(r'$D_a$')
    ax.set_ylabel('punctum FWHM (nm)')
    ax.set_title('Punctum width against $D_a$ (N=%d, %d seeds)'
                 % (args.N, len(seeds)), fontsize=10)
    ax.legend(fontsize=8)
    fig.text(0.01, 0.01, 'Vertical line: the value used in the paper. '
             'Widths are means over seeds of the per-peak full width at half '
             'prominence.', fontsize=6.5, color='#555', va='bottom')
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = os.path.join(
        args.outdir, 'punctum_width_scaling_N%d%s.png' % (args.N, _fig_suffix())
    )
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print('\n  written to %s\n  per-seed values in %s\n' % (out, path))


def _mathexp(v):
    """Render a clean power of ten as mathtext (1e-07 -> 10^{-7}),
    so figure titles do not show "1e-07"."""
    import math
    if v > 0:
        k = math.log10(v)
        if abs(k - round(k)) < 1e-9:
            return '10^{%d}' % int(round(k))
    return '%g' % v


def width_sigma(args):
    """Punctum width against the initial correlation length, at fixed D_a.

    This is the evidence that punctum width is inherited from sigma rather
    than generated by the dynamics. Each sigma has its own parameter tag, so
    values already measured by `--panels width --sigma X` are reused.
    """
    import matplotlib.pyplot as plt
    sigmas = args.width_sigma
    db = (args.width_db or [BASE['D_a']])[0]
    nseed = max(1, min(args.reps, 5))
    seeds = list(range(args.seed, args.seed + nseed))
    keep = BASE['sigma_ic']
    out = []
    print('\nPunctum width against the initial correlation length  '
          '(N=%d, D_a=%g, %d seeds)\n' % (args.N, db, len(seeds)))
    print('  %10s %10s %14s %12s' % ('sigma', 'sigma (nm)', 'FWHM (nm)', 'SD (nm)'))
    for sig in sigmas:
        BASE['sigma_ic'] = sig
        path = os.path.join(args.outdir, 'width2_%s.npz' % _run_tag(args))
        key, ks = 'w_%g' % db, 's_%g' % db
        vals = None
        if os.path.exists(path) and not args.force:
            with np.load(path) as f:
                if key in f.files and np.array_equal(f[ks], np.array(seeds)):
                    vals = f[key]
        tag = '  (cached)'
        if vals is None:
            res = run_many('MD', dict(D_a=db), args.tmax, seeds, args.N, args.jobs)
            vals = np.array([r[5] for r in res if np.isfinite(r[5])])
            store = {}
            if os.path.exists(path):
                with np.load(path) as f:
                    store = {k: f[k] for k in f.files}
            store[key], store[ks] = vals, np.array(seeds)
            np.savez(path, **store)
            tag = ''
        nm = perimeter_to_nm(vals)
        out.append((sig, nm.mean(), nm.std()))
        print('  %10g %10.0f %14.0f %12.0f%s'
              % (sig, perimeter_to_nm(sig), nm.mean(), nm.std(), tag), flush=True)
    BASE['sigma_ic'] = keep

    xs = np.array([perimeter_to_nm(o[0]) for o in out])
    ys = np.array([o[1] for o in out])
    es = np.array([o[2] for o in out])
    slope, intercept = np.polyfit(np.log10(xs), np.log10(ys), 1)
    ratio = (ys / xs).mean()
    print(
        '\n  fitted FWHM ~ sigma^%.2f; FWHM / sigma = %.2f on average' % (slope, ratio)
    )

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.errorbar(xs, ys, yerr=es, fmt='o', color='#2ca02c', ms=7, capsize=3,
                label='simulated')
    grid = np.logspace(np.log10(xs.min()), np.log10(xs.max()), 50)
    ax.plot(grid, ratio * grid, '-', color='#d62728', lw=1.4,
            label='FWHM $= %.1f\\sigma$' % ratio)
    ax.set_xscale('log')
    ax.set_yscale('log')
    # label the sigma values actually simulated, not log decades
    from matplotlib.ticker import NullLocator, NullFormatter
    ax.set_xticks(xs)
    ax.set_xticklabels(['%g' % v for v in xs])
    ax.xaxis.set_minor_locator(NullLocator())
    yt = [t for t in (400, 600, 800, 1000, 1500, 2000)
          if ys.min() / 1.3 <= t <= ys.max() * 1.3]
    if len(yt) >= 2:
        ax.set_yticks(yt)
        ax.set_yticklabels(['%g' % t for t in yt])
        ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel('initial correlation length $\\sigma$ (nm)')
    ax.set_ylabel('punctum FWHM (nm)')
    ax.set_title('Punctum width against initial correlation length\n'
                 '($D_a = %s$, N=%d, %d seeds)'
                 % (_mathexp(db), args.N, len(seeds)),
                 fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    name = os.path.join(args.outdir, 'width_vs_sigma_N%d.png' % args.N)
    fig.savefig(name, dpi=200)
    plt.close(fig)
    print('  written to %s\n' % name)


def ic_test(args):
    """Shows that the correlated initial condition removes the grid dependence."""
    print('\nPuncta number under mesh refinement, D_a = %g\n' % BASE['D_a'])
    print('%8s %22s %22s' % ('N', 'white noise (sigma=0)', 'correlated (sigma=%.3f)'
                             % BASE['sigma_ic']))
    for N in (100, 200, 400):
        a = run_many('MD', dict(sigma_ic=0.0), args.tmax, [args.seed], N, 1)[0]
        b = run_many('MD', dict(), args.tmax, [args.seed], N, 1)[0]
        print('%8d %22.1f %22.1f' % (N, a[1], b[1]))
    print('\nWhite noise has a correlation length equal to the grid spacing, so\n'
          'finer grids nucleate more puncta. The correlated initial condition\n'
          'gives a grid-independent count.\n')


def sigma_test(args):
    """Sensitivity to the assumed correlation length of the initial fluctuations.

    sigma_ic sets punctum number in both models (it is not selected by the
    dynamics), so this is the parameter to tune against the observed two to
    three large puncta per cell, once the CKIe parameters have been fixed from
    the FRAP data.

    Per-seed polarity and punctum number are saved to sigma_test_<tag>.npz
    (tag = N, reps, tmax and a parameter hash) as each sigma finishes, and
    reused on a later run unless --force is given, so an interrupted run can
    be resumed. The summary plot,
    sigma_polarity_puncta_N<N>.png, shows polarity against punctum number with
    one curve per model and each point labelled by sigma: this is the
    comparison of the two models at matched punctum number.
    """
    import matplotlib.pyplot as plt
    sigmas = args.sigmas or [0.005, 0.007, 0.010, 0.014, 0.020, 0.040]
    seeds = list(range(args.seed, args.seed + max(1, args.reps)))
    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, 'sigma_test_%s.npz' % _run_tag(args))
    store = {}
    if os.path.exists(path) and not args.force:
        with np.load(path) as f:
            store = {k: f[k] for k in f.files}
    header = ('N=%d, tmax=%g, %d seeds, k_off_W=%.4f, D_CKIe=%.5f, L=%.3f, '
              'mu=%.3f, nu=%.3f, k_basal=%.4f, k_on_W=%.4f, K=%g'
              % (args.N, args.tmax, len(seeds), BASE['k_off_W'], BASE['D_CKIe'],
                 np.sqrt(BASE['D_CKIe'] / BASE['k_off_W']), BASE['mu'],
                 BASE['nu'], BASE['k_basal'], BASE['k_on_W'], BASE['K']))
    print('\nSensitivity to the initial correlation length\n  (%s)\n' % header)
    for model, label in (('MD', 'mutual destabilisation'), ('CKI', 'CKIe')):
        print('  %s' % label)
        print('  %8s %8s %18s %16s' % ('sigma', 'nm', 'polarity', 'puncta'))
        for sig in sigmas:
            kp, ku = '%s_%g_pol' % (model, sig), '%s_%g_pun' % (model, sig)
            ks = '%s_%g_seeds' % (model, sig)
            if kp in store and ks in store and \
                    np.array_equal(store[ks], np.array(seeds)):
                pol, pun, tag = store[kp], store[ku], '  (cached)'
            else:
                res = run_many(model, dict(sigma_ic=sig), args.tmax, seeds,
                               args.N, args.jobs)
                pol = np.array([r[0] for r in res])
                pun = np.array([r[1] for r in res])
                store[kp], store[ku], store[ks] = pol, pun, np.array(seeds)
                store['header'] = np.array(header)
                np.savez(path, **store)          # save as we go
                tag = ''
            print('  %8.3f %8.0f %9.3f +/- %.3f %7.1f +/- %.1f%s'
                  % (sig, perimeter_to_nm(sig), pol.mean(), pol.std(),
                     pun.mean(), pun.std(), tag), flush=True)
        print()
    print('Target: 2-3 large puncta per cell by confocal.\n')

    # Polarity against punctum number, one curve per model.
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    ax.axvspan(2, 3, color='#2ca02c', alpha=0.12, lw=0)
    for model, c, lab in (('MD', '#1f77b4', 'mutual destabilisation'),
                          ('CKI', '#d62728', r'CKI$\varepsilon$ model')):
        pol = [store['%s_%g_pol' % (model, s)] for s in sigmas]
        pun = [store['%s_%g_pun' % (model, s)] for s in sigmas]
        for p, u in zip(pol, pun):                 # individual seeds
            ax.scatter(u, p, s=9, color=c, alpha=0.25, lw=0)
        xm = np.array([u.mean() for u in pun])
        ym = np.array([p.mean() for p in pol])
        ax.errorbar(xm, ym, xerr=[u.std() for u in pun],
                    yerr=[p.std() for p in pol], color=c, marker='o',
                    ms=5, capsize=2.5, lw=1.3, label=lab)
        for s, x, y in zip(sigmas, xm, ym):
            ax.annotate('%.0f nm' % perimeter_to_nm(s), (x, y),
                        xytext=(4, 5), textcoords='offset points',
                        fontsize=7, color=c)
    ax.axhline(POLARITY_THRESHOLD, ls='--', c='#888', lw=1)
    ax.set_xlabel('puncta per cell')
    ax.set_ylabel('Fz polarity')
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    ax.legend(fontsize=8, loc='best')
    ax.set_title('Polarity against punctum number (N=%d, %d seeds per point)'
                 % (args.N, len(seeds)), fontsize=9)
    fig.text(0.01, 0.01, 'Shaded: 2-3 puncta observed by confocal. Point labels: '
             'initial correlation length. Faint dots: individual seeds; '
             'bars: SD.', fontsize=6.5, color='#555', va='bottom')
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = os.path.join(
        args.outdir, 'sigma_polarity_puncta_N%d%s.png' % (args.N, _fig_suffix())
    )
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print('  written to %s\n  per-seed values in %s\n' % (out, path))


def pk_dose_test(args):
    """Punctum number and polarity against Pk dose in the CKIe model.

    Figure 6H uses a single dose (pk_dose=2); this sweep (Figure S6I,J) shows
    that punctum number peaks near 1.5-fold and falls at higher doses.

    Doses are relative to wild type and act on the initial free Pk density.
    Per-seed values are saved to pk_dose_<tag>.npz as each dose finishes and
    reused on a later run unless --force is given, so an interrupted run
    resumes.
    """
    import matplotlib.pyplot as plt
    doses = args.pk_doses or [1.0, 1.5, 2.0, 3.0, 4.0, 8.0]
    seeds = list(range(args.seed, args.seed + max(1, args.reps)))
    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, 'pk_dose_%s.npz' % _run_tag(args))
    store = {}
    if os.path.exists(path) and not args.force:
        with np.load(path) as f:
            store = {k: f[k] for k in f.files}
    print('\nPk dose response, CKIe model  (N=%d, tmax=%g, %d seeds)\n'
          % (args.N, args.tmax, len(seeds)))
    print('  %8s %28s %22s' % ('Pk dose', 'polarity', 'puncta'))
    for d in doses:
        kp, ku = 'pol_%g' % d, 'pun_%g' % d
        ks = 'seeds_%g' % d
        if kp in store and ks in store and \
                np.array_equal(store[ks], np.array(seeds)):
            pol, pun, tag = store[kp], store[ku], '  (cached)'
        else:
            res = run_many('CKI', dict(pk_dose=d), args.tmax, seeds,
                           args.N, args.jobs)
            pol = np.array([r[0] for r in res])
            pun = np.array([r[1] for r in res])
            store[kp], store[ku], store[ks] = pol, pun, np.array(seeds)
            np.savez(path, **store)              # save as we go
            tag = ''
        print('  %8.2g %9.3f +/- %.3f (med %.3f) %6.1f +/- %.1f (med %.1f)%s'
              % (d, pol.mean(), pol.std(), np.median(pol),
                 pun.mean(), pun.std(), np.median(pun), tag), flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    for ax, key, ylab in ((axes[0], 'pun', 'puncta per cell'),
                          (axes[1], 'pol', 'Fz polarity')):
        vals = [store['%s_%g' % (key, d)] for d in doses]
        for d, v in zip(doses, vals):
            ax.scatter(np.full(len(v), d), v, s=10, color='#666',
                       alpha=0.35, lw=0)
        ax.errorbar(doses, [v.mean() for v in vals],
                    yerr=[v.std() for v in vals], color='#d62728',
                    marker='o', ms=5, capsize=2.5, lw=1.3)
        ax.set_xscale('log')
        ax.set_xticks(doses)
        ax.set_xticklabels(['%g' % d for d in doses])
        # the log locator otherwise adds a stray minor label ("6 x 10^0")
        # between the plotted doses
        ax.minorticks_off()
        ax.set_xlabel('Pk dose (relative to wild type)')
        ax.set_ylabel(ylab)
        ax.set_ylim(bottom=0)
    axes[1].axhline(POLARITY_THRESHOLD, ls='--', c='#888', lw=1)
    fig.suptitle('Pk dose response, CKI$\\varepsilon$ model '
                 '(N=%d, %d seeds per dose)' % (args.N, len(seeds)),
                 fontsize=10)
    fig.text(0.01, 0.01, 'Faint dots: individual simulations; bars: SD. '
             'Dose 2 is the condition shown in Figure 6H.',
             fontsize=6.5, color='#555', va='bottom')
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    out = os.path.join(args.outdir, 'pk_dose_N%d%s.png' % (args.N, _fig_suffix()))
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print('\n  written to %s\n  per-seed values in %s\n' % (out, path))


# Parameter grids for --sensitivity. Each entry is a parameter of BASE and the
# values it takes; everything else stays at its production value. The MD grid
# covers the claim that no parameter choice gives multiple puncta together with
# strong polarity; the CKI grid covers the parameters FRAP does not constrain.
SENS_GRID = {
    'MD': [('init_free', [2.0, 4.0, 8.0]),
           ('tau',       [0.1, 0.3, 1.0]),
           ('eta',       [0.1, 0.3, 1.0]),
           ('gamma',     [0.05, 0.2, 0.8]),
           ('xi',        [0.05, 0.2, 0.8]),
           ('k_unbind',  [0.03, 0.1, 0.3]),
           ('D_f',       [0.01, 0.1, 1.0, 10.0]),
           ('n',         [1, 2, 3])],
    'CKI': [('K',        [0.7, 1.0, 2.0, 3.0]),
            ('nu',       [0.0, 0.005, 0.02, 0.05]),
            ('mu',       [0.05, 0.1, 0.2, 0.4]),
            ('D_CKIe',   [0.0, 2e-5, 2.2e-4, 2.2e-3, 7e-3, 2e-2]),
            ('k_on_W',   [0.003, 0.0065, 0.013]),
            ('sigma_ic', [0.005, 0.01, 0.02, 0.04]),
            ('n',        [1, 2, 3])],
}


def sensitivity_test(args):
    """Polarity against punctum number across one-parameter sweeps.

    Draws one panel per parameter on the axes of Figure 2H, with points
    coloured by parameter value, the polarity threshold marked and a band at
    two to three puncta per cell shaded. For the mutual destabilisation model
    (Figure S3C-J), no parameter value gives median polarity above 0.20 with
    two or more puncta; for the CKIe model (Figure S6A-G), polarity and
    puncta coexist across the range tested.

    Results are saved per parameter value, so an interrupted run resumes.
    """
    import matplotlib.pyplot as plt
    model = args.sensitivity
    grid = SENS_GRID[model]
    if args.sens_params:
        wanted = set(args.sens_params)
        grid = [g for g in grid if g[0] in wanted]
        if not grid:
            raise SystemExit('no matching parameters; choose from: %s'
                             % ', '.join(p for p, _ in SENS_GRID[model]))
    seeds = list(range(args.seed, args.seed + max(1, args.reps)))
    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir,
                        'sensitivity_%s_%s.npz' % (model, _run_tag(args)))
    store = {}
    if os.path.exists(path) and not args.force:
        with np.load(path) as f:
            store = {k: f[k] for k in f.files}

    label = 'mutual destabilisation' if model == 'MD' else 'CKIe'
    print('\nSensitivity of the %s model  (N=%d, tmax=%g, %d seeds)\n'
          % (label, args.N, args.tmax, len(seeds)))
    for name, values in grid:
        print('  %s' % name)
        print('  %12s %20s %18s %10s' % (name, 'polarity', 'puncta',
                                         'both?'))
        for val in values:
            kp, ku = '%s_%g_pol' % (name, val), '%s_%g_pun' % (name, val)
            if kp in store:
                pol, pun, tag = store[kp], store[ku], '  (cached)'
            else:
                res = run_many(model, {name: val}, args.tmax, seeds,
                               args.N, args.jobs)
                pol = np.array([r[0] for r in res])
                pun = np.array([r[1] for r in res])
                store[kp], store[ku] = pol, pun
                np.savez(path, **store)          # save as we go
                tag = ''
            # "both" = multiple puncta AND polarity above threshold
            both = 100.0 * ((pun >= 2) & (pol > POLARITY_THRESHOLD)).mean()
            print('  %12g %9.3f +/- %.3f %9.1f +/- %.1f %9.0f%%%s'
                  % (val, pol.mean(), pol.std(), pun.mean(), pun.std(),
                     both, tag), flush=True)
        print()

    ncol = min(4, len(grid))
    nrow = int(np.ceil(len(grid) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 3.3 * nrow),
                             squeeze=False, sharex=True, sharey=True)
    for ax, (name, values) in zip(axes.ravel(), grid):
        vals = np.array(values, float)
        logscale = vals.min() > 0 and vals.max() / vals.min() > 50
        norm = plt.Normalize(np.log10(vals).min(), np.log10(vals).max()) \
            if logscale else plt.Normalize(vals.min(), vals.max())
        ax.axvspan(2, 3, color='#2ca02c', alpha=0.12, lw=0)
        ax.axhline(POLARITY_THRESHOLD, ls='--', c='k', lw=1)
        for val in values:
            pol, pun = (
                store['%s_%g_pol' % (name, val)],
                store['%s_%g_pun' % (name, val)],
            )
            c = plt.cm.viridis(norm(np.log10(val) if logscale else val))
            ax.scatter(pun, pol, s=18, color=c, edgecolors='k', linewidths=0.2,
                       alpha=0.85, label='%g' % val)
        ax.set_title(name, fontsize=10)
        ax.legend(fontsize=6, title='value', title_fontsize=6, loc='upper right',
                  framealpha=0.85)
    for ax in axes.ravel()[len(grid):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel('puncta per cell')
    for row in axes:
        row[0].set_ylabel('Fz polarity')
    fig.suptitle('%s model: polarity against punctum number across parameters '
                 '(N=%d, %d simulations per value)' % (label, args.N, len(seeds)),
                 fontsize=11)
    fig.text(0.01, 0.01, 'Dashed line: polarity threshold 0.15. Shaded band: '
             '2-3 puncta per cell, as observed by confocal microscopy. Each '
             'point is one simulation.', fontsize=7, color='#555')
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out = os.path.join(
        args.outdir, 'sensitivity_%s_N%d%s.png' % (model, args.N, _fig_suffix())
    )
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print('  written to %s\n  per-seed values in %s\n' % (out, path))


def sweep_test(args):
    """Sweep one parameter and report polarity and puncta as distributions.

    Intended for the load-bearing parameters that FRAP cannot constrain: mu
    (CKIe destabilisation of Stbm), nu (CKIe stabilisation of Dsh), the
    feedback strengths, k_bind and k_unbind. The question for each is whether
    any value gives strong polarity together with several puncta.
    """
    import matplotlib.pyplot as plt
    name = args.sweep
    if not args.sweep_values:
        raise SystemExit('--sweep needs --sweep-values')
    if name not in BASE:
        raise SystemExit(
            'unknown parameter %r; choose from: %s'
            % (
                name,
                ', '.join(sorted(k for k in BASE if isinstance(BASE[k], (int, float)))),
            )
        )
    seeds = list(range(args.seed, args.seed + max(1, args.reps)))
    models = ['CKI'] if args.sweep_model == 'CKI' else \
             (['MD'] if args.sweep_model == 'MD' else ['MD', 'CKI'])
    store, path = {}, os.path.join(args.outdir, 'sweep_%s_N%d.npz' % (name, args.N))

    print('\nSweep of %s  (N=%d, tmax=%g, %d seeds, k_off_W=%.4f, '
          'D_CKIe=%.5f, mu=%.3f, k_basal=%.4f, k_on_W=%.4f)\n'
          % (name, args.N, args.tmax, len(seeds), BASE['k_off_W'],
             BASE['D_CKIe'], BASE['mu'], BASE['k_basal'], BASE['k_on_W']))
    for model in models:
        print('  %s' % ('mutual destabilisation' if model == 'MD' else 'CKIe'))
        print('  %12s %20s %18s %10s %14s %10s' % (name, 'polarity', 'puncta',
                                                   '%% < %.2f' % POLARITY_THRESHOLD,
                                                   'CKIe bound', 'Fz t1/2'))
        for val in args.sweep_values:
            res = run_many(model, {name: val}, args.tmax, seeds, args.N, args.jobs)
            pol = np.array([r[0] for r in res])
            pun = np.array([r[1] for r in res])
            store['%s_%g_pol' % (model, val)] = pol
            store['%s_%g_pun' % (model, val)] = pun
            cb = np.array([r[6].get('cki_bound', np.nan) for r in res])
            store['%s_%g_ckibound' % (model, val)] = cb
            th = np.array([r[7]['peak'] for r in res]) * time_unit_seconds() / 60
            store['%s_%g_fz_halflife_min' % (model, val)] = th
            np.savez(path, **store)          # save as we go
            cbs = ('%5.1f%% +/- %.1f' % (100 * np.nanmean(cb), 100 * np.nanstd(cb))
                   if np.isfinite(cb).any() else '-')
            print('  %12g %9.3f +/- %.3f %9.1f +/- %.1f %9.0f%% %14s %10.0f'
                  % (val, pol.mean(), pol.std(), pun.mean(), pun.std(),
                     100.0 * (pol < POLARITY_THRESHOLD).mean(), cbs, th.mean()),
                  flush=True)
        print()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for model, c in zip(models, ('#1f77b4', '#d62728')):
        vals = args.sweep_values
        pol = [store['%s_%g_pol' % (model, v)] for v in vals]
        pun = [store['%s_%g_pun' % (model, v)] for v in vals]
        lab = 'mutual destab.' if model == 'MD' else r'CKI$\varepsilon$'
        axes[0].errorbar(vals, [p.mean() for p in pol],
                         yerr=[p.std() for p in pol], marker='o', color=c,
                         label=lab, capsize=3)
        axes[1].errorbar(vals, [p.mean() for p in pun],
                         yerr=[p.std() for p in pun], marker='o', color=c,
                         label=lab, capsize=3)
    axes[0].axhline(POLARITY_THRESHOLD, ls='--', c='#888', lw=1)
    axes[0].set_ylabel('Fz polarity')
    axes[1].axhspan(2, 3, color='#2ca02c', alpha=0.12)
    axes[1].set_ylabel('puncta per cell')
    axes[1].text(0.02, 0.95, 'shaded: 2-3 puncta observed by confocal',
                 transform=axes[1].transAxes, fontsize=8, va='top')
    for ax in axes:
        ax.set_xlabel(name)
        ax.legend(fontsize=8)
    fig.suptitle('Sweep of %s (N=%d, %d seeds)' % (name, args.N, len(seeds)),
                 fontsize=11)
    fig.tight_layout()
    out = os.path.join(args.outdir, 'sweep_%s_N%d.png' % (name, args.N))
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print('  written to %s\n  per-seed values in %s\n' % (out, path))


def _job_tc(args):
    """One simulation, recording polarity and puncta at a series of times."""
    model, params, times, seed, N = args
    Grid.N, Grid.dx = N, 1.0 / N
    fields = FIELDS_MD if model == 'MD' else FIELDS_CKI
    rhs = rhs_MD if model == 'MD' else rhs_CKI
    sparsity = jac_sparsity(len(fields), fields,
                            nonlocal_W=bool(params.get('uniform_W')))
    sol = solve_ivp(rhs, (0, times[-1]), initial_state(params, model, seed),
                    args=(params,), method='BDF', jac_sparsity=sparsity,
                    rtol=1e-6, atol=1e-8, t_eval=times)
    if not sol.success:
        raise RuntimeError('integration failed: ' + sol.message)
    pol, pun = [], []
    for k in range(sol.y.shape[1]):
        v = unpack(sol.y[:, k], fields)
        fz = v['Fa1'] + v['C1']
        pol.append(polarity_index(fz))
        pun.append(count_puncta(fz))
    return np.array(pol), np.array(pun)


def time_course(args):
    """Polarity and punctum number against time, for the choice of t_max.

    Puncta form quickly and then merge slowly, so the system does not reach a
    steady state on developmental timescales and results must be reported at a
    stated time. This is the panel that justifies that choice.
    """
    import matplotlib.pyplot as plt
    times = np.unique(np.concatenate([
        np.array([0.0]), np.geomspace(50.0, args.tmax, 40)]))
    seeds = list(range(args.seed, args.seed + max(1, args.reps)))
    T0 = time_unit_seconds()
    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, 'time_course_%s.npz' % _run_tag(args))
    store = {'times': times}
    if os.path.exists(path) and not args.force:
        with np.load(path) as f:
            store = {k: f[k] for k in f.files}
    print('\nPolarity and puncta against time  (N=%d, %d seeds, to t=%g '
          '= %.0f h at T0 = %.2f s)\n' % (args.N, len(seeds), args.tmax,
                                          args.tmax * T0 / 3600, T0))
    for model in ('MD', 'CKI'):
        if ('%s_pol' % model) not in store:
            params = dict(BASE)
            tasks = [(model, params, times, s, args.N) for s in seeds]
            if min(args.jobs, len(tasks)) > 1:
                from multiprocessing import Pool
                with Pool(min(args.jobs, len(tasks))) as pool:
                    res = pool.map(_job_tc, tasks)
            else:
                res = [_job_tc(t) for t in tasks]
            store['%s_pol' % model] = np.array([r[0] for r in res])
            store['%s_pun' % model] = np.array([r[1] for r in res])
            np.savez(path, **store)
        pol, pun = store['%s_pol' % model], store['%s_pun' % model]
        print('  %s' % ('mutual destabilisation' if model == 'MD' else 'CKIe'))
        print('  %10s %8s %18s %14s' % ('t', 'hours', 'polarity', 'puncta'))
        for k in range(0, len(times), max(1, len(times) // 8)):
            print('  %10.0f %8.1f %9.3f +/- %.3f %6.1f +/- %.1f'
                  % (times[k], times[k] * T0 / 3600, pol[:, k].mean(),
                     pol[:, k].std(), pun[:, k].mean(), pun[:, k].std()))
        print(flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for model, c, lab in (('MD', '#1f77b4', 'mutual destabilisation'),
                          ('CKI', '#d62728', r'CKI$\varepsilon$ model')):
        hrs = times * T0 / 3600
        for ax, key in zip(axes, ('pol', 'pun')):
            y = store['%s_%s' % (model, key)]
            ax.plot(hrs, y.mean(axis=0), color=c, lw=1.5, label=lab)
            ax.fill_between(hrs, y.mean(axis=0) - y.std(axis=0),
                            y.mean(axis=0) + y.std(axis=0), color=c, alpha=0.15,
                            lw=0)
    for ax in axes:
        ax.set_xscale('log')
        ax.set_xlabel('time (h)')
        ax.axvline(20000 * T0 / 3600, ls='--', c='#444', lw=1)
    axes[0].set_ylabel('Fz polarity')
    axes[0].axhline(POLARITY_THRESHOLD, ls=':', c='#888', lw=1)
    axes[1].set_ylabel('puncta per cell')
    axes[1].axhspan(2, 3, color='#2ca02c', alpha=0.12, lw=0)
    axes[0].legend(fontsize=8)
    fig.text(0.01, 0.01, 'Dashed line: the reporting time used for the figures '
             '(t = 2e4, ~18 h). Shaded band on the right: 2-3 puncta by confocal. '
             'Bands: SD over %d seeds.' % len(seeds), fontsize=6.5, color='#555')
    fig.suptitle('Approach to the reported state (N=%d)' % args.N, fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = os.path.join(args.outdir, 'time_course_N%d%s.png' % (args.N, _fig_suffix()))
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print('  written to %s\n  per-seed values in %s\n' % (out, path))


def timescale_test(args):
    """Residence time of core complexes at puncta, against long-term FRAP.

    T0 is set from the Dsh/Pk fast phases (18.6 and 45.9 s), matched to their
    unmodulated off-rate k_off_VP. This is independent of the feedback and of
    the model, and at the defaults it gives T0 ~ 3.3 s, consistent with the
    CKIe slow phase (36 s) through k_off_W = 0.065. The model half-life of Fz
    at the punctum peak, converted with this T0, is then a prediction to set
    against the measured core slow phases (848-1944 s, mean 1374 s). nu is
    fitted so that the CKIe model matches the mean.
    """
    seeds = list(range(args.seed, args.seed + max(1, args.reps)))
    T0 = time_unit_seconds()
    slow = np.array(list(FRAP_SLOW_CORE.values()), float)
    print('\nCore-complex residence at puncta  (N=%d, tmax=%g, %d seeds, nu=%g, '
          'mu=%g, k_on_W=%g)' % (args.N, args.tmax, len(seeds), BASE['nu'],
                                  BASE['mu'], BASE['k_on_W']))
    print('  T0 from Dsh/Pk fast phases = %.2f s  (CKIe residence ln2/k_off_W '
          '= %.1f s; measured slow phase %.1f s)'
          % (T0, T0 * np.log(2) / BASE['k_off_W'], FRAP_SLOW_CKIE))
    print('  measured core slow phases: %.0f-%.0f s, mean %.0f s (%.1f min)\n'
          % (slow.min(), slow.max(), slow.mean(), slow.mean() / 60))
    print('  %-24s %22s %22s %9s %9s' % ('model', 'Fz t1/2 at peak',
                                         'trans-complex-weighted', 'polarity',
                                         'puncta'))
    for model, label in (('MD', 'mutual destabilisation'), ('CKI', 'CKIe')):
        res = run_many(model, {}, args.tmax, seeds, args.N, args.jobs)
        pk = np.array([r[7]['peak'] for r in res])
        wt = np.array([r[7]['weighted'] for r in res])
        pol = np.array([r[0] for r in res])
        pun = np.array([r[1] for r in res])
        print('  %-24s %7.0f (%5.1f min) %9.0f (%5.1f min) %9.3f %9.1f'
              % (label, pk.mean(), pk.mean() * T0 / 60, wt.mean(),
                 wt.mean() * T0 / 60, pol.mean(), pun.mean()), flush=True)
    print('\n  Target: %.0f model time units at the punctum peak (mean core slow '
          'phase / T0).\n' % (slow.mean() / T0))


def fractions_test(args):
    """Steady-state partition of each species, for comparison with FRAP.

    Runs both models at the parameters in force and reports, as mean +/- SD
    across seeds, the fraction of each species that is complex-associated.
    The FRAP reference values printed alongside are from Helen's two-phase
    fits (Table S1): the slow-phase share of recovering junctional
    fluorescence. They are not strictly like-for-like (see state_fractions),
    and the CKIe comparison is the meaningful one.
    """
    seeds = list(range(args.seed, args.seed + max(1, args.reps)))
    frap = dict(cki_bound=0.433, dsh_bound=0.563, pk_bound=0.385)
    labels = dict(cki_bound='CKIe complex-associated',
                  dsh_bound='Dsh complex-associated',
                  pk_bound='Pk complex-associated',
                  fz_assoc='Fz Fmi-assoc. or trans-complexed',
                  fz_inter='share of that Fz not in trans-complexes')
    print('\nSteady-state fractions  (N=%d, tmax=%g, %d seeds)'
          % (args.N, args.tmax, len(seeds)))
    print('CKIe parameters: k_off_W=%.4f, D_CKIe=%.5f (L=%.3f), mu=%.3f, '
          'nu=%.3f, k_basal=%.4f, k_on_W=%.4f\n'
          % (BASE['k_off_W'], BASE['D_CKIe'],
             np.sqrt(BASE['D_CKIe'] / BASE['k_off_W']), BASE['mu'],
             BASE['nu'], BASE['k_basal'], BASE['k_on_W']))
    store = {}
    for model, title in (('MD', 'mutual destabilisation'), ('CKI', 'CKIe')):
        res = run_many(model, {}, args.tmax, seeds, args.N, args.jobs)
        print('  %s  (polarity %.3f +/- %.3f)'
              % (title, np.mean([r[0] for r in res]), np.std([r[0] for r in res])))
        print('  %-42s %16s %22s' % ('', 'model', 'FRAP slow fraction'))
        for key in ('cki_bound', 'dsh_bound', 'pk_bound', 'fz_assoc', 'fz_inter'):
            vals = np.array([r[6][key] for r in res
                             if key in r[6] and np.isfinite(r[6][key])])
            if len(vals) == 0:
                continue
            store['%s_%s' % (model, key)] = vals
            ref = ('%.0f%%' % (100 * frap[key])) if key in frap else ''
            print('  %-42s %7.1f%% +/- %4.1f %22s'
                  % (labels[key], 100 * vals.mean(), 100 * vals.std(), ref),
                  flush=True)
        print()
    # Tagged by the parameter set, so that runs differing only in a parameter
    # (e.g. --hill-q 1 against the default 2) do not overwrite one another.
    path = os.path.join(args.outdir, 'fractions_%s.npz' % _run_tag(args))
    np.savez(path, seeds=seeds, **store)
    print('  FRAP column: slow-phase share of recovering junctional signal from')
    print('  the two-phase fits. David reads CKIe as ~30% of total once the')
    print('  ~10% FRAP-artefact loss is allowed for. The Fz rows have no direct')
    print('  FRAP counterpart; the Fz fast fraction is ~49%.')
    print('  per-seed values written to %s\n' % path)


def mesh_test(args):
    """Convergence check at the production settings, reported as distributions.

    Runs both models over a range of grid resolutions with several random
    initial conditions each. Initial conditions are generated on a fixed
    reference grid (see correlated_noise), so one seed gives the same physical
    field at every N and refinement comparisons are PAIRED: the seed-to-seed
    spread cancels out of the difference between resolutions.

    Per-seed values are saved to mesh_test.npz and plotted as distributions,
    since the spread between initial conditions is itself a result: what
    distinguishes the two models is not only mean polarity but how RELIABLY
    they polarise.
    """
    import matplotlib.pyplot as plt
    seeds = list(range(args.seed, args.seed + max(1, args.reps)))
    store, path = {}, os.path.join(args.outdir, 'mesh_test.npz')
    print('\nConvergence check (tmax=%g, sigma_ic=%.3f, n_ref=%d, %d seeds,'
          ' CKIe recruitment: %s)'
          % (args.tmax, BASE['sigma_ic'], BASE['n_ref'], len(seeds),
             BASE['cki_recruit']))
    print('CKIe parameters: k_off_W=%.4f, D_CKIe=%.5f (L=%.3f), mu=%.3f, '
          'nu=%.3f, k_basal=%.4f, k_on_W=%.4f\n'
          % (BASE['k_off_W'], BASE['D_CKIe'],
             np.sqrt(BASE['D_CKIe'] / BASE['k_off_W']), BASE['mu'],
             BASE['nu'], BASE['k_basal'], BASE['k_on_W']))
    for model, label in (('MD', 'mutual destabilisation, D_a=%g' % BASE['D_a']),
                         ('CKI', 'CKIe, D_CKIe=%g' % BASE['D_CKIe'])):
        print('  %s' % label, flush=True)
        print('  %6s %16s %16s %18s %10s %8s'
              % ('N', 'polarity', 'puncta', 'paired d(polarity)',
                 '%% < %.2f' % POLARITY_THRESHOLD, 'dx (nm)'), flush=True)
        base_pol = None
        for N in args.mesh_N:
            res = run_many(model, {}, args.tmax, seeds, N, args.jobs)
            pol = np.array([r[0] for r in res])
            pun = np.array([r[1] for r in res])
            store['%s_pol_%d' % (model, N)] = pol
            store['%s_pun_%d' % (model, N)] = pun
            np.savez(path, seeds=seeds, mesh_N=args.mesh_N, **store)  # as we go
            if base_pol is None:
                base_pol, paired = pol, '      (reference)'
            else:
                d = pol - base_pol
                paired = '%8.3f +/- %.3f' % (d.mean(), d.std())
            frac = 100.0 * (pol < POLARITY_THRESHOLD).mean()
            print('  %6d %7.3f +/- %.3f %7.1f +/- %.1f %18s %9.0f%% %8.0f'
                  % (N, pol.mean(), pol.std(), pun.mean(), pun.std(),
                     paired, frac, perimeter_to_nm(1.0 / N)), flush=True)
        print()
    np.savez(path, seeds=seeds, mesh_N=args.mesh_N, **store)

    # ---- distributions ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    rng = np.random.default_rng(0)
    for ax, (model, title) in zip(axes, (('MD', 'mutual destabilisation'),
                                         ('CKI', r'CKI$\varepsilon$'))):
        for k, N in enumerate(args.mesh_N, start=1):
            pol = store['%s_pol_%d' % (model, N)]
            ax.boxplot([pol], positions=[k], widths=0.5, showfliers=False,
                       medianprops=dict(color='k'))
            ax.plot(np.full(len(pol), k) + rng.uniform(-.14, .14, len(pol)),
                    pol, '.', color='#1f77b4', ms=6, alpha=0.7)
        ax.axhline(POLARITY_THRESHOLD, ls='--', c='#888', lw=1)
        ax.set_xticks(range(1, len(args.mesh_N) + 1))
        ax.set_xticklabels([str(N) for N in args.mesh_N])
        ax.set_xlabel('grid points N')
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel('Fz polarity')
    fig.suptitle('Polarity across initial conditions (%d seeds per point)'
                 % len(seeds), fontsize=11)
    fig.tight_layout()
    fig.savefig(
        os.path.join(args.outdir, 'mesh_distributions%s.png' % _fig_suffix()), dpi=200
    )
    plt.close(fig)

    # ---- model comparison at the finest grid ----
    Nf = args.mesh_N[-1]
    a, b = store['MD_pol_%d' % Nf], store['CKI_pol_%d' % Nf]
    fig, ax = plt.subplots(figsize=(5.0, 4.4))
    for k, v in enumerate((a, b), 1):
        ax.boxplot([v], positions=[k], widths=0.5, showfliers=False,
                   medianprops=dict(color='k'))
        ax.plot(np.full(len(v), k) + rng.uniform(-.14, .14, len(v)), v, '.',
                color='#1f77b4', ms=6, alpha=0.7)
    ax.axhline(POLARITY_THRESHOLD, ls='--', c='#888', lw=1)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['mutual destab.', r'CKI$\varepsilon$'])
    ax.set_ylabel('Fz polarity')
    ax.set_title('N = %d' % Nf, fontsize=10)
    fig.tight_layout()
    fig.savefig(
        os.path.join(args.outdir, 'model_comparison%s.png' % _fig_suffix()), dpi=200
    )
    plt.close(fig)

    print('  at N = %d: %.0f%% of mutual destabilisation runs fall below the '
          'threshold,\n  against %.0f%% of CKIe runs.\n'
          % (Nf, 100 * (a < POLARITY_THRESHOLD).mean(),
             100 * (b < POLARITY_THRESHOLD).mean()))
    print('  per-seed values written to %s' % path)
    print('  figures: mesh_distributions.png, model_comparison.png\n')


# ======================================================================
def main():
    """Parse arguments, apply parameter overrides and run the requested mode."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--panels', default='all',
                    help="comma-separated: 2H,2IL,6BD,6EM,6NO,width or 'all'")
    ap.add_argument('--reps', type=int, default=35,
                    help='simulations per condition (35 in the manuscript)')
    ap.add_argument('--jobs', type=int, default=1,
                    help='parallel processes. Memory scales with jobs x N: '
                         'scipy builds a sparse numerical Jacobian per worker, '
                         'which is several hundred MB at N=1600. Reduce jobs at '
                         'large N if you hit a MemoryError.')
    ap.add_argument('--N', type=int, default=200,
                    help='grid points (manuscript figures use 800)')
    ap.add_argument('--tmax', type=float, default=20000,
                    help='simulation end time (nondimensional)')
    ap.add_argument('--K', type=float, default=1.0,
                    help='half-saturation of G(y); chosen, not measured')
    ap.add_argument('--hill-q', type=int, default=None,
                    help='exponent of the Dsh/Pk recruitment function G(y); '
                         'default 2 (cooperative), 1 gives the '
                         'non-cooperative Michaelis-Menten form')
    ap.add_argument('--width-sigma', type=float, nargs='+', default=None,
                    help='initial correlation lengths for the width-vs-sigma '
                         'panel, e.g. 0.005 0.01 0.02; reuses values already '
                         'measured by --panels width --sigma X')
    ap.add_argument('--width-db', type=float, nargs='+', default=None,
                    help='D_a values for the punctum-width panel (default '
                         '1e-7 3e-7 1e-6 3e-6 1e-5 3e-5 1e-4)')
    ap.add_argument('--seed', type=int, default=0,
                    help='seed for the single-run diagnostic modes')
    ap.add_argument('--outdir', default='figures',
                    help='directory for figures and the .npz cache')
    ap.add_argument('--force', action='store_true', help='recompute, ignore cache')
    ap.add_argument('--smoke', action='store_true',
                    help='reps=2, tmax=2000, N=100: quick sanity check only')
    ap.add_argument('--cki-recruit', choices=['mass_action', 'hill'], default=None,
                    help="form of CKIe recruitment by Dsh (default mass_action)")
    ap.add_argument('--D_a', type=float, default=None,
                    help='diffusion coefficient of the Fmi-associated '
                         'intermediates (default 1e-7); 0 makes them strictly '
                         'immobile, like trans-complexes and complex-associated '
                         'Dsh and Pk')
    ap.add_argument('--D_CKIe', type=float, default=None,
                    help='diffusion coefficient of complex-associated CKIe '
                         '(default 0.00218, i.e. spread length 0.183 at k_off_W 0.065)')
    ap.add_argument('--k_basal', type=float, default=None,
                    help='basal recruitment rate for Dsh, Pk and CKIe '
                         '(default 0.01). This is a CONSTANT term added to the '
                         'recruitment rate, so it recruits everywhere '
                         'regardless of the local density of the recruiting '
                         'species, and therefore opposes the distal bias the '
                         'CKIe mechanism relies on. At 0.01 against k_on=0.07 '
                         'it is a floor of roughly 14%% of maximum recruitment. '
                         'Set 0 to remove it.')
    ap.add_argument('--k_on_W', type=float, default=None,
                    help='rate at which complex-associated Dsh recruits CKIe '
                         '(default 0.0065, giving 23%% of CKIe '
                         'complex-associated, against ~30%% by FRAP; the shared '
                         'k_on of 0.07 gives ~80%%). '
                         'Note k_basal alone sets a floor of about '
                         'k_basal/(k_basal+k_off_W), ~13%% at the defaults.')
    ap.add_argument('--mu', type=float, default=None,
                    help='strength of CKIe-mediated destabilisation of Stbm '
                         '(default 0.2, chosen to give 2-3 puncta per cell at the '
                         'FRAP-derived CKIe parameters; mu=0.4 gives ~1.4)')
    ap.add_argument('--nu', type=float, default=None,
                    help='strength of CKIe-mediated stabilisation of Dsh '
                         '(default 0.005, fitted to the core slow-phase '
                         'half-life; see --timescale)')
    ap.add_argument('--k_off_W', type=float, default=None,
                    help='off-rate of complex-associated CKIe (default 0.065, '
                         'from a measured residence half-life of ~36 s at '
                         'T0 = 3.3 s). Note '
                         'that changing this alone alters the CKIe spread '
                         'length sqrt(D_CKIe/k_off_W); to hold the spread '
                         'length fixed, set --D_CKIe to L^2 * k_off_W.')
    ap.add_argument('--spread-length', type=float, default=None,
                    help='set D_CKIe from the CKIe spread length L, as '
                         'D_CKIe = L^2 * k_off_W, using whichever k_off_W is '
                         'in force. Overrides --D_CKIe. L is in units of cell '
                         'perimeter; the default is L = 0.183, i.e. about '
                         '3.7 um.')
    ap.add_argument('--sigma', type=float, default=None,
                    help='correlation length of initial fluctuations, in units of '
                         'cell perimeter (default 0.01 = 200 nm); 0 = white noise')
    ap.add_argument('--mesh-N', type=int, nargs='+', default=[200, 400, 800],
                    help='grid resolutions for --mesh-test')
    ap.add_argument('--peak-prominence', type=float, default=None,
                    help='prominence a peak must have to count as a punctum, '
                         'as a fraction of the mean density of the profile '
                         '(default 0.02). 0 restores the earlier height-only '
                         'rule, which counts numerical ripples on a uniform '
                         'profile as puncta.')
    ap.add_argument('--time-course', action='store_true',
                    help='polarity and puncta against time in both models, '
                         'the basis for reporting at a fixed time rather than '
                         'at steady state (set the end time with --tmax)')
    ap.add_argument('--timescale', action='store_true',
                    help='residence half-life of core complexes at puncta in '
                         'both models, against the long-term FRAP slow phases '
                         '(the basis for the value of nu)')
    ap.add_argument('--fractions', action='store_true',
                    help='report steady-state bound fractions of CKIe, Dsh, '
                         'Pk and Fz, for comparison with FRAP')
    ap.add_argument('--mesh-test', action='store_true',
                    help='convergence of polarity and puncta with grid resolution')
    ap.add_argument('--ic-test', action='store_true',
                    help='compare white-noise and correlated initial conditions')
    ap.add_argument('--sensitivity', choices=['MD', 'CKI'], default=None,
                    help='one-parameter sweeps for the supplementary '
                         'sensitivity figure, plotted as polarity against '
                         'punctum number, one panel per parameter')
    ap.add_argument('--sens-params', nargs='+', default=None,
                    help='restrict --sensitivity to these parameters')
    ap.add_argument('--sweep', default=None,
                    help='name of a parameter in BASE to sweep, e.g. mu, nu, '
                         'tau, eta, gamma, xi, k_bind, k_unbind, n')
    ap.add_argument('--sweep-values', type=float, nargs='+', default=None,
                    help='values to sweep over')
    ap.add_argument('--sweep-model', choices=['MD', 'CKI', 'both'],
                    default='CKI', help='which model to sweep (default CKI)')
    ap.add_argument('--pk-dose', action='store_true',
                    help='punctum number and polarity against Pk dose in the '
                         'CKIe model: does the punctum increase keep going '
                         'with dose, as in vivo, or saturate?')
    ap.add_argument('--pk-doses', type=float, nargs='+', default=None,
                    help='doses for --pk-dose, relative to wild type '
                         '(default 1 1.5 2 3 4 8)')
    ap.add_argument('--sigma-test', action='store_true',
                    help='sensitivity to the initial correlation length')
    ap.add_argument('--sigmas', type=float, nargs='+', default=None,
                    help='correlation lengths for --sigma-test, in units of '
                         'cell perimeter (0.01 = 200 nm)')
    args = ap.parse_args()

    if args.smoke:
        args.reps, args.tmax, args.N = 2, 2000, 100
    BASE['K'] = args.K
    if args.hill_q is not None:
        BASE['hill_q'] = args.hill_q
    if args.sigma is not None:
        BASE['sigma_ic'] = args.sigma
    if args.cki_recruit is not None:
        BASE['cki_recruit'] = args.cki_recruit
    if args.D_a is not None:
        BASE['D_a'] = args.D_a
    if args.D_CKIe is not None:
        BASE['D_CKIe'] = args.D_CKIe
    if args.k_basal is not None:
        BASE['k_basal'] = args.k_basal
    if args.mu is not None:
        BASE['mu'] = args.mu
    if args.k_on_W is not None:
        BASE['k_on_W'] = args.k_on_W
    if args.nu is not None:
        BASE['nu'] = args.nu
    if args.peak_prominence is not None:
        globals()['PEAK_PROMINENCE_FRAC'] = args.peak_prominence
    if args.k_off_W is not None:
        BASE['k_off_W'] = args.k_off_W
    if args.spread_length is not None:
        BASE['D_CKIe'] = args.spread_length**2 * BASE['k_off_W']
        print('spread length L = %.3f with k_off_W = %.4f  ->  D_CKIe = %.5f'
              % (args.spread_length, BASE['k_off_W'], BASE['D_CKIe']))
    os.makedirs(args.outdir, exist_ok=True)

    import matplotlib
    matplotlib.use('Agg')

    if args.time_course:
        time_course(args)
        return
    if args.timescale:
        timescale_test(args)
        return
    if args.fractions:
        fractions_test(args)
        return
    if args.mesh_test:
        mesh_test(args)
        return
    if args.ic_test:
        ic_test(args)
        return
    if args.sigma_test:
        sigma_test(args)
        return
    if args.pk_dose:
        pk_dose_test(args)
        return
    if args.width_sigma:
        width_sigma(args)
        return
    if args.sensitivity:
        sensitivity_test(args)
        return
    if args.sweep:
        sweep_test(args)
        return

    want = ([p.strip() for p in args.panels.split(',')]
            if args.panels != 'all'
            else ['2H', '2IL', '6BD', '6EM', '6NO', 'width'])
    print('regenerating: %s  (reps=%d, N=%d, tmax=%g, K=%g)'
          % (', '.join(want), args.reps, args.N, args.tmax, args.K))

    if '2H' in want:
        print('Figure 2H')
        panel_2H(args)
    if '2IL' in want:
        print('Figure 2I-L')
        panel_2IL(args)
    for w, tag in (('6BD', 'BD'), ('6EM', 'EM'), ('6NO', 'NO')):
        if w in want:
            print('Figure 6%s' % tag)
            panel_6(args, tag)
    if 'width' in want:
        print('Punctum width scaling')
        panel_width(args)

    print('\nwritten to %s/' % args.outdir)
    print('Checks worth running: --ic-test (grid independence), '
          '--sigma-test (sensitivity to the assumed correlation length), '
          '--mesh-test (refinement).')


if __name__ == '__main__':
    main()
