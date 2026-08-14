"""
Shared publication-quality plot style, constants, and helpers.

All plotting scripts should import from here rather than redefining these values.
"""

# ── Recurrence colormap ──────────────────────────────────────────────────────

import logging

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

logger = logging.getLogger(__name__)

RECURRENCE_CMAP = plt.cm.viridis


def recurrence_color(score, vmin=0.0, vmax=1.0):
    """Map a continuous recurrence score to a colour via *RECURRENCE_CMAP*."""
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    return RECURRENCE_CMAP(norm(score))


def make_recurrence_colorbar(ax, vmin=0.0, vmax=1.0, label="Recurrence score"):
    """Add a continuous recurrence-score colorbar to *ax*."""
    sm = plt.cm.ScalarMappable(
        cmap=RECURRENCE_CMAP, norm=mcolors.Normalize(vmin=vmin, vmax=vmax),
    )
    sm.set_array([])
    cbar = ax.figure.colorbar(sm, ax=ax, label=label, shrink=0.8)
    return cbar


# Re-export so callers can do: from utils.plot_style import TR_SECONDS
TR_SECONDS = 1.49


# ── Subject encoding ─────────────────────────────────────────────────────────
# Project-wide rule: subject identity is usually encoded by axis position
# (small multiples) or marker shape, not color. Network colors already occupy
# most of the usable hue range, so subject-as-color conflicts with network
# overlays. Use these constants instead.

SUBJECT_NEUTRAL = "#4A4A4A"
SUBJECT_ACCENT = "#D62728"

SUBJECT_MARKERS = {
    "sub-01": "o",
    "sub-02": "s",
    "sub-03": "^",
    "sub-04": "D",
    "sub-05": "v",
    "sub-06": "P",
}


# ── Network constants ────────────────────────────────────────────────────────

NETWORK_ORDER = [
    "Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default",
    "BG", "Midbrain-DA", "Midbrain-Diencephalic", "Thalamus", "Hipp/Amyg",
    "Cerebellum",
]

NETWORK_COLORS = {
    # Cortical: colorblind-safe Okabe–Ito hues (2026-05-26). The canonical Yeo-7
    # colors (Vis purple, DMN red, etc.) are field-standard but NOT CB-safe - the
    # Vis-purple / SalVentAttn-magenta pair and the green/orange/red trio collide
    # under deuteranopia/protanopia. At most 7 cortical networks ever appear as
    # *dominant* states in any subject (subcortical never wins the per-state max,
    # an amplitude-bias property), so 7 CB-distinct hues suffice. Mnemonics kept
    # where Okabe–Ito allows: Vis purple-pink, SomMot blue, DorsAttn green,
    # Cont orange, Default red/vermillion.
    "Vis": "#CC79A7", "SomMot": "#56B4E9", "DorsAttn": "#009E73",
    "SalVentAttn": "#0072B2", "Limbic": "#F0E442", "Cont": "#E69F00",
    "Default": "#D55E00",
    # Subcortical subgroups (v2 canonical BG circuit, 2026-05-11): a desaturated
    # warm-brown ramp distinguished by lightness (the one channel preserved across
    # all CVD types), kept muted so it reads as a single "subcortical" family and
    # never competes with the vivid cortical hues. Only surfaced in the F1 Sankey
    # (~5% of states); never a dominant node in the F2 transition graph.
    "BG": "#98775B", "Midbrain-DA": "#5E3F26",
    "Midbrain-Diencephalic": "#6F4E37",
    "Thalamus": "#B0997D", "Hipp/Amyg": "#7D5A3C", "Cerebellum": "#C9BBA8",
}

# Canonical display names for manuscript figures (legends, axis labels, tick
# labels). Data keys stay as the short NETWORK_ORDER strings; only the rendered
# text uses these. "Readable but compact": expand the cryptic cortical
# abbreviations, keep "Attn" short, leave subcortical groups as-is.
# Use display_network() to resolve.
NETWORK_DISPLAY = {
    "Vis": "Visual",
    "SomMot": "Somatomotor",
    "DorsAttn": "Dorsal Attn",
    "SalVentAttn": "Ventral Attn",
    "Limbic": "Limbic",
    "Cont": "Control",
    "Default": "Default",
    "BG": "BG",
    "Midbrain-DA": "Midbrain-DA",
    "Midbrain-Diencephalic": "Midbrain-Dien",
    "Thalamus": "Thalamus",
    "Hipp/Amyg": "Hipp/Amyg",
    "Cerebellum": "Cerebellum",
}


def display_network(net):
    """Map an internal network key to its manuscript display name."""
    return NETWORK_DISPLAY.get(net, net)


# Subcortical structure -> bin mapping. Canonical basal-ganglia circuit per
# Alexander, DeLong & Strick (1986) Annu Rev Neurosci 9:357-381; ventral-BG
# extension per Haber & Knutson (2010) Neuropsychopharmacology 35:4-26.
#
# v2 (2026-05-11) rationale per CIT168 structure:
#   - Pu, Ca, NAC -> BG: striatum (Haber & Knutson 2010 for NAC limbic-BG).
#   - GPe, GPi -> BG: pallidum (canonical BG output for GPi).
#   - STH -> BG: subthalamic nucleus, indirect/hyperdirect-pathway core
#     (Nambu, Tokuno & Takada 2002 Neurosci Res 43:111-117).
#   - SNr -> BG: substantia nigra reticulata, BG output paired with GPi
#     (Alexander-DeLong-Strick 1986; Lanciego, Luquin & Obeso 2012
#     Cold Spring Harb Perspect Med 2:a009621).
#   - VeP -> BG: ventral pallidum, limbic-BG (Haber & Knutson 2010).
#   - SNc_PBP_VTA -> Midbrain-DA (new bin): mesolimbic/nigrostriatal DA
#     midbrain cluster (CIT168 atlas fuses these); distinct from BG
#     circuit (Haber 2014 Dialogues Clin Neurosci 16:317-330; Morales &
#     Margolis 2017 Nat Rev Neurosci 18:73-85).
#   - EXA -> Hipp/Amyg: extended amygdala (CeA + BNST + sublenticular
#     SI) is amygdaloid, not BG (Heimer & Van Hoesen 2006 Neurosci
#     Biobehav Rev 30:126-147; Alheid 2003 Ann N Y Acad Sci 985:185-205).
#   - RN, HN, HTH, MN -> Midbrain-Diencephalic (renamed from Brainstem):
#     only RN is true midbrain; HN is epithalamic (Hikosaka 2010 Nat Rev
#     Neurosci 11:503-513); HTH/MN are hypothalamic/diencephalic.
#     "Brainstem" was anatomically incorrect for the residual bin.
#
# script/03b_pca_loadings.py keeps a parallel local copy of this dict for its
# Yeo-17 cortical convention; keep both in sync when editing.
_SUBCORT_GROUPS = {
    # Basal ganglia (16 parcels bilateral)
    'Pu': 'BG', 'Ca': 'BG', 'NAC': 'BG',
    'GPe': 'BG', 'GPi': 'BG',
    'STH': 'BG', 'SNr': 'BG', 'VeP': 'BG',
    # Midbrain dopaminergic (2 parcels bilateral; CIT168 fuses SNc+PBP+VTA)
    'SNc_PBP_VTA': 'Midbrain-DA',
    # Midbrain-diencephalic residual (8 parcels bilateral)
    'RN': 'Midbrain-Diencephalic',
    'HN': 'Midbrain-Diencephalic',
    'HTH': 'Midbrain-Diencephalic',
    'MN': 'Midbrain-Diencephalic',
    # Thalamus (14 parcels bilateral)
    'Pulvinar': 'Thalamus', 'Anterior': 'Thalamus', 'Medio_Dorsal': 'Thalamus',
    'Ventral_Latero_Dorsal': 'Thalamus',
    'Central_Lateral-Lateral_Posterior-Medial_Pulvinar': 'Thalamus',
    'Ventral_Anterior': 'Thalamus', 'Ventral_Latero_Ventral': 'Thalamus',
    # Hippocampus + amygdala + extended amygdala (6 parcels bilateral)
    'Hippocampus': 'Hipp/Amyg', 'Amygdala': 'Hipp/Amyg',
    'EXA': 'Hipp/Amyg',
}


def assign_network(label):
    """Map a parcel label to its subcortical network group.

    Returns the subcortical group name for subcortical/cerebellar parcels,
    or None for cortical parcels (use the network_label column instead).
    """
    if label.startswith('Cerebellar'):
        return 'Cerebellum'
    for sep in ['-', '_']:
        if sep in label:
            idx = label.index(sep)
            prefix = label[:idx]
            if prefix in ('LH', 'RH'):
                structure = label[idx + 1:]
                if structure in _SUBCORT_GROUPS:
                    return _SUBCORT_GROUPS[structure]
    return None


def load_parcel_networks(parcellation):
    """Build parcel-to-network mapping from atlas labels.

    Returns list of network names (length = n_parcels), or None on failure.
    """
    try:
        from utils.viz_yabplot import load_parcel_labels
        label_df = load_parcel_labels(parcellation)
    except Exception as e:
        logger.warning("Could not load parcel labels: %s - network analysis disabled", e)
        return None

    n_parcels = len(label_df)
    parcel_networks = []
    for idx in range(n_parcels):
        row = label_df[label_df["index"] == idx + 1]  # 1-based index
        if len(row) == 0:
            parcel_networks.append("Unknown")
            continue
        label = row.iloc[0]["label"]
        net = assign_network(label)
        if net is None:
            net = row.iloc[0].get("network_label", "Unknown")
        parcel_networks.append(net)

    return parcel_networks


def compute_dominant_networks(state_means, active_states, parcel_networks,
                              include_sign=False):
    """Assign each active state to its dominant network.

    For each state, compute mean |activation| per network and pick the max.

    Args:
        state_means: (n_states, n_parcels) array
        active_states: array of active state indices
        parcel_networks: list of network names per parcel
        include_sign: if True, return (network_name, polarity) tuples where
            polarity is "+" (activation) or "-" (deactivation) based on the
            signed mean activation of the dominant network's parcels.

    Returns:
        dict: state_index -> network name (str) if include_sign is False,
              state_index -> (network name, polarity) if include_sign is True
    """
    parcel_nets = np.array(parcel_networks)
    dominant = {}
    for s in active_states:
        abs_mean = np.abs(state_means[s])
        net_activation = {}
        for net in NETWORK_ORDER:
            mask = parcel_nets == net
            if np.any(mask):
                net_activation[net] = float(np.mean(abs_mean[mask]))
        if net_activation:
            best_net = max(net_activation, key=net_activation.get)
            if include_sign:
                best_mask = parcel_nets == best_net
                signed_mean = float(np.mean(state_means[s][best_mask]))
                polarity = "+" if signed_mean >= 0 else "-"
                dominant[int(s)] = (best_net, polarity)
            else:
                dominant[int(s)] = best_net
        else:
            dominant[int(s)] = ("Unknown", "+") if include_sign else "Unknown"
    return dominant


def format_signed(value, spec=".2f"):
    """Sign-aware numeric label for annotations; 'n/a' for None (degenerate stat)."""
    if value is None:
        return "n/a"
    return format(value, "+" + spec)


# ── Style application ─────────────────────────────────────────────────────────

def apply_publication_style() -> None:
    """Apply canonical publication-quality matplotlib settings.

    Call this once at the top of every script/notebook cell before any
    plt.figure() call.  Sets backend to Agg (headless-safe), resets to the
    'default' style, then applies the shared rcParams.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("default")
    plt.rcParams.update({
        "figure.dpi":       300,
        "font.size":        10,
        "svg.fonttype":     "none",
        "figure.titlesize": 12,
        "axes.titlesize":   10,
        "axes.labelsize":   9,
        "ytick.labelsize":  8,
        "xtick.labelsize":  8,
        "axes.facecolor":   "white",
        "figure.facecolor": "white",
    })


