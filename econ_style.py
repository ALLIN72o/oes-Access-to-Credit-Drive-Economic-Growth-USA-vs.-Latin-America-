"""Economist-style chart furniture for matplotlib.

Reproduces the house style of The Economist's data charts: the red flag in the
top-left corner, left-aligned title/subtitle block, horizontal-only gridlines,
y-axis labels on the right, and a source line in the bottom gutter.

Nothing here is specific to this project — `analysis.py` imports it and supplies
the data. Usage:

    fig, ax = plt.subplots(figsize=(9, 5.5))
    econ_style.apply()                       # once, before creating figures
    ...plot onto ax...
    econ_style.axes(ax, units="% of GDP")
    econ_style.finish(fig, title=..., subtitle=..., source=...)
"""
from matplotlib import font_manager, rcParams
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

# --- The Economist's data palette -------------------------------------------
# Hex values from the paper's published style guide. ECON_RED is reserved for
# the corner flag and for emphasis; it is not a general-purpose series color,
# though with only two series (USA vs. LatAm) red/blue is the house pairing.
ECON_RED = "#E3120B"
BLUE = "#006BA2"
CYAN = "#3EBCD2"
TEAL = "#379A8B"
YELLOW = "#EBB434"
PURPLE = "#9A607F"
TAN = "#D1B07C"
GREY = "#758D99"

# Ink and furniture.
INK = "#121212"
INK_SOFT = "#5A5A5A"
GRID = "#D9D9D9"
SURFACE = "#FFFFFF"

# Two-series assignment used throughout this project. Validated with the
# dataviz palette checker against a white surface: worst pair ΔE 20.4 under
# simulated protanopia, 34.6 for normal vision — comfortably above the floors.
SERIES = {"USA": BLUE, "LatAm": ECON_RED}

# The Economist sets its charts in Econ Sans Condensed, which is proprietary.
# Fall back through the closest condensed grotesques likely to be installed.
FONT_STACK = [
    "Econ Sans Condensed", "EconSansCndReg", "Avenir Next Condensed",
    "Arial Narrow", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans",
]


def available_font() -> str:
    """First font in the stack that is actually installed (for reporting)."""
    installed = {f.name for f in font_manager.fontManager.ttflist}
    return next((f for f in FONT_STACK if f in installed), "DejaVu Sans")


def apply():
    """Set global rcParams. Call once before building any figure."""
    rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": FONT_STACK,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.titlesize": 11,
        "font.size": 10.5,
        "legend.frameon": False,
        "axes.unicode_minus": False,   # use ASCII hyphen; the house style is a short minus
    })


def axes(ax, units: str = "", y_right: bool = True, zero_line: bool = True,
         x_grid: bool = False):
    """Strip an axes down to the house style.

    The Economist draws the plotting area as nothing but horizontal rules: no
    box, no y-axis spine, no tick marks on the value axis. Only the x-axis
    baseline is drawn, in full-strength ink.
    """
    ax.set_facecolor(SURFACE)
    for side in ("top", "left", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK)
    ax.spines["bottom"].set_linewidth(0.9)

    ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
    if x_grid:
        ax.grid(axis="x", color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)

    if y_right:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
    ax.tick_params(axis="y", length=0, labelsize=9.5, pad=4)
    ax.tick_params(axis="x", length=3, width=0.9, direction="out", labelsize=9.5,
                   color=INK)

    if zero_line and ax.get_ylim()[0] < 0 < ax.get_ylim()[1]:
        ax.axhline(0, color=INK, linewidth=0.9, zorder=2)

    if units:
        # Units sit above the plot area, left-aligned, instead of a rotated
        # y-axis label — the house convention.
        ax.text(0, 1.02, units, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=9.5, color=INK)


def units_right(ax, units: str):
    """Unit caption above the right-hand axis (for the dual-scale chart)."""
    ax.text(1, 1.02, units, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9.5, color=INK)


def legend(ax, entries, loc="upper left", ncol=1, **kw):
    """Flat legend with round swatches rather than line samples.

    `entries` is a list of (label, color) pairs.
    """
    handles = [Line2D([], [], color=c, marker="o", linestyle="none", markersize=7,
                      label=lab) for lab, c in entries]
    leg = ax.legend(handles=handles, loc=loc, ncol=ncol, frameon=False,
                    handletextpad=0.4, columnspacing=1.2, fontsize=9.5,
                    borderpad=0, **kw)
    for text in leg.get_texts():
        text.set_color(INK)
    return leg


def finish(fig, title: str, subtitle: str = "", source: str = "",
           note: str = "", top_in: float = 1.15, bottom_in: float = 0.80,
           left_in: float = 0.62, right_in: float = 0.70):
    """Add the flag, title block and source gutter, and reserve space for them.

    Margins are specified in inches rather than figure fractions so that the
    title block keeps a constant physical size across differently sized figures.
    """
    w, h = fig.get_figwidth(), fig.get_figheight()
    fig.subplots_adjust(top=1 - top_in / h, bottom=bottom_in / h,
                        left=left_in / w, right=1 - right_in / w)

    x = left_in / w / 2 if left_in < 0.3 else 0.010
    # Red flag: ~0.55in wide, ~0.10in tall, flush to the top-left corner.
    fig.add_artist(Rectangle((x, 1 - 0.20 / h), 0.55 / w, 0.10 / h,
                             facecolor=ECON_RED, edgecolor="none"))
    fig.text(x, 1 - 0.46 / h, title, ha="left", va="center", fontsize=14.5,
             fontweight="bold", color=INK)
    if subtitle:
        fig.text(x, 1 - 0.72 / h, subtitle, ha="left", va="center", fontsize=11,
                 color=INK)

    gutter = []
    if note:
        gutter.append(note)
    if source:
        gutter.append(f"Source: {source}")
    if gutter:
        fig.text(x, 0.20 / h, "\n".join(gutter), ha="left", va="center",
                 fontsize=8.5, color=INK_SOFT, linespacing=1.4)
