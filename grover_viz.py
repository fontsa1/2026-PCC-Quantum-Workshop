"""
grover_viz.py -- helper functions for the Week 2 Grover notebooks (Days 6 and 7).

Students do not need to read or edit this file. It does three jobs:

  1. Builds Grover circuits in Qiskit (phase oracles, the diffuser, and a
     Caesar-cipher oracle that "encrypts in superposition").
  2. Uses Qiskit's Statevector to compute the exact quantum state after every
     oracle and diffuser step.
  3. Animates those states in the style of 3Blue1Brown's Grover video
     (https://www.youtube.com/watch?v=RQWpF2Gb-gU):
        left   -- the state as an arrow in the "key vs. everything else" plane,
                  where every step is a reflection and every round is a 2*theta turn
        right  -- signed amplitudes (with their mean) and probabilities per answer

Every frame of the animation comes from a real Qiskit Statevector. The in-between
frames of each flip are drawn by smoothly reflecting from one Qiskit state to
the next, the way the video flips the plane over a mirror line.
"""

from math import asin, sqrt, pi, degrees, atan2, cos, floor

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Arc, Circle, FancyArrowPatch
from IPython.display import HTML

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error


# Colors borrowed from the manim palette 3Blue1Brown uses in the video
BG = "#000000"
WHITE = "#FFFFFF"
GREY = "#777777"
DARK_GREY = "#333333"
BLUE = "#58C4DD"
TEAL = "#5CD0B3"
YELLOW = "#FFFF00"
RED = "#FC6255"
GREEN = "#83C167"

BASIS_GATES = ["cx", "sx", "x", "rz"]

plt.rcParams["animation.embed_limit"] = 80  # MB, big enough for 5-6 qubit animations


# ---------------------------------------------------------------------------
# 1. Building Grover circuits
# ---------------------------------------------------------------------------

def _check_bitstrings(n_qubits, marked):
    if isinstance(marked, str):
        marked = [marked]
    for m in marked:
        if len(m) != n_qubits or any(b not in "01" for b in m):
            raise ValueError(
                f"'{m}' is not a valid answer for {n_qubits} qubits. "
                f"Use exactly {n_qubits} characters, each 0 or 1 (example: '{'1' * n_qubits}')."
            )
    return list(marked)


def _mcz(qc, qubits):
    """Multi-controlled Z: flips the sign of |11...1> on the given qubits."""
    if len(qubits) == 1:
        qc.z(qubits[0])
        return
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])


def phase_oracle(n_qubits, marked):
    """
    Oracle that flips the sign (phase) of each marked answer.
    `marked` is a bitstring like '101' (same order as histogram labels) or a list of them.
    """
    marked = _check_bitstrings(n_qubits, marked)
    oracle = QuantumCircuit(n_qubits, name="Oracle")
    for m in marked:
        # Qiskit counts qubits right-to-left, so reverse the string.
        zeros = [q for q, bit in enumerate(reversed(m)) if bit == "0"]
        if zeros:
            oracle.x(zeros)  # turn the marked answer into |11...1>
        _mcz(oracle, list(range(n_qubits)))
        if zeros:
            oracle.x(zeros)  # undo
    return oracle


def diffuser(n_qubits):
    """Grover diffuser: reflects every amplitude about the mean amplitude."""
    diff = QuantumCircuit(n_qubits, name="Diffuser")
    diff.h(range(n_qubits))
    diff.x(range(n_qubits))
    _mcz(diff, list(range(n_qubits)))
    diff.x(range(n_qubits))
    diff.h(range(n_qubits))
    # H-X-MCZ-X-H is minus the textbook reflection. A global -1 can never be
    # measured, but it would draw every amplitude bar upside down, so cancel it.
    diff.global_phase = pi
    return diff


def _to_oracle(n_qubits, oracle):
    if isinstance(oracle, QuantumCircuit):
        if oracle.num_qubits != n_qubits:
            raise ValueError(f"The oracle uses {oracle.num_qubits} qubits but n_qubits = {n_qubits}.")
        return oracle
    return phase_oracle(n_qubits, oracle)


def marked_states(oracle):
    """
    Return the answers an oracle marks, found by running it on every answer at once.
    (Only the *pictures* use this, to know which bar to color yellow. Grover's
    algorithm itself never looks inside the oracle.)
    """
    n = oracle.num_qubits
    start = QuantumCircuit(n)
    start.h(range(n))
    amps = np.real(Statevector(start).evolve(oracle).data)
    return [format(i, f"0{n}b") for i in np.flatnonzero(amps < 0)]


def best_iterations(n_qubits, n_marked=1):
    """Number of Grover rounds that gets closest to 100%: about (pi/4) * sqrt(N / M)."""
    theta = asin(sqrt(n_marked / 2 ** n_qubits))
    return max(1, floor(pi / (4 * theta)))


def grover_circuit(n_qubits, oracle, iterations=None, measure=True):
    """
    Full Grover circuit: Hadamards, then (oracle + diffuser) repeated, then measure.
    `oracle` can be a bitstring like '101', a list of bitstrings, or an oracle QuantumCircuit.
    """
    oracle = _to_oracle(n_qubits, oracle)
    if iterations is None:
        iterations = best_iterations(n_qubits, len(marked_states(oracle)))
    qc = QuantumCircuit(n_qubits, n_qubits if measure else 0)
    qc.h(range(n_qubits))
    diff = diffuser(n_qubits)
    for _ in range(iterations):
        qc.compose(oracle, inplace=True, wrap=True)
        qc.compose(diff, inplace=True, wrap=True)
    if measure:
        qc.measure(range(n_qubits), range(n_qubits))
    return qc


def grover_snapshots(n_qubits, oracle, iterations):
    """
    Run Grover one step at a time with Qiskit's Statevector.
    Returns a list of (step_name, round_number, Statevector).
    """
    oracle = _to_oracle(n_qubits, oracle)
    diff = diffuser(n_qubits)
    start = QuantumCircuit(n_qubits)
    start.h(range(n_qubits))
    state = Statevector(start)
    snaps = [("start", 0, state)]
    for k in range(1, iterations + 1):
        state = state.evolve(oracle)
        snaps.append(("oracle", k, state))
        state = state.evolve(diff)
        snaps.append(("diffuser", k, state))
    return snaps


# ---------------------------------------------------------------------------
# 2. Caesar cipher on a 32-symbol alphabet (32 = 2**5, so keys fit in 5 qubits)
# ---------------------------------------------------------------------------

ALPHABET32 = "ABCDEFGHIJKLMNOPQRSTUVWXYZ .,!?'"


def caesar_encrypt(text, key):
    out = []
    for ch in text.upper():
        if ch in ALPHABET32:
            out.append(ALPHABET32[(ALPHABET32.index(ch) + key) % 32])
        else:
            out.append(ch)
    return "".join(out)


def caesar_decrypt(text, key):
    return caesar_encrypt(text, -key)


def add_constant(n_qubits, value, name=None):
    """
    Quantum adder for a fixed number: |k>  ->  |k + value  (mod 2**n)>.
    Adding 2**j is the same as counting up by one on qubits j, j+1, ..., n-1, and
    counting up is a chain of controlled-NOTs: a bit flips when every bit below it is 1.
    """
    value %= 2 ** n_qubits
    qc = QuantumCircuit(n_qubits, name=name or f"+{value}")
    for j in range(n_qubits):
        if (value >> j) & 1:
            reg = list(range(j, n_qubits))
            for i in range(len(reg) - 1, 0, -1):
                qc.mcx(reg[:i], reg[i])
            qc.x(reg[0])
    return qc


def caesar_oracle(plain_letter, cipher_letter):
    """
    Grover oracle for a known-plaintext attack on the Caesar cipher.

    The 5 qubits hold a guess for the key. The oracle:
      1. encrypts plain_letter with the guessed key (adds the letter's number to the key),
      2. flips the sign if the result equals cipher_letter,
      3. un-encrypts, so the register holds the key guess again.
    It only ever uses public information (one letter we know and the matching ciphertext letter).
    """
    p = ALPHABET32.index(plain_letter.upper())
    c = ALPHABET32.index(cipher_letter.upper())
    encrypt = add_constant(5, p, name=f"Encrypt '{plain_letter.upper()}'")
    check = phase_oracle(5, format(c, "05b"))
    check.name = f"Is it '{cipher_letter.upper()}'?"
    undo = encrypt.inverse()
    undo.name = "Undo encrypt"

    oracle = QuantumCircuit(5, name="Caesar oracle")
    oracle.compose(encrypt, inplace=True, wrap=True)
    oracle.compose(check, inplace=True, wrap=True)
    oracle.compose(undo, inplace=True, wrap=True)
    return oracle


# ---------------------------------------------------------------------------
# 3. The 3Blue1Brown-style picture
# ---------------------------------------------------------------------------

def _marked_mask(n_qubits, marked):
    if isinstance(marked, QuantumCircuit):
        marked = marked_states(marked)
    marked = _check_bitstrings(n_qubits, marked)
    mask = np.zeros(2 ** n_qubits, dtype=bool)
    for m in marked:
        mask[int(m, 2)] = True
    return mask


def _real_amps(state):
    if isinstance(state, QuantumCircuit):
        state = Statevector(state)
    data = np.asarray(state.data)
    if np.max(np.abs(data.imag)) > 1e-9:
        raise ValueError("This picture only works for real amplitudes (Grover's states always are).")
    return data.real.copy()


def _plane_xy(amps, mask):
    """Coordinates of the state in the plane spanned by |others> (x) and |key> (y)."""
    m, rest = mask.sum(), (~mask).sum()
    y = amps[mask].sum() / sqrt(m) if m else 0.0
    x = amps[~mask].sum() / sqrt(rest) if rest else 0.0
    return x, y


class _GroverPicture:
    """One figure (geometry + amplitudes + probabilities) that can be redrawn per frame."""

    def __init__(self, n_qubits, mask, labels=None, show_geometry=True, title=None, dpi=100):
        self.n = n_qubits
        self.N = 2 ** n_qubits
        self.mask = mask
        self.theta = asin(sqrt(mask.sum() / self.N))
        self.show_geometry = show_geometry
        if labels is None:
            labels = [format(i, f"0{n_qubits}b") for i in range(self.N)]
        self.labels = list(labels)

        width = 13 if show_geometry else 9
        top = 0.80 if title else 0.85
        self.fig = plt.figure(figsize=(width, 5.8), facecolor=BG, dpi=dpi)
        if show_geometry:
            gs = self.fig.add_gridspec(2, 2, width_ratios=[1, 1.45], hspace=0.55, wspace=0.12,
                                       left=0.02, right=0.98, top=top, bottom=0.12)
            self.ax_geo = self.fig.add_subplot(gs[:, 0])
            self.ax_amp = self.fig.add_subplot(gs[0, 1])
            self.ax_prob = self.fig.add_subplot(gs[1, 1])
        else:
            gs = self.fig.add_gridspec(2, 1, hspace=0.55, left=0.08, right=0.97, top=top, bottom=0.12)
            self.ax_geo = None
            self.ax_amp = self.fig.add_subplot(gs[0])
            self.ax_prob = self.fig.add_subplot(gs[1])

        self.caption = self.fig.text(0.5, 0.95, "", ha="center", va="center", color=WHITE, fontsize=14)
        if title:
            self.fig.text(0.5, 0.895, title, ha="center", va="center", color=GREY, fontsize=11)

        self._setup_bars()
        if show_geometry:
            self._setup_geometry()
        self._temporary = []

    # ----- static parts -----
    def _style(self, ax):
        ax.set_facecolor(BG)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GREY)
        ax.tick_params(colors=GREY, labelsize=8)

    def _setup_bars(self):
        x = np.arange(self.N)
        colors = [YELLOW if k else BLUE for k in self.mask]
        width = 0.8 if self.N <= 32 else 1.0

        ax = self.ax_amp
        self._style(ax)
        self.amp_bars = ax.bar(x, np.zeros(self.N), color=colors, width=width)
        self.zero_line = ax.axhline(0, color=WHITE, lw=1, alpha=0.4)
        self.mean_line = ax.axhline(0, color=RED, lw=1.3, ls="--")
        self.mean_label = ax.text(0.995, 0, "mean", color=RED, fontsize=9, ha="right", va="bottom",
                                  transform=ax.get_yaxis_transform())
        ax.set_xlim(-0.7, self.N - 0.3)
        ax.set_ylim(-1.05, 1.05)
        ax.set_title("Amplitudes  (these can be negative!)", color=WHITE, fontsize=11, loc="left")
        ax.set_xticks([])

        ax = self.ax_prob
        self._style(ax)
        self.prob_bars = ax.bar(x, np.zeros(self.N), color=colors, width=width)
        ax.set_xlim(-0.7, self.N - 0.3)
        ax.set_ylim(0, 1.05)
        ax.set_title("Probabilities = amplitude²", color=WHITE, fontsize=11, loc="left")
        self.prob_text = ax.text(1.0, 1.04, "", transform=ax.transAxes, color=YELLOW,
                                 fontsize=12, ha="right", va="bottom", fontweight="bold")
        if self.N <= 64:
            ax.set_xticks(x)
            rotate = max(len(lbl) for lbl in self.labels) > 2
            size = 9 if self.N <= 8 else 8 if self.N <= 16 else 6.5 if self.N <= 32 else 5
            ax.set_xticklabels(self.labels, rotation=90 if rotate else 0, fontsize=size)
            for tick, key in zip(ax.get_xticklabels(), self.mask):
                tick.set_color(YELLOW if key else GREY)
        else:
            ticks = np.flatnonzero(self.mask)
            ax.set_xticks(ticks)
            ax.set_xticklabels([self.labels[t] for t in ticks], color=YELLOW, fontsize=8, rotation=90)

    def _setup_geometry(self):
        ax = self.ax_geo
        ax.set_facecolor(BG)
        ax.set_xlim(-1.32, 1.32)
        ax.set_ylim(-1.32, 1.32)
        ax.set_aspect("equal")
        ax.axis("off")
        th = self.theta

        ax.add_patch(Circle((0, 0), 1, fill=False, color=DARK_GREY, lw=1))
        # vertical axis = the key, horizontal axis = all the other answers
        ax.plot([0, 0], [-1.15, 1.15], color=YELLOW, lw=1.2, alpha=0.7)
        ax.text(0.05, 1.18, r"$|\mathrm{key}\rangle$", color=YELLOW, fontsize=14)
        self.h_mirror, = ax.plot([-1.2, 1.2], [0, 0], color=WHITE, lw=1, alpha=0.45)
        ax.text(1.2, -0.1, r"$|\mathrm{others}\rangle$", color=WHITE, fontsize=12, ha="right", va="top")
        # balanced state |b> (all answers equal): where we start, theta above |others>
        bx, by = cos(th), sqrt(1 - cos(th) ** 2)
        self.b_mirror, = ax.plot([-1.25 * bx, 1.25 * bx], [-1.25 * by, 1.25 * by], color=TEAL, lw=1, alpha=0.35)
        ax.text(1.27 * bx, 1.27 * by, r"$|b\rangle$", color=TEAL, fontsize=13, ha="left", va="center")
        ax.add_patch(Arc((0, 0), 0.5, 0.5, theta1=0, theta2=degrees(th), color=TEAL, lw=1.5))
        mid = th / 2
        ax.text(0.33 * cos(mid), 0.33 * np.sin(mid), r"$\theta$", color=TEAL, fontsize=11, va="center")

        ax.text(0, -1.3,
                f"N = {self.N} answers    θ = {degrees(th):.1f}°    each round turns the arrow 2θ = {2 * degrees(th):.1f}°",
                color=GREY, fontsize=9.5, ha="center")
        self.arrow = FancyArrowPatch((0, 0), (1, 0), arrowstyle="-|>", mutation_scale=20,
                                     lw=3, color=BLUE, zorder=5)
        ax.add_patch(self.arrow)

    # ----- per-frame drawing -----
    def draw(self, amps, caption="", mirror=None, ghosts=(), arcs=()):
        for bar, a in zip(self.amp_bars, amps):
            bar.set_height(a)
        for bar, a in zip(self.prob_bars, amps):
            bar.set_height(a * a)
        mean = amps.mean()
        self.mean_line.set_ydata([mean, mean])
        self.mean_label.set_y(mean)
        p_key = float(np.sum(amps[self.mask] ** 2))
        self.prob_text.set_text(f"P(key) = {100 * p_key:.1f}%")
        self.caption.set_text(caption)

        # The mirror for the oracle is the zero line (only the key's bar flips);
        # the mirror for the diffuser is the mean line (every bar flips around it).
        self.zero_line.set_linewidth(2.5 if mirror == "oracle" else 1)
        self.zero_line.set_alpha(1 if mirror == "oracle" else 0.4)
        self.mean_line.set_linewidth(2.8 if mirror == "diffuser" else 1.3)

        if self.ax_geo is None:
            return
        for artist in self._temporary:
            artist.remove()
        self._temporary = []

        x, y = _plane_xy(amps, self.mask)
        self.arrow.set_positions((0, 0), (x, y))
        self.h_mirror.set(lw=2.5 if mirror == "oracle" else 1, alpha=1 if mirror == "oracle" else 0.45,
                          ls="--" if mirror == "oracle" else "-")
        self.b_mirror.set(lw=2.5 if mirror == "diffuser" else 1, alpha=1 if mirror == "diffuser" else 0.35,
                          ls="--" if mirror == "diffuser" else "-")

        for gx, gy in ghosts:
            g = FancyArrowPatch((0, 0), (gx, gy), arrowstyle="-|>", mutation_scale=16,
                                lw=2, color=BLUE, alpha=0.22, zorder=4)
            self.ax_geo.add_patch(g)
            self._temporary.append(g)
        for i, (a0, a1) in enumerate(arcs):
            color = RED if i % 2 == 0 else GREEN
            radius = 0.62 + 0.08 * (i % 2)
            arc = Arc((0, 0), 2 * radius, 2 * radius, theta1=min(a0, a1), theta2=max(a0, a1),
                      color=color, lw=2.5, zorder=3)
            self.ax_geo.add_patch(arc)
            mid = np.radians((a0 + a1) / 2)
            label = self.ax_geo.text((radius + 0.13) * cos(mid), (radius + 0.13) * np.sin(mid), "2θ",
                                     color=color, fontsize=10, ha="center", va="center")
            self._temporary += [arc, label]


def show_state(state, marked, caption="", labels=None, show_geometry=True):
    """
    Draw one quantum state (a Qiskit Statevector, or a circuit with no measurements).
    `marked` is the key: a bitstring like '101', a list of them, or an oracle circuit.
    """
    amps = _real_amps(state)
    n = int(np.log2(len(amps)))
    pic = _GroverPicture(n, _marked_mask(n, marked), labels, show_geometry)
    pic.draw(amps, caption)
    plt.show()


def animate_grover(n_qubits, oracle, iterations=None, labels=None, show_geometry=True,
                   title=None, speed=1.0, save_gif=None):
    """
    Animate Grover's algorithm step by step, 3Blue1Brown style.

    n_qubits   : number of qubits (2 to 6 works well)
    oracle     : the key as a bitstring like '101', or an oracle QuantumCircuit
    iterations : number of Grover rounds (None = the best number)
    labels     : optional names for the answers (for example cipher keys 0..31)
    speed      : 2.0 plays twice as fast, 0.5 half as fast
    save_gif   : optional filename like 'grover.gif' to also save the animation for slides
    """
    oracle = _to_oracle(n_qubits, oracle)
    mask = _marked_mask(n_qubits, oracle)
    if iterations is None:
        iterations = best_iterations(n_qubits, int(mask.sum()))
    snaps = grover_snapshots(n_qubits, oracle, iterations)
    N = 2 ** n_qubits

    flip_frames, hold_frames = 12, 5
    frames, ghosts, arcs = [], [], []

    def hold(amps, caption, count):
        for _ in range(count):
            frames.append((amps, caption, None, tuple(ghosts), tuple(arcs)))

    def flip(a0, a1, caption, mirror):
        # a0 and a1 are both real Qiskit states and a1 is the mirror image of a0,
        # so their midpoint lies on the mirror. Sliding through it with cos() looks
        # exactly like flipping the picture over the mirror line.
        mid = (a0 + a1) / 2
        for i in range(1, flip_frames + 1):
            t = cos(pi * i / flip_frames)
            frames.append((mid + t * (a0 - mid), caption, mirror, tuple(ghosts), tuple(arcs)))

    def angle(amps):
        return degrees(atan2(*reversed(_plane_xy(amps, mask))))

    start = _real_amps(snaps[0][2])
    hold(start, "Start: Hadamard on every qubit → every answer equally likely", hold_frames + 6)
    prev = start
    for step, k, sv in snaps[1:]:
        amps = _real_amps(sv)
        if step == "oracle":
            caption = f"Round {k} of {iterations}:  Oracle flips the sign of the key"
        else:
            caption = f"Round {k} of {iterations}:  Diffuser reflects every amplitude about the mean"
        flip(prev, amps, caption, step)
        ghosts.append(_plane_xy(prev, mask))
        if step == "diffuser":
            a0 = arcs[-1][1] if arcs else angle(start)
            turn = (angle(amps) - a0 + 180) % 360 - 180  # keep counting past 180 degrees
            arcs.append((a0, a0 + turn))
        hold(amps, caption, hold_frames)
        prev = amps
    p_final = float(np.sum(prev[mask] ** 2))
    hold(prev, f"After {iterations} rounds:  P(key) = {100 * p_final:.1f}%   "
               f"(a random guess: {100 * mask.sum() / N:.1f}%)", hold_frames + 12)

    # Low resolution keeps the notebook small; every frame is stored as an image.
    pic = _GroverPicture(n_qubits, mask, labels, show_geometry, title, dpi=72)

    def update(i):
        pic.draw(*frames[i])
        return []

    fps = 14 * speed
    anim = animation.FuncAnimation(pic.fig, update, frames=len(frames), interval=1000 / fps, blit=False)
    if save_gif:
        anim.save(save_gif, writer=animation.PillowWriter(fps=fps), dpi=110)
    html = anim.to_jshtml(fps=fps, default_mode="once")
    plt.close(pic.fig)
    return HTML(html)


# ---------------------------------------------------------------------------
# 4. Charts: success vs. rounds, measurement histograms, noise
# ---------------------------------------------------------------------------

def _dark_axes(figsize=(9, 4)):
    fig, ax = plt.subplots(figsize=figsize, facecolor=BG)
    ax.set_facecolor(BG)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GREY)
    ax.tick_params(colors=GREY)
    return fig, ax


def plot_success_vs_rounds(n_qubits, oracle, max_rounds=None):
    """Probability of measuring the key after 0, 1, 2, ... Grover rounds (exact, from Statevector)."""
    oracle = _to_oracle(n_qubits, oracle)
    mask = _marked_mask(n_qubits, oracle)
    best = best_iterations(n_qubits, int(mask.sum()))
    if max_rounds is None:
        max_rounds = 3 * best + 2
    snaps = grover_snapshots(n_qubits, oracle, max_rounds)
    probs = [np.sum(_real_amps(sv)[mask] ** 2) for step, k, sv in snaps if step in ("start", "diffuser")]
    rounds = np.arange(len(probs))

    fig, ax = _dark_axes()
    ax.plot(rounds, probs, color=BLUE, lw=2, marker="o", markersize=7)
    ax.plot(best, probs[best], marker="o", markersize=13, color=YELLOW, zorder=5)
    ax.annotate(f"best: {best} rounds ({100 * probs[best]:.1f}%)", (best, probs[best]),
                xytext=(10, 8), textcoords="offset points", color=YELLOW)
    ax.axhline(mask.sum() / 2 ** n_qubits, color=GREY, ls="--", lw=1)
    ax.text(max_rounds, mask.sum() / 2 ** n_qubits + 0.02, "random guess", color=GREY, ha="right")
    ax.set_xticks(rounds)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("Number of Grover rounds", color=WHITE)
    ax.set_ylabel("P(measure the key)", color=WHITE)
    ax.set_title(f"{n_qubits} qubits = {2 ** n_qubits} possible answers", color=WHITE)
    plt.show()


def run_grover(n_qubits, oracle, iterations=None, shots=1024, two_qubit_error=0.0):
    """
    Build the Grover circuit and run it on Qiskit's Aer simulator. Returns measurement counts.
    two_qubit_error > 0 adds noise like real hardware (0.01 = a 1% error on every CNOT).
    """
    qc = grover_circuit(n_qubits, oracle, iterations, measure=True)
    if two_qubit_error > 0:
        sim = noisy_simulator(two_qubit_error)
        tqc = transpile(qc, basis_gates=BASIS_GATES, coupling_map=CouplingMap.from_line(n_qubits),
                        optimization_level=1, seed_transpiler=7)
    else:
        sim = AerSimulator()
        tqc = transpile(qc, sim)
    return sim.run(tqc, shots=shots).result().get_counts()


def noisy_simulator(two_qubit_error=0.01):
    """Aer simulator with simple depolarizing noise (one-qubit gates are 10x better than CNOTs)."""
    noise = NoiseModel()
    noise.add_all_qubit_quantum_error(depolarizing_error(two_qubit_error / 10, 1), ["sx", "x"])
    noise.add_all_qubit_quantum_error(depolarizing_error(two_qubit_error, 2), ["cx"])
    return AerSimulator(noise_model=noise)


def circuit_cost(n_qubits, oracle, iterations=None):
    """
    How big the circuit gets once compiled for a chip whose qubits sit in a line
    and only know CNOT plus single-qubit gates.
    """
    qc = grover_circuit(n_qubits, oracle, iterations, measure=False)
    tqc = transpile(qc, basis_gates=BASIS_GATES, coupling_map=CouplingMap.from_line(n_qubits),
                    optimization_level=1, seed_transpiler=7)
    ops = tqc.count_ops()
    return {"two-qubit gates (CNOT)": ops.get("cx", 0), "total gates": sum(ops.values()), "depth": tqc.depth()}


def plot_counts(counts, marked=None, labels=None, title="Measurement results"):
    """Histogram of every possible answer, with the key(s) in yellow."""
    n = len(next(iter(counts)))
    N = 2 ** n
    keys = [format(i, f"0{n}b") for i in range(N)]
    values = [counts.get(k, 0) for k in keys]
    mask = _marked_mask(n, marked) if marked is not None else np.zeros(N, dtype=bool)
    shots = sum(values)

    fig, ax = _dark_axes((max(7, min(14, 0.35 * N + 4)), 4))
    ax.bar(range(N), values, color=[YELLOW if m else BLUE for m in mask])
    ax.set_xticks(range(N))
    ax.set_xticklabels(labels if labels is not None else keys, rotation=90 if n > 2 else 0,
                       fontsize=9 if N <= 16 else 7)
    for tick, m in zip(ax.get_xticklabels(), mask):
        tick.set_color(YELLOW if m else GREY)
    if mask.any():
        hits = sum(v for v, m in zip(values, mask) if m)
        ax.text(0.01, 0.95, f"Key found {hits} / {shots} times ({100 * hits / shots:.1f}%)",
                transform=ax.transAxes, color=YELLOW, fontsize=12, va="top", fontweight="bold")
    ax.set_ylabel("Count", color=WHITE)
    ax.set_title(title, color=WHITE)
    plt.show()


def noise_experiment(qubit_counts=(2, 3, 4, 5), two_qubit_error=0.01, shots=4000):
    """
    For each problem size: P(key) on a perfect simulator vs. a noisy one vs. a random guess.
    """
    rows = []
    for n in qubit_counts:
        key = ("10" * n)[:n]
        ideal = run_grover(n, key, shots=shots)
        noisy = run_grover(n, key, shots=shots, two_qubit_error=two_qubit_error)
        cost = circuit_cost(n, key)
        rows.append((n, ideal.get(key, 0) / shots, noisy.get(key, 0) / shots, 1 / 2 ** n,
                     cost["two-qubit gates (CNOT)"]))

    print(f"{'qubits':>6} {'answers':>8} {'CNOTs':>6} {'perfect':>9} {'noisy':>8} {'random':>8}")
    for n, p_ideal, p_noisy, p_rand, cx in rows:
        print(f"{n:>6} {2 ** n:>8} {cx:>6} {100 * p_ideal:>8.1f}% {100 * p_noisy:>7.1f}% {100 * p_rand:>7.1f}%")

    fig, ax = _dark_axes((9, 4))
    x = np.arange(len(rows))
    w = 0.27
    ax.bar(x - w, [r[1] for r in rows], w, color=BLUE, label="perfect quantum computer")
    ax.bar(x, [r[2] for r in rows], w, color=RED, label=f"noisy ({100 * two_qubit_error:g}% error per CNOT)")
    ax.bar(x + w, [r[3] for r in rows], w, color=GREY, label="random guess")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r[0]} qubits\n{r[4]} CNOTs" for r in rows], color=WHITE)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("P(find the key)", color=WHITE)
    ax.legend(facecolor=BG, edgecolor=GREY, labelcolor=WHITE)
    ax.set_title("Bigger searches need deeper circuits, and noise wins", color=WHITE)
    plt.show()
    return rows
