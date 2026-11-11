"""Interactive forward kinematics for a two-link planar arm with a wrist."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import CheckButtons, Slider


LINK_LENGTH = 1.0  # meters, for each link
UTENSIL_LENGTH = 0.35  # meters, from the attachment to the arrow tip


def arm_pose(theta1, theta2, theta3=0.0):
    """Return the base, elbow, attachment, and arrow tip for angles in radians."""
    directions = theta1 + np.array([0.0, theta2])
    link_vectors = LINK_LENGTH * np.column_stack(
        (np.cos(directions), np.sin(directions))
    )
    base = np.zeros(2)
    elbow, attachment = np.cumsum(link_vectors, axis=0)
    utensil_angle = directions[1] + theta3
    arrow_tip = attachment + UTENSIL_LENGTH * np.array(
        [np.cos(utensil_angle), np.sin(utensil_angle)]
    )
    return base, elbow, attachment, arrow_tip


def draw_angle(arc, label, center, start, sweep, radius):
    """Update a signed angle arc and place its label at the midpoint."""
    angles = np.linspace(start, start + sweep, 80)
    arc.set_data(
        center[0] + radius * np.cos(angles),
        center[1] + radius * np.sin(angles),
    )
    midpoint = start + sweep / 2
    label.set_position(
        center + (radius + 0.16) * np.array([np.cos(midpoint), np.sin(midpoint)])
    )


def build_viewer():
    """Build the figure and return its sliders so they stay responsive."""
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.subplots_adjust(left=0.13, right=0.93, top=0.9, bottom=0.295)
    fig.suptitle("2d Robot Arm Testing", fontsize=19, fontweight="bold", y=0.97)
    limit = 2 * LINK_LENGTH + UTENSIL_LENGTH + 0.25
    ax.set(xlim=(-limit, limit), ylim=(-limit, limit), xlabel="x (m)", ylabel="y (m)")
    ax.set_aspect("equal")
    ax.set_axisbelow(True)
    ax.grid(color="#e2e8f0")
    ax.axhline(0, color="#a1aab8", linewidth=1)
    ax.axvline(0, color="#a1aab8", linewidth=1)

    colors = ("#245bb4", "#06776b", "#a63d12")
    link1, = ax.plot([], [], color=colors[0], linewidth=5, label="Link 1", zorder=3)
    link2, = ax.plot([], [], color=colors[1], linewidth=5, label="Link 2", zorder=3)
    joints, = ax.plot(
        [], [], "o", color="#243448", markeredgecolor="white", markersize=9, zorder=5
    )
    guide1, = ax.plot([], [], "--", color=colors[0], alpha=0.65)
    guide2, = ax.plot([], [], "--", color=colors[1], alpha=0.65)
    guide3, = ax.plot([], [], "--", color=colors[2], alpha=0.65)
    arc1, = ax.plot([], [], color=colors[0], linewidth=1.6, zorder=4)
    arc2, = ax.plot([], [], color=colors[1], linewidth=1.6, zorder=4)
    arc3, = ax.plot([], [], color=colors[2], linewidth=1.6, zorder=4)
    label_style = dict(ha="center", va="center", zorder=6, fontweight="bold")
    label1 = ax.text(0, 0, "θ₁", color=colors[0], **label_style)
    label2 = ax.text(0, 0, "θ₂", color=colors[1], **label_style)
    label3 = ax.text(0, 0, "θ₃", color=colors[2], **label_style)
    utensil = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", color=colors[2], lw=2.5, shrinkA=0, shrinkB=0),
        zorder=4,
    )
    ax.legend(loc="upper left", framealpha=0.95)
    readout = fig.text(0.5, 0.225, "", ha="center", color=colors[2])

    slider1 = Slider(
        fig.add_axes([0.23, 0.158, 0.55, 0.03]),
        "θ₁  Base", -180, 180, valinit=35, valfmt="%+.1f°", color=colors[0],
    )
    slider2 = Slider(
        fig.add_axes([0.23, 0.103, 0.55, 0.03]),
        "θ₂  Elbow", -180, 180, valinit=50, valfmt="%+.1f°", color=colors[1],
    )
    slider3 = Slider(
        fig.add_axes([0.23, 0.048, 0.55, 0.03]),
        "θ₃  Wrist", -180, 180, valinit=0, valfmt="%+.1f°", color=colors[2],
    )
    wrist_toggle_ax = fig.add_axes([0.23, 0.005, 0.22, 0.04], frameon=False)
    wrist_toggle = CheckButtons(wrist_toggle_ax, ["Enable wrist"], [True])

    def update(_):
        wrist_enabled = wrist_toggle.get_status()[0]
        wrist_angle = slider3.val if wrist_enabled else 0.0
        theta1, theta2, theta3 = np.deg2rad([slider1.val, slider2.val, wrist_angle])
        base, elbow, attachment, arrow_tip = arm_pose(theta1, theta2, theta3)
        link1.set_data(*np.array([base, elbow]).T)
        link2.set_data(*np.array([elbow, attachment]).T)
        joints.set_data(*np.array([base, elbow, attachment]).T)
        guide1.set_data([0, 0.66], [0, 0])
        elbow_reference = elbow + 0.58 * (elbow - base) / LINK_LENGTH
        guide2.set_data(*np.array([elbow, elbow_reference]).T)
        wrist_reference = attachment + 0.35 * (attachment - elbow) / LINK_LENGTH
        guide3.set_data(*np.array([attachment, wrist_reference]).T)
        draw_angle(arc1, label1, base, 0, theta1, 0.43)
        draw_angle(arc2, label2, elbow, theta1, theta2, 0.29)
        draw_angle(arc3, label3, attachment, theta1 + theta2, theta3, 0.29)
        utensil.set_position(attachment)
        utensil.xy = arrow_tip
        readout.set_text(f"Utensil attachment   ({attachment[0]:+.2f}, {attachment[1]:+.2f}) m")
        fig.canvas.draw_idle()

    def toggle_wrist(_):
        wrist_enabled = wrist_toggle.get_status()[0]
        slider3.set_active(wrist_enabled)
        slider3.ax.set_visible(wrist_enabled)
        for artist in (guide3, arc3, label3):
            artist.set_visible(wrist_enabled)
        update(None)

    slider1.on_changed(update)
    slider2.on_changed(update)
    slider3.on_changed(update)
    wrist_toggle.on_clicked(toggle_wrist)
    update(None)
    return fig, slider1, slider2, slider3


if __name__ == "__main__":
    viewer = build_viewer()
    plt.show()
