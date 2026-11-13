import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation

from scipy.interpolate import make_interp_spline

def line_delta_animation(Xs, Ys, colors, labels, title, xlabel, ylabel, 
                            frames_morph=50, frames_hold=25, fname=None):
    
    if len(Ys.shape) == 2:
        Ys = np.array([[y] for y in Ys])
        colors = [[c] for c in colors]

    def init():
        for idx in range(len(lines)):
            lines[idx].set_data([], [])
            #control_dots.set_offsets(np.empty((0, 2)))
            fill_poly = polys[idx]
            if fill_poly[0] is not None:
                fill_poly[0].remove()
                fill_poly[0] = None

        return *lines, legend_obj #, control_dots

    minmax = [Xs.min().astype(np.int32), Xs.max().astype(np.int32)]

    # Evaluation grid for high-resolution spline rendering
    x_dense = np.linspace(minmax[0], minmax[1], (minmax[1] - minmax[0]) * 100)

    # Precompute the dense evaluation arrays for static baseline reference
    dense_lines = []
    for y_ctrl in Ys:
        dense_group = []
        for idx in range(len(y_ctrl)):
            spl = make_interp_spline(Xs, y_ctrl[idx], k=3)
            dense_group.append(spl(x_dense))
        dense_lines.append(dense_group)

    cycle_frames = frames_morph + frames_hold
    num_transitions = len(Ys) - 1
    total_frames = num_transitions * cycle_frames

    # Smooth ease-in-out curve for natural speed acceleration/deceleration
    def ease_in_out(t):
        return t * t * (3 - 2 * t)

    # 3. Figure set up
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.set_xlim(minmax[0], minmax[1])
    #ax.set_ylim(-1.8, 1.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle='--', alpha=0.5)

    # Render initial base elements
    lines, polys, proxy_handles, eq_lines = [], [], [], []
    for idx in range(len(Ys[0])):
        line, = ax.plot([], [], lw=2.5, zorder=3)
        lines.append(line)
    #control_dots = ax.scatter([], [], color='black', s=30, zorder=4) #, label='Control Points')
    
        fill_poly = [None]  # Container for active fill_between PolyCollection
        polys.append(fill_poly)

        # Create proxy handles for each dataset so all appear in the legend simultaneously
        if idx == 0:
            proxy_handles.extend([
                Line2D([0], [0], color=colors[i][idx], lw=2.5, label=labels[i])
                for i in range(len(labels))
            ])
        
        if Ys[0][idx][0] < Ys[0][idx][-1]:
            eq_line, = ax.plot([0, Xs[-1]], [0, 1], linestyle="--", color="black")
            eq_lines.append(eq_line)
        else:
            eq_line, = ax.plot([0, Xs[-1]], [1, 0], linestyle="--", color="black")
            eq_lines.append(eq_line)


    legend_obj = ax.legend(handles=proxy_handles, prop={"size": 10}, 
                    loc='center left', bbox_to_anchor=(1, 0.5), framealpha=0.25)
    legend_lines = legend_obj.get_lines()  
    legend_texts = legend_obj.get_texts()  
    fig.subplots_adjust(right=0.7)
    
    # 4. Animation update loop
    def update(frame):
        
        for idx in range(len(Ys[0])):

            # Determine which transition segment we are in (0: Line 1->2, 1: Line 2->3)
            trans_idx = min(frame // cycle_frames, num_transitions - 1)
            frame_in_cycle = frame % cycle_frames

            # Reference control points and static baseline curve for this step
            y_ctrl_start = Ys[trans_idx][idx]
            y_ctrl_end = Ys[trans_idx + 1][idx]
            y_dense_start = dense_lines[trans_idx][idx]
            
            color_start = colors[trans_idx][idx]
            color_end = colors[trans_idx + 1][idx]

            # Calculate transition progress t
            if frame_in_cycle < frames_morph:
                raw_t = frame_in_cycle / (frames_morph - 1)
                t = ease_in_out(raw_t)
            else:
                t = 1.0  # Static hold phase

            # Morph control points via linear interpolation
            y_ctrl_curr = (1 - t) * y_ctrl_start + t * y_ctrl_end

            # Reconstruct cubic B-spline (k=3) from updated control points
            spline = make_interp_spline(Xs, y_ctrl_curr, k=3)
            y_dense_curr = spline(x_dense)

            # Blend line stroke color smoothly during transformation
            c1 = np.array(plt.cm.colors.to_rgb(color_start))
            c2 = np.array(plt.cm.colors.to_rgb(color_end))
            current_color = c1 * (1 - t) + c2 * t

            # Update line geometry and control handles
            lines[idx].set_data(x_dense, y_dense_curr)
            lines[idx].set_color(current_color)
            # lines[idx].set_label(f"Morphing: {labels[trans_idx]} → {labels[trans_idx+1]}")

            #control_dots.set_offsets(np.c_[Xs, y_ctrl_curr])

            # Remove previous frame's fill collection to avoid stacking memory/draws
            if polys[idx][0] is not None:
                polys[idx][0].remove()

            # Draw transparent shaded region between the starting baseline curve (Line N) 
            # and the morphing spline curve using Line N's color
            polys[idx][0] = ax.fill_between(
                x_dense,
                y_dense_start,
                y_dense_curr,
                color=color_start,
                alpha=0.30,
                zorder=2
            )

        # Dynamically highlight active legend entries based on transition progress t
        for i in range(len(labels)):
            if i == trans_idx:
                weight = 1.0 - t      # Fades out as it morphs into the next line
            elif i == trans_idx + 1:
                weight = t            # Fades in as it becomes the active line
            else:
                weight = 0.0          # Inactive dataset

            alpha = 0.35 + 0.65 * weight
            legend_lines[i].set_alpha(alpha)
            legend_lines[i].set_linewidth(2.0 + 1.5 * weight)
            legend_texts[i].set_alpha(alpha)
            legend_texts[i].set_fontweight('bold' if weight >= 0.5 else 'normal')

        return *lines, *[x[0] for x in polys], *eq_lines, legend_obj #control_dots

    # 5. Execute Animation (blit=False allows dynamic collection creation/removal)
    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=total_frames,
        interval=30,
        blit=False,
        cache_frame_data=False
    )

    if fname is None:
        plt.show()
    else:
        assert fname.endswith(".gif")
        print ("Saving animation to %s" % fname)
        anim.save(fname, writer="pillow", fps=30)

if __name__ == "__main__":

    # 1. Define control points for 3 distinct line datasets
    Xs = np.linspace(0, 10, 8)
    Ys = np.array([
        np.sin(Xs * 0.8),                                          # Line 1
        np.cos(Xs * 1.2) * np.exp(-Xs / 8),               # Line 2
        -0.6 * np.sin(Xs * 1.5) + 0.04 * (Xs - 5)**2 - 0.5   # Line 3
    ])

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Line 1: Blue, Line 2: Orange, Line 3: Green
    labels = ['Line 1', 'Line 2', 'Line 3']

    line_delta_animation(Xs, Ys, colors, labels, 'A', 'B', 'C', frames_morph=200, frames_hold=50, fname="gif.gif")
