from opensoundscape import Audio, Spectrogram

import io
import base64
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import HTML, display
from scipy.io import wavfile
import plotly.graph_objects as go
import ipywidgets as widgets
import plotly.express as px


def inspect(
    rows,
    dur=None,
    N=20,
    bandpass_range=None,
    dB_range=[-100, -20],
    cmap="Greys",
    normalize_audio=False,
    apply_noise_reduction=False,
    cell_width=150,
    cell_height=250,
    display_inline=True,
):
    """Display an interactive grid of spectrograms with click-to-play audio.

    Args:
        rows (pd.DataFrame): DataFrame with columns 'file', 'start_time', (optional 'end_time')
        dur (float, optional): Duration of audio clips in seconds. If None, uses end_time - start_time.
        N (int): Number of samples to display (randomly selected if more are available).
        bandpass_range (tuple, optional): Frequency range (min_freq, max_freq) for bandpass filtering.
        dB_range (list): [min_dB, max_dB] for spectrogram clipping.
        cmap (str): Matplotlib colormap for spectrograms.
        normalize_audio (bool): Whether to normalize audio clips.
        apply_noise_reduction (bool): Whether to apply noise reduction to audio clips.
        cell_width (int): Width of each cell in the grid (in pixels).
        cell_height (int): Height of each cell in the grid (in pixels).
        display_inline (bool): Whether to display the HTML output immediately.

    Returns:
        HTML object with the interactive grid.
    """

    rows = rows.sample(min(N, len(rows)))

    cells = []

    for _, row in rows.iterrows():
        if dur is None:
            start = row.start_time
            dur = row.end_time - row.start_time
        else:
            center_t = (row.start_time + row.end_time) / 2
            start = max(0, center_t - dur / 2)

        a = Audio.from_file(
            row.file,
            offset=start,
            duration=dur,
            out_of_bounds_mode="ignore",
        )

        if apply_noise_reduction:
            a = a.reduce_noise()

        if normalize_audio:
            a = a.normalize()

        s = Spectrogram.from_audio(a)
        if bandpass_range is not None:
            s = s.bandpass(*bandpass_range)

        # --- spectrogram array ---
        spec = s.spectrogram  # (freq, time)

        spec = np.clip(spec, a_min=dB_range[0], a_max=dB_range[1])

        # --- render spectrogram to PNG ---
        fig, ax = plt.subplots(figsize=(2.2, 2.2))
        ax.imshow(
            spec,
            origin="lower",
            aspect="auto",
            cmap=cmap,
        )
        ax.axis("off")

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        img_b64 = base64.b64encode(buf.getvalue()).decode()

        # --- audio → WAV bytes ---
        wav_buf = io.BytesIO()

        samples = a.samples
        # if samples.ndim > 1:
        #     samples = samples.mean(axis=0)

        # normalize safely to int16
        samples = samples / max(1e-9, np.max(np.abs(samples)))
        samples_int16 = (samples * 32767).astype(np.int16)

        wavfile.write(wav_buf, a.sample_rate, samples_int16)

        audio_b64 = base64.b64encode(wav_buf.getvalue()).decode()

        cells.append(
            f"""
            <div class="cell">
                <img src="data:image/png;base64,{img_b64}"
                     onclick="this.nextElementSibling.play()"/>
                <audio src="data:audio/wav;base64,{audio_b64}"></audio>
            </div>
            """
        )

    html = f"""
        <style>
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax({cell_width}px, {cell_width}px));
            gap: 10px;
        }}

        .cell {{
            cursor: pointer;
            border-radius: 6px;
            overflow: hidden;
            box-shadow: 0 2px 6px rgba(0,0,0,0.15);
            transition: transform 0.1s ease;
        }}

        .cell:hover {{
            transform: scale(1.03);
        }}

        .cell img {{
            width: 100%;
            height: {cell_height}px;
            object-fit: fill;
            display: block;
            background: black;
        }}
        </style>

        <div class="grid">
            {''.join(cells)}
        </div>
    """

    if display_inline:
        display(HTML(html))
    return HTML(html)


def get_selected_row_ids(fw):
    row_ids = []

    for tr in fw.data:
        if tr.selectedpoints is None:
            continue

        pts = np.asarray(tr.selectedpoints, dtype=int)
        row_ids.extend(tr.customdata[pts, 0])

    return np.unique(row_ids).astype(int)


def explore_features(
    df,
    x_col="x",
    y_col="y",
    color_col=None,
    symbol_col=None,
    size_col=None,
    hover_name_col=None,
    **inspect_kwargs,
):

    fig_out = widgets.Output()
    inspect_out = widgets.Output()

    df = df.copy()
    df["_row_id"] = np.arange(len(df))
    df["x"] = df[x_col]
    df["y"] = df[y_col]

    fig = px.scatter(
        df,
        x="x",
        y="y",
        color=color_col,
        hover_name=hover_name_col,
        symbol=symbol_col,
        size=size_col,
        opacity=0.8,
        custom_data=["_row_id"],
    )

    fw = go.FigureWidget(fig)
    # fw.selected_row_ids = np.array([], dtype=int)

    def on_select(trace, points, selector):
        # Only handle box selections
        if not hasattr(selector, "xrange"):
            return

        row_ids = []
        for tr in fw.data:
            # skip if trace is not visible

            if tr.selectedpoints is None:
                continue

            # selectedpoints are trace-local indices
            tr_points = np.asarray(tr.selectedpoints, dtype=int)

            # map to row ids via customdata
            tr_row_ids = tr.customdata[tr_points, 0]
            row_ids.extend(tr_row_ids)

        row_ids = np.unique(row_ids).astype(int)
        if len(row_ids) == 0:
            return
        selected = df.iloc[row_ids]

        with inspect_out:
            inspect_out.clear_output(wait=True)
            print(f"{len(selected)} points selected")
            inspect(selected, **inspect_kwargs)

    # Attach to one trace, don't want to see a bunch trying to plot over each other
    fw.data[0].on_selection(on_select)
    # for tr in fw.data:
    #     tr.on_selection(on_select)

    with fig_out:
        display(fw)

    display(fig_out, inspect_out)

    return fw


def make_label_buttons(fw, df, label_col="label"):
    btn0 = widgets.Button(
        description="Label selected = 0",
        button_style="danger",
        icon="times",
    )

    btn1 = widgets.Button(
        description="Label selected = 1",
        button_style="success",
        icon="check",
    )

    out = widgets.Output()

    def apply_label(label):
        with out:
            out.clear_output(wait=True)

            idx = get_selected_row_ids(fw)

            if len(idx) == 0:
                print("No points selected")
                return

            df.loc[idx, label_col] = label
            print(f"Labeled {len(idx)} points as {label}")

    btn0.on_click(lambda b: apply_label(0))
    btn1.on_click(lambda b: apply_label(1))

    display(widgets.HBox([btn0, btn1]), out)


def explore_histogram(
    df,
    value_col,
    label_col=None,
    positive_value=1,
    negative_value=0,
    bins=30,
    **inspect_kwargs,
):
    """Interactive histogram for exploring feature distributions.

    Args:
        df (pd.DataFrame): DataFrame with columns:
            - value_col: numeric values to plot histogram
            - label_col (optional): binary labels for splitting histogram
            - file, start_time, end_time (for default inspect fn)
        value_col (str): Column name for numeric values.
        label_col (str, optional): Column name for categorical labels. If None, all data
            is treated as one category.
        positive_value: Value in label_col representing positive class.
        negative_value: Value in label_col representing negative class.
        bins (int): Number of histogram bins.
        **inspect_kwargs,: Additional keyword arguments passed to the inspect function.
            including 'duration', 'N', 'bandpass_range', 'dB_range',
            'cmap', 'normalize_audio', 'apply_noise_reduction', 'cell_width',
            'cell_height', 'display_inline'.

    Returns:
        container (ipywidgets.VBox): Container widget with histogram and controls.
        fw (plotly.graph_objects.FigureWidget): Figure widget for the histogram.

    Usage:
    ```
    container, fw = explore_histogram(
        df,
        value_col='feature1',
        label_col='label',
        positive_value=1,
        negative_value=0,
        bins=50,
    )
    display(container)
    ```
    """

    fig_out = widgets.Output()
    ctrl_out = widgets.Output()

    # split data
    if label_col is None:
        df_pos = df
        df_neg = df.iloc[0:0]
    else:
        df_pos = df[df[label_col] == positive_value]
        df_neg = df[df[label_col] == negative_value]

    fw = go.FigureWidget()

    fw.add_histogram(
        x=df_pos[value_col],
        nbinsx=bins,
        name="Positive",
        opacity=0.6,
        marker_color="green",
    )

    fw.add_histogram(
        x=df_neg[value_col],
        nbinsx=bins,
        name="Negative",
        opacity=0.6,
        marker_color="red",
    )

    fw.update_layout(
        barmode="overlay",
        dragmode="zoom",
        width=900,
        height=450,
    )

    # --- Controls ---
    show_pos = widgets.ToggleButton(
        description="Positive",
        value=True,
        button_style="success",
    )

    show_neg = widgets.ToggleButton(
        description="Negative",
        value=True,
        button_style="danger",
    )

    sample_btn = widgets.Button(
        description="Inspect random selection",
        button_style="info",
        icon="search",
    )

    status = widgets.Output()

    # --- Helpers ---
    def get_visible_range():
        r = fw.layout.xaxis.range
        if r is None:
            return None
        return float(r[0]), float(r[1])

    def get_selected_rows():
        r = get_visible_range()
        if r is None:
            return df

        lo, hi = r
        mask = (df[value_col] >= lo) & (df[value_col] <= hi)

        if show_pos.value and not show_neg.value:
            mask &= df[label_col] == positive_value
        elif show_neg.value and not show_pos.value:
            mask &= df[label_col] == negative_value

        return df[mask]

    # --- Callbacks ---
    def update_visibility(*args):
        fw.data[0].visible = show_pos.value
        fw.data[1].visible = show_neg.value

    show_pos.observe(update_visibility, names="value")
    show_neg.observe(update_visibility, names="value")

    def on_sample_click(b):
        with status:
            status.clear_output(wait=True)

            sel = get_selected_rows()
            if len(sel) == 0:
                print("No samples in selected range")
                return
            inspect(sel, **inspect_kwargs)

    sample_btn.on_click(on_sample_click)

    # --- Display ---
    container = widgets.VBox([fig_out, ctrl_out])

    with fig_out:
        display(fw)

    with ctrl_out:
        display(
            widgets.HBox([show_pos, show_neg, sample_btn]),
            status,
        )

    return container, fw
