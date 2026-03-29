from opensoundscape.audio import Audio
from opensoundscape.spectrogram import Spectrogram

import io
import base64
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import pandas as pd

# Optional dependencies — only needed at function call time, not on module import
try:
    from IPython.display import HTML, display
except ImportError:
    HTML = None
    display = None

try:
    import ipywidgets as widgets
except ImportError:
    widgets = None

try:
    import plotly.graph_objects as go
    import plotly.express as px
except ImportError:
    go = None
    px = None


def _require(*names):
    """Check that optional dependencies are available; raise a helpful error if not."""
    missing = []
    for name in names:
        if name == "ipywidgets" and widgets is None:
            missing.append("ipywidgets")
        elif name == "plotly" and go is None:
            missing.append("plotly")
        elif name == "IPython" and display is None:
            missing.append("IPython")
    if missing:
        raise ImportError(
            f"This function requires packages that are not installed: "
            f"{', '.join(missing)}. "
            f"Install them with: pip install {' '.join(missing)}"
        )


def annotate(
    clip_df,
    indices=None,
    annotation_buttons=None,
    dur=None,
    N=20,
    bandpass_range=None,
    dB_range=[-50, 0],
    cmap="Greys",
    cell_width=250,
    cell_height=125,
    apply_noise_reduction=False,
    normalize_audio=True,
    spec_kwargs=None,
):
    """Display an interactive grid of spectrograms with annotation toggle buttons.

    Each clip is shown as a spectrogram with click-to-play audio. If
    ``annotation_buttons`` is provided, toggle buttons appear below each clip.
    Activating a button sets the value to ``True``; deactivating it sets the
    value to ``None``.  Annotations are written in-place to ``annotations_df``
    if provided, otherwise to ``clip_df``.

    Optionally pass indices to subset the dataframe to rows to select from


    Args:
        clip_df (pd.DataFrame): DataFrame with columns 'file', 'start_time',
            'end_time'.  Used to load and display audio clips.
        indices (list, optional): indices of clip_df to subset to before
            selecting clips for display
        annotation_buttons (list[str], optional): Labels for annotation toggle
            buttons displayed below each clip.
        dur (float, optional): Duration of each audio clip in seconds.  If
            None, uses ``end_time - start_time`` for each row.
        N (int): Maximum number of clips to display (randomly sampled).
        bandpass_range (tuple, optional): ``(min_freq, max_freq)`` for
            bandpass filtering the spectrogram.
        dB_range (list): ``[min_dB, max_dB]`` for clipping spectrogram values.
        cmap (str): Matplotlib colormap name.
        cell_width (int): Minimum width of each grid cell in pixels.
        cell_height (int): Height of each spectrogram image in pixels.
        apply_noise_reduction (bool): if True uses noisereduce on audio clips with default params
        normalize_audio (bool): if True, normalizes audio clips to peak=1.0
        spec_kwargs (dict or None): keyword arguments to Spectrogram.from_audio()
    Returns:
        ipywidgets.GridBox: The displayed widget container.
    """
    _require("ipywidgets", "IPython")
    if spec_kwargs is None:
        spec_kwargs = {}
    # Determine which dataframe receives annotation writes
    if indices is None:
        indices = clip_df.index
    if len(indices) > N:
        indices = np.random.choice(indices, size=N, replace=False)
    view = clip_df.loc[indices]

    cell_widgets = []

    for _, row in view.iterrows():
        row_idx = row.name

        # --- time window ---
        if dur is None:
            start = row.start_time
            dur_i = row.end_time - row.start_time
        else:
            center_t = (row.start_time + row.end_time) / 2
            start = max(0, center_t - dur / 2)
            dur_i = dur

        # --- load audio ---
        a = Audio.from_file(
            row.file,
            offset=start,
            duration=dur_i,
            out_of_bounds_mode="ignore",
        )

        if apply_noise_reduction:
            a = a.reduce_noise()

        if normalize_audio:
            a = a.normalize()

        s = Spectrogram.from_audio(a, **spec_kwargs)
        if bandpass_range is not None:
            s = s.bandpass(*bandpass_range)

        spec = np.clip(
            s.spectrogram,
            a_min=dB_range[0],
            a_max=dB_range[1],
        )

        # --- render spectrogram to PNG ---
        fig, ax = plt.subplots(figsize=(2, 2), dpi=100)
        ax.imshow(spec, origin="lower", aspect="auto", cmap=cmap)
        ax.axis("off")

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        # --- audio to WAV bytes ---
        wav_buf = io.BytesIO()
        samples = a.samples
        samples = samples / max(1e-9, np.max(np.abs(samples)))
        samples_int16 = (samples * 32767).astype(np.int16)
        wavfile.write(wav_buf, a.sample_rate, samples_int16)
        audio_b64 = base64.b64encode(wav_buf.getvalue()).decode()

        # --- HTML widget with spectrogram image and click-to-play audio ---
        cell_html = widgets.HTML(
            value=(
                f'<img src="data:image/png;base64,{img_b64}" '
                f'style="width:100%;height:{cell_height}px;object-fit:fill;'
                f'display:block;cursor:pointer;" '
                f'onclick="this.nextElementSibling.play()"/>'
                f'<audio src="data:audio/wav;base64,{audio_b64}"></audio>'
            )
        )

        # --- annotation toggle buttons (ipywidgets) ---
        btn_widgets = []
        if annotation_buttons:
            for btn_label in annotation_buttons:
                # Initialize column if it doesn't exist yet
                if btn_label not in clip_df.columns:
                    clip_df[btn_label] = None

                # Reflect any existing annotation state
                existing = clip_df.at[row_idx, btn_label]
                initial_value = bool(existing)

                toggle = widgets.ToggleButton(
                    value=initial_value,
                    description=btn_label,
                    button_style="success" if initial_value else "",
                    layout=widgets.Layout(
                        flex="1 1 0%", height="24px", padding="0px 2px"
                    ),
                    style={"font_size": "11px"},
                )

                # Closure to capture current row_idx and btn_label
                def _make_observer(df, ridx, col, btn):
                    def _on_toggle(change):
                        if change["new"]:
                            df.at[ridx, col] = True
                            btn.button_style = "success"
                        else:
                            df.at[ridx, col] = None
                            btn.button_style = ""

                    return _on_toggle

                toggle.observe(
                    _make_observer(clip_df, row_idx, btn_label, toggle),
                    names="value",
                )
                btn_widgets.append(toggle)

        # --- assemble cell ---
        if btn_widgets:
            btn_box = widgets.HBox(
                btn_widgets,
                layout=widgets.Layout(
                    justify_content="center",
                    padding="4px",
                ),
            )
            cell = widgets.VBox(
                [cell_html, btn_box],
                layout=widgets.Layout(
                    border="1px solid #ddd",
                    overflow="hidden",
                    width=f"{cell_width}px",
                ),
            )
        else:
            cell = widgets.VBox(
                [cell_html],
                layout=widgets.Layout(
                    border="1px solid #ddd",
                    overflow="hidden",
                    width=f"{cell_width}px",
                ),
            )

        cell_widgets.append(cell)

    # --- grid layout ---
    grid = widgets.GridBox(
        cell_widgets,
        layout=widgets.Layout(
            grid_template_columns=f"repeat(auto-fill, minmax({cell_width}px, 1fr))",
            grid_gap="0px",
        ),
    )

    display(grid)
    return grid


def inspect(
    clip_df,
    dur=None,
    N=20,
    bandpass_range=None,
    dB_range=[-100, -20],
    cmap="Greys",
    normalize_audio=False,
    apply_noise_reduction=False,
    cell_width=250,
    cell_height=125,
    display_inline=True,
):
    """Display an interactive grid of spectrograms with click-to-play audio.

    Args:
        clip_df (pd.DataFrame): DataFrame with columns (or multi-index) 'file', 'start_time', (optional 'end_time')
        dur (float, optional): Duration of audio clips in seconds. If None, uses end_time - start_time.
            Note: if dur is specified but end_time is not present, will center the clip on start_time.
            If dur is None, requires end_time column to determine clip duration.
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
    _require("IPython")

    clip_df = clip_df.sample(min(N, len(clip_df))).copy()

    # if multi-index df, reset index to get 'file', 'start_time', etc. as columns
    if isinstance(clip_df.index, pd.MultiIndex):
        clip_df = clip_df.reset_index()

    cells = []

    for _, row in clip_df.iterrows():
        if dur is None:
            # require end_time column
            assert (
                "end_time" in row
            ), "If dur is None, rows must have an 'end_time' column"
            start = row.start_time
            dur = row.end_time - row.start_time
        else:
            # if end time not present, just start from the start time
            if "end_time" in row:
                center_t = (row.start_time + row.end_time) / 2
                start = center_t - dur / 2

            else:
                start = row.start_time
            # clamp negative values to 0
            start = max(0, start)

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
        dpi = 100
        fig_w = cell_width / dpi
        fig_h = cell_height / dpi

        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
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
    """Extract the row IDs of points currently selected in a Plotly FigureWidget.

    Args:
        fw: plotly FigureWidget with selectedpoints set on its traces

    Returns:
        np.ndarray of unique integer row IDs for the selected points
    """

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
    """Interactive 2-D scatter plot for exploring feature-space embeddings.

    Displays a Plotly scatter plot of the data, with an "Inspect selected" button that
    calls :func:`inspect` on the points lasso-selected in the plot.

    Requires ``ipywidgets``, ``plotly``, and ``IPython`` to be installed.

    Args:
        df (pd.DataFrame): DataFrame with at least ``x_col`` and ``y_col`` columns,
            plus 'file', 'start_time', 'end_time' columns for the inspect callback.
        x_col (str): Column name for x-axis values [default: "x"].
        y_col (str): Column name for y-axis values [default: "y"].
        color_col (str, optional): Column name for point colors.
        symbol_col (str, optional): Column name for point symbols.
        size_col (str, optional): Column name for point sizes.
        hover_name_col (str, optional): Column name for hover labels.
        **inspect_kwargs: Additional keyword arguments passed to :func:`inspect`.

    Returns:
        plotly.graph_objects.FigureWidget: The interactive scatter plot widget.
    """

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

    inspect_btn = widgets.Button(
        description="Inspect selected",
        button_style="info",
        icon="search",
    )

    def on_inspect_click(b):
        row_ids = []
        for tr in fw.data:
            if tr.selectedpoints is None:
                continue
            tr_points = np.asarray(tr.selectedpoints, dtype=int)
            tr_row_ids = tr.customdata[tr_points, 0]
            row_ids.extend(tr_row_ids)

        row_ids = np.unique(row_ids).astype(int)

        with inspect_out:
            inspect_out.clear_output(wait=True)
            if len(row_ids) == 0:
                print("No points selected")
                return
            selected = df.iloc[row_ids]
            print(f"{len(selected)} points selected")
            inspect(selected, **inspect_kwargs)

    inspect_btn.on_click(on_inspect_click)

    with fig_out:
        display(fw)

    display(fig_out, inspect_btn, inspect_out)

    return fw


def explore_histogram(
    df,
    value_col,
    label_col=None,
    bins=30,
    **inspect_kwargs,
):
    """Interactive histogram for exploring feature distributions.

    Displays one overlaid histogram trace per unique value in ``label_col``
    (or a single trace if ``label_col`` is None).  Each label gets a toggle
    button to show/hide its histogram trace and to include/exclude it from
    the "Inspect random selection" sample.

    Args:
        df (pd.DataFrame): DataFrame with columns including ``value_col``
            and (optionally) ``label_col``, plus 'file', 'start_time',
            'end_time' for the inspect callback.
        value_col (str): Column name for numeric values to histogram.
        label_col (str, optional): Column name for categorical labels.
            If None, all data is shown as a single histogram.
        bins (int): Number of histogram bins.
        **inspect_kwargs: Additional keyword arguments passed to
            :func:`inspect`.

    Returns:
        container (ipywidgets.VBox): Container widget with histogram and
            controls.
        fw (plotly.graph_objects.FigureWidget): The Plotly figure widget.
    """
    _require("ipywidgets", "plotly")

    # --- Determine unique labels and build one histogram trace per label ---
    if label_col is not None:
        unique_labels = list(df[label_col].unique())
    else:
        unique_labels = []

    # Use distinct label colors
    default_colors = px.colors.qualitative.Bold

    fw = go.FigureWidget()

    if unique_labels:
        for i, label in enumerate(unique_labels):
            color = default_colors[i % len(default_colors)]
            subset = df[df[label_col] == label]
            fw.add_histogram(
                x=subset[value_col],
                nbinsx=bins,
                name=str(label),
                opacity=0.6,
                marker_color=color,
            )
    else:
        # No label column — single histogram for all data
        fw.add_histogram(
            x=df[value_col],
            nbinsx=bins,
            name="all",
            opacity=0.6,
        )

    fw.update_layout(
        barmode="overlay",
        dragmode="zoom",
        width=900,
        height=450,
    )

    # --- Controls: one toggle per label + sample button ---
    sample_btn = widgets.Button(
        description="Inspect random selection",
        button_style="info",
        icon="search",
    )

    # Map label -> (toggle_button, color); button color matches histogram trace
    toggle_buttons = {}
    for i, label in enumerate(unique_labels):
        color = default_colors[i % len(default_colors)]
        btn = widgets.ToggleButton(
            description=str(label),
            value=True,
        )
        btn.style.button_color = color
        toggle_buttons[label] = (btn, color)

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
            mask = np.ones(len(df), dtype=bool)
        else:
            lo, hi = r
            mask = (df[value_col] >= lo) & (df[value_col] <= hi)

        # Exclude labels whose toggle is off
        for label, (btn, _color) in toggle_buttons.items():
            if not btn.value:
                mask &= df[label_col] != label

        return df[mask]

    # --- Callbacks ---
    def _make_visibility_callback(trace_idx, btn, color):
        """Create a callback that syncs a toggle button to its trace and color."""

        def _update(change):
            fw.data[trace_idx].visible = btn.value
            btn.style.button_color = color if btn.value else "#cccccc"

        return _update

    for i, label in enumerate(unique_labels):
        btn, color = toggle_buttons[label]
        btn.observe(
            _make_visibility_callback(i, btn, color),
            names="value",
        )

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
    all_buttons = [btn for btn, _color in toggle_buttons.values()] + [sample_btn]
    container = widgets.VBox(
        [
            fw,
            widgets.HBox(all_buttons),
            status,
        ]
    )

    display(container)

    return container
