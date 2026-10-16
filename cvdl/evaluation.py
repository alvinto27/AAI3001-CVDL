"""Prediction, fixed-panel evaluation and saved-prediction breakdowns.

All runtime choices are explicit. This module never reads environment settings.
Notebook 02 remains an independent standalone reference during the transition.
"""
from pathlib import Path
import hashlib
import json
import time

import numpy as np
import pandas as pd
import torch

from .metrics import auc_numpy, probability_error, metrics, summary_rows, global_summary
from .models import model_from_hparams
from .settings import SCHEMA
from lsb_notebook import fixed_subset, complete_bpp_bins
from lsb_torch import Corpus, FixedPairs, OrderedPairLoader, RunSettings

def predict_pairs(model, loader, *, device, data_mode) -> pd.DataFrame:
    """Predict clean and stego probabilities while retaining pair metadata."""
    if data_mode not in {"otf", "png"}:
        raise ValueError("data_mode must be otf/png")
    model.eval()
    network = getattr(model, "net", model)
    frame = loader.dataset.frame
    rows = []
    with torch.inference_mode():
        for clean, stego, indices in loader:
            clean = clean.to(device, dtype=torch.float32, non_blocking=True)
            stego = stego.to(device, dtype=torch.float32, non_blocking=True)
            probabilities = torch.softmax(network(torch.cat([clean, stego], dim=0)), dim=1)[:, 1]
            count = clean.shape[0]
            clean_probabilities = probabilities[:count].cpu().numpy()
            stego_probabilities = probabilities[count:].cpu().numpy()
            for position, pair_index in enumerate(indices.tolist()):
                row = frame.iloc[int(pair_index)]
                common = {
                    "cover_id": row["cover_id"],
                    "source_photo": row["source_photo"],
                    "camera_model": row["camera_model"],
                    "k": row["k"],
                    "bit_mode": row["bit_mode"],
                    "method": row["method"],
                    "placement": row["placement"],
                    "payload_kind": row["payload_kind"],
                    "bpp_actual": row["bpp_actual"],
                }
                if data_mode == "otf":
                    common.update(row.to_dict())
                    common["data_mode"] = "otf"
                else:
                    common["source"] = "raise"
                rows.append({**common, "file": row.get("file_clean", str(row["cover_id"])+":clean"), "label": 0, "prob_stego": float(clean_probabilities[position])})
                rows.append({**common, "file": row.get("file_stego", str(row["cover_id"])+":stego"), "label": 1, "prob_stego": float(stego_probabilities[position])})
    return pd.DataFrame(rows)


def prediction_accuracy(frame: pd.DataFrame) -> float:
    y_true = frame["label"].to_numpy(dtype=np.int64)
    y_score = frame["prob_stego"].to_numpy(dtype=np.float64)
    return float(((y_score >= 0.5).astype(np.int64) == y_true).mean())


def per_setting_table(predictions, *, data_mode):
    """Notebook-05 grouped P_E with each group's matching clean cover IDs."""
    if data_mode not in {'otf', 'png'}:
        raise ValueError('data_mode must be otf/png')
    stego_predictions = predictions.loc[predictions["label"] == 1].copy()
    clean_predictions = predictions.loc[predictions["label"] == 0].copy()
    stego_predictions["bit_mode_k"] = (
        stego_predictions["bit_mode"].astype(str) + "/" + pd.to_numeric(stego_predictions["k"], errors="raise").astype(int).astype(str)
    )
    stego_predictions["method_x_placement"] = (
        stego_predictions["method"].astype(str) + "/" + stego_predictions["placement"].astype(str)
    )
    if data_mode == "otf":
        stego_predictions["bpp_bin"] = complete_bpp_bins(stego_predictions["bpp_actual"])
    else:
        stego_predictions["bpp_bin"] = pd.cut(
            pd.to_numeric(stego_predictions["bpp_actual"], errors="coerce"),
            bins=[0.05, 0.15, 0.30, 0.50],
            labels=["0.05-0.15", "0.15-0.30", "0.30-0.50"],
            include_lowest=True,
            right=True,
        )
    assert not stego_predictions["bpp_bin"].isna().any(), "bpp bins must retain every pair"
    setting_rows = []
    setting_specs = [("source", "source"), ("camera_model", "camera_model"),
        ("bit_mode/k", "bit_mode_k"),
        ("bit_mode", "bit_mode"),
        ("method", "method"),
        ("placement", "placement"),
        ("method x placement", "method_x_placement"),
        ("payload_kind", "payload_kind"),
        ("bpp_bin", "bpp_bin"),
    ]
    for setting_label, setting_column in setting_specs:
        for value, group in stego_predictions.groupby(setting_column, dropna=False, observed=True):
            if pd.isna(value):
                continue
            cover_ids = set(group["cover_id"])
            paired_clean = clean_predictions.loc[clean_predictions["cover_id"].isin(cover_ids)]
            paired_images = pd.concat([paired_clean, group], ignore_index=True)
            group_labels = paired_images["label"].to_numpy(dtype=np.int64)
            group_scores = paired_images["prob_stego"].to_numpy(dtype=np.float64)
            setting_rows.append({
                "setting": setting_label,
                "value": str(value),
                "pairs": len(group),
                "P_E": probability_error(group_labels, group_scores),
            })
    per_setting = pd.DataFrame(setting_rows, columns=["setting", "value", "pairs", "P_E"])
    return per_setting


def save_predictions(predictions, path):
    """Save notebook-compatible CSV bytes without adding an index."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(path, index=False)
    return path


def panel_dataset(dataset, *, quick=False, quick_pairs=SCHEMA['quick_eval_pairs'].default):
    """Select the fixed QUICK panel subset (seed 1), or retain the full panel."""
    if quick_pairs <= 0:
        raise ValueError('quick_pairs must be positive')
    return fixed_subset(dataset, quick_pairs, seed=1) if quick else dataset


def evaluate(checkpoint, eval_yaml, *, root, device='cpu', split='test', output,
             workers=6, source=None, manifest_root=None, quick=False,
             quick_pairs=SCHEMA['quick_eval_pairs'].default):
    """Score one frozen panel and return the summary; default is the standard test."""
    if (split not in ('test','val','test_seen','test_unseen') or workers < 0
            or source not in (None,'alaska','lsdir')
            or (source is None and split not in ('test','val'))
            or (source=='lsdir' and split not in ('test','val'))
            or (source=='alaska' and split=='test')):
        raise ValueError('unsupported source/split or worker count')
    start = time.perf_counter()
    checkpoint = Path(checkpoint).resolve()
    evaluation_yaml = Path(eval_yaml).resolve()
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'output must be empty: {output}')
    output.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    hparams = ckpt.get('hyper_parameters', {})
    norm = hparams.get('norm', 'group')
    model = model_from_hparams(hparams).eval()
    state = {key.removeprefix('net.'): value for key, value in ckpt['state_dict'].items()}
    if len(state) != len(ckpt['state_dict']):
        raise ValueError('duplicate checkpoint keys')
    model.load_state_dict(state, strict=True)
    device = torch.device(device)
    model.to(device)
    settings = RunSettings(evaluation_yaml=str(evaluation_yaml),
                           sources=(source,) if source else ('raise','div2k'),
                           manifest_root=str(manifest_root) if manifest_root else None)
    dataset = FixedPairs(Corpus(Path(root), settings), split, sources=(source,) if source else ('raise','div2k'))
    if not len(dataset):
        raise ValueError('selected evaluation panel is empty')
    dataset = panel_dataset(dataset, quick=quick, quick_pairs=quick_pairs)
    loader = OrderedPairLoader(dataset, batch_size=16, num_workers=workers,
                               pin_memory=device.type == 'cuda', prefetch_images=128)
    pairs = dataset.frame
    setting_fields = ['cover_id','k','bit_mode','method','placement','payload_kind','stride',
                      'requested_bpp','effective_bpp','clipped']
    setting_digest = hashlib.sha256(pairs[setting_fields].to_csv(index=False).encode()).hexdigest()
    predictions = predict_pairs(model, loader, device=device, data_mode='otf')
    # Keep the same first fields as notebook 04; no request/embedding fields are lost.
    front = ['cover_id','source_photo','camera_model','k','bit_mode','method','placement',
             'payload_kind','bpp_actual']
    predictions = predictions[front + [x for x in predictions if x not in front]]
    predictions.to_csv(output / 'test_predictions.csv', index=False)
    bins = [0., .05, .15, .3, .5, .75, 1.]
    labels = ['<=0.05','0.05-0.15','0.15-0.30','0.30-0.50','0.50-0.75','0.75-1.0']
    predictions['bpp_bin'] = pd.cut(predictions['requested_bpp'], bins=bins,
                                     labels=labels, include_lowest=True, right=True)
    if predictions['bpp_bin'].isna().any():
        raise ValueError('requested bpp outside 0..1')
    summary = summary_rows(predictions)
    pd.DataFrame(summary).to_csv(output / 'summary.csv', index=False)
    clipped = (pairs.groupby('placement', sort=True)['clipped'].agg(['sum','count'])
               .reset_index().rename(columns={'sum':'clipped_pairs','count':'pairs'}))
    clipped.to_csv(output / 'clipped_by_placement.csv', index=False)
    info = dict(checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                norm=norm, clip_t=hparams.get('clip_t', 3.0), evaluation_yaml=str(evaluation_yaml),
                split=split, workers=workers, settings_sha256=setting_digest,
                source=source or 'standard', manifest_root=str(manifest_root) if manifest_root else None,
                pairs=len(dataset), quick_pair_cap=quick_pairs if quick else None,
                elapsed_seconds=time.perf_counter() - start, overall=summary[0])
    (output / 'run.json').write_text(json.dumps(info, indent=2) + '\n')
    print(json.dumps(info, indent=2), flush=True)
    return info, pd.DataFrame(summary)


def load_breakdown(predictions_path, *, labels_path=None):
    """Read notebook-02 metadata; return (frame, is_self_contained_otf)."""
    PRED_PATH = Path(predictions_path)
    LABELS_PATH = Path(labels_path) if labels_path is not None else None
    predictions = pd.read_csv(PRED_PATH, dtype={"cover_id": "string"})
    OTF_METADATA = {"pair_id", "source", "camera_model", "requested_bpp", "effective_bpp", "clipped"} <= set(predictions.columns)
    if OTF_METADATA:
        # Self-contained OTF predictions: no PNG label/path lookup.
        frame = predictions.copy()
        required = ["pair_id","cover_id","source","camera_model","bit_mode","k","method","placement","payload_kind","bpp_actual","requested_bpp","effective_bpp","clipped","label","prob_stego"]
        if frame[required].isna().any().any():
            raise ValueError("Incomplete OTF prediction metadata")
        if not np.isfinite(frame["prob_stego"].to_numpy(dtype=float)).all():
            raise ValueError("nonfinite score")
        if set(frame["label"].unique()) != {0,1}:
            raise ValueError("Both image classes required")
        if frame.duplicated(["pair_id","label"]).any():
            raise ValueError("Duplicate OTF pair/class")
        print("Metadata: self-contained OTF predictions; no raise512-run1 lookup")
    else:
        assert LABELS_PATH is not None and LABELS_PATH.exists(), f"Missing labels file: {LABELS_PATH}"

        predictions = pd.read_csv(PRED_PATH, dtype={"cover_id": "string"})
        labels = pd.read_csv(LABELS_PATH, dtype={"cover_id": "string"})

        required_prediction_columns = {"cover_id", "label", "prob_stego"}
        missing = required_prediction_columns - set(predictions.columns)
        if missing:
            raise ValueError(f"Predictions are missing columns: {sorted(missing)}")
        if "cover_id" not in labels or "label" not in labels:
            raise ValueError("Labels must contain cover_id and label")

        def normalise_cover_id(values: pd.Series) -> pd.Series:
            return values.astype("string").str.strip().str.zfill(6)

        predictions["cover_id"] = normalise_cover_id(predictions["cover_id"])
        labels["cover_id"] = normalise_cover_id(labels["cover_id"])
        predictions["label"] = pd.to_numeric(predictions["label"], errors="raise").astype(int)
        labels["label"] = pd.to_numeric(labels["label"], errors="raise").astype(int)
        predictions["prob_stego"] = pd.to_numeric(predictions["prob_stego"], errors="raise")
        if not np.isfinite(predictions["prob_stego"].to_numpy(dtype=np.float64)).all():
            raise ValueError("prob_stego contains a non-finite value")
        if set(predictions["label"].unique()) != {0, 1}:
            raise ValueError("Predictions must contain both labels 0 and 1")

        extra_columns = [
            "stride",
            "start_random",
            "payload_bytes",
            "units_changed",
            "bpp_actual",
            "size_value",
        ]
        setting_columns = ["bit_mode", "k", "method", "placement", "payload_kind", *extra_columns]
        label_setting_columns = ["cover_id", *setting_columns]
        missing_labels = set(label_setting_columns) - set(labels.columns)
        if missing_labels:
            raise ValueError(f"Labels are missing columns: {sorted(missing_labels)}")

        stego_labels = labels.loc[labels["label"] == 1, label_setting_columns].copy()
        if stego_labels["cover_id"].duplicated().any():
            raise ValueError("labels.csv has more than one stego row for a cover_id")
        frame = predictions.merge(
            stego_labels,
            on="cover_id",
            how="left",
            suffixes=("_prediction", "_labels"),
            validate="many_to_one",
        )
        for column in setting_columns:
            label_column = f"{column}_labels" if f"{column}_labels" in frame else column
            frame[column] = frame[label_column]
        required_metadata_columns = [column for column in setting_columns if column != "stride"]
        if frame[required_metadata_columns].isna().any().any():
            raise ValueError("A prediction row has no matching stego metadata in labels.csv")


    return frame, OTF_METADATA


def breakdown_tables(frame, *, otf_metadata, global_threshold):
    """Return notebook-02 setting table, heatmap cells and ordered bpp labels."""
    stego_predictions = frame.loc[frame["label"] == 1].copy()
    clean_predictions = frame.loc[frame["label"] == 0].copy()
    stego_predictions["k_number"] = pd.to_numeric(stego_predictions["k"], errors="raise").astype(int)
    stego_predictions["bit_mode_k"] = (
        stego_predictions["bit_mode"].astype(str) + "/" + stego_predictions["k_number"].astype(str)
    )

    def _placement_value(row: pd.Series) -> str:
        placement = str(row["placement"])
        if placement == "stride":
            return f"stride/{int(float(row['stride']))}"
        return placement

    stego_predictions["placement_group"] = stego_predictions.apply(_placement_value, axis=1)
    stego_predictions["method_x_placement"] = (
        stego_predictions["method"].astype(str) + " x " + stego_predictions["placement_group"]
    )
    stego_predictions["bpp_number"] = pd.to_numeric(stego_predictions["bpp_actual"], errors="raise")
    if otf_metadata:
        BPP_LABELS = ["<=0.05", "0.05-0.15", "0.15-0.30", "0.30-0.50", ">0.50"]
        stego_predictions["bpp_bin"] = pd.cut(stego_predictions["bpp_number"],
            bins=[-np.inf,.05,.15,.30,.50,np.inf], labels=BPP_LABELS, right=True)
        assert not stego_predictions["bpp_bin"].isna().any()
        print("OTF bpp bin pair counts:")
        print(stego_predictions["bpp_bin"].value_counts(sort=False).to_string())
    else:
        BPP_LABELS = ["0.05-0.15", "0.15-0.30", "0.30-0.50"]
        stego_predictions["bpp_bin"] = pd.cut(
            stego_predictions["bpp_number"],
            bins=[0.05, 0.15, 0.30, 0.50],
            labels=BPP_LABELS,
            include_lowest=True,
        )


    def _group_summary(group: pd.DataFrame) -> tuple[int, float, float]:
        cover_ids = set(group["cover_id"])
        paired_clean = clean_predictions.loc[clean_predictions["cover_id"].isin(cover_ids)]
        paired = pd.concat([paired_clean, group], ignore_index=True)
        group_labels = paired["label"].to_numpy(dtype=np.int64)
        group_scores = paired["prob_stego"].to_numpy(dtype=np.float64)
        detection = float((group["prob_stego"].to_numpy(dtype=np.float64) >= global_threshold).mean())
        return len(group), probability_error(group_labels, group_scores), detection

    setting_specs = [
        ("bit_mode/k", "bit_mode_k"),
        ("bit_mode", "bit_mode"),
        ("method", "method"),
        ("placement", "placement_group"),
        ("method x placement", "method_x_placement"),
        ("payload_kind", "payload_kind"),
        ("bpp bin", "bpp_bin"),
    ]
    if otf_metadata:
        setting_specs += [("source", "source"), ("camera_model", "camera_model")]
    setting_rows = []
    for setting_name, setting_column in setting_specs:
        for value, group in stego_predictions.groupby(setting_column, dropna=False, observed=True):
            if pd.isna(value):
                continue
            pairs, p_e, detection = _group_summary(group)
            setting_rows.append({
                "setting": setting_name,
                "value": str(value),
                "pairs": pairs,
                "P_E": p_e,
                "detection_rate_global": detection,
            })

    per_setting = pd.DataFrame(
        setting_rows,
        columns=["setting", "value", "pairs", "P_E", "detection_rate_global"],
    ).sort_values(["setting", "value"]).reset_index(drop=True)
    cell_rows = []
    for (bit_mode, k_number, bpp_bin), group in stego_predictions.groupby(
        ["bit_mode", "k_number", "bpp_bin"], dropna=False, observed=True
    ):
        if pd.isna(bpp_bin):
            continue
        pairs, p_e, detection = _group_summary(group)
        cell_rows.append({
            "bit_mode": str(bit_mode),
            "k": int(k_number),
            "bpp_bin": str(bpp_bin),
            "pairs": pairs,
            "P_E": p_e,
            "detection_rate_global": detection,
        })
    cell_table = pd.DataFrame(cell_rows)


    return per_setting, cell_table, BPP_LABELS


def save_heatmaps(cell_table, bpp_labels, predictions_path, *, show=False):
    """Save the four notebook-02 figures, masking cells with fewer than 30 pairs."""
    import matplotlib.pyplot as plt
    BPP_LABELS = bpp_labels
    PRED_PATH = Path(predictions_path)
    def save_heatmap(bit_mode: str, metric: str, file_suffix: str, title: str, vmin: float, vmax: float) -> Path:
        values = np.full((8, len(BPP_LABELS)), np.nan, dtype=np.float64)
        counts = np.zeros((8, len(BPP_LABELS)), dtype=np.int64)
        if not cell_table.empty:
            selected = cell_table.loc[cell_table["bit_mode"] == bit_mode]
            for row in selected.itertuples(index=False):
                row_index = int(row.k) - 1
                column_index = BPP_LABELS.index(row.bpp_bin)
                values[row_index, column_index] = float(getattr(row, metric))
                counts[row_index, column_index] = int(row.pairs)
        visible = np.isfinite(values) & (counts >= 30)
        masked_values = np.ma.masked_where(~visible, values)
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad("#d0d0d0")
        fig, axis = plt.subplots(figsize=(8, 7))
        image = axis.imshow(masked_values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(f"{title}: {bit_mode}")
        axis.set_xlabel("bpp bin")
        axis.set_ylabel("k")
        axis.set_xticks(range(len(BPP_LABELS)), BPP_LABELS)
        axis.set_yticks(range(8), range(1, 9))
        for row_index in range(8):
            for column_index in range(len(BPP_LABELS)):
                count = counts[row_index, column_index]
                value = values[row_index, column_index]
                if count == 0:
                    text = "n=0"
                elif count < 30:
                    text = f"<30\nn={count}"
                else:
                    text = f"{value:.3f}\nn={count}"
                axis.text(column_index, row_index, text, ha="center", va="center", fontsize=8)
        fig.colorbar(image, ax=axis, label=metric)
        fig.tight_layout()
        path = PRED_PATH.with_name(f"{PRED_PATH.stem}_{file_suffix}_{bit_mode}.png")
        fig.savefig(path, dpi=150)
        if show:
            plt.show()
        plt.close(fig)
        return path

    heatmap_paths = []
    for bit_mode in ("lowest", "plane"):
        heatmap_paths.append(save_heatmap(bit_mode, "P_E", "heatmap_pe", "P_E", 0.0, 0.5))
    for bit_mode in ("lowest", "plane"):
        heatmap_paths.append(save_heatmap(bit_mode, "detection_rate_global", "heatmap_detection", "Detection rate at global threshold", 0.0, 1.0))
    return heatmap_paths


def breakdown(predictions_path, *, labels_path=None, show=False):
    """Write notebook-02 per-setting CSV and four heatmaps from saved scores."""
    frame, otf = load_breakdown(predictions_path, labels_path=labels_path)
    summary = global_summary(frame)
    table, cells, labels = breakdown_tables(frame, otf_metadata=otf,
                                           global_threshold=summary['threshold'])
    path = Path(predictions_path)
    table.to_csv(path.with_name(f'{path.stem}_per_setting.csv'), index=False)
    heatmaps = save_heatmaps(cells, labels, path, show=show)
    return summary, table, cells, heatmaps


def extra_panels(checkpoint, *, root, output, panels, workers, device, quick,
                 data_mode, manifest_root=None, source_panels=()):
    """Run requested bpp-1 and source panels using the shared checkpoint scorer."""
    root, output = Path(root), Path(output)
    selected = tuple(panels.split(',')) if isinstance(panels, str) else tuple(panels)
    if not set(selected) <= {'none', 'test-bpp1', 'source-panels'} or ('none' in selected and len(selected)>1):
        raise ValueError('unsupported extra panels')
    results = {}
    if 'test-bpp1' in selected:
        results['test-bpp1'] = evaluate(checkpoint, root/'configs/test-bpp1.yaml',
            root=root, device=device, output=output/'extra-test-bpp1', workers=workers, quick=quick)
    if 'source-panels' in selected:
        if data_mode != 'otf' or not source_panels:
            raise ValueError('source-panels requires OTF and selected new-source manifests')
        for name in source_panels:
            source, split = name.split('-', 1)
            results[name] = evaluate(checkpoint, root/'configs/raise512-run1.yaml',
                root=root, device=device, split=split, source=source, manifest_root=manifest_root,
                output=output/('extra-'+name), workers=workers, quick=quick)
    return results
