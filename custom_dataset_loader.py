"""Custom e-nose dataset loader for Machine 4 (49 gas sensors, 9 gases).

Matches the interface of uci_dataset_loader.py for drop-in compatibility.
"""
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
import os, re

# ── Constants ─────────────────────────────────────────────────────
GAS_CLASSES = ['acetaldehyde', 'acetic_acid', 'acetone', 'benzaldehyde',
               'ethanol', 'ethyl_acetate', 'isobutanol', 'n_hexane', 'thf']

NUM_SENSORS = 49
NUM_TIMESTEPS = 8  # downsample P15 (300s) to 8 time steps
NUM_NODE_FEATURES = NUM_TIMESTEPS + NUM_SENSORS  # 8 temporal + 49 one-hot ID

# 49 sensor column indices (0-based) in the 86-column row
# 37 basic (cols 12-48) + 4 ext1 (56-59) + 3 ext2 (69-71) + 3 optical (76-78) + 2 alcohol (81-82)
SENSOR_COLS = list(range(12, 49)) + list(range(56, 60)) + list(range(69, 72)) \
              + list(range(76, 79)) + list(range(81, 83))

# Stage column index
STAGE_COL = 4

# Gas name mapping (pinyin/abbreviation -> canonical)
_GAS_MAP = {
    'bingtong': 'acetone', 'bt': 'acetone',
    'yichun': 'ethanol', 'yc': 'ethanol',
    'yisuanyizhi': 'ethyl_acetate', 'ysyz': 'ethyl_acetate',
    'yiquan': 'acetaldehyde', 'yq': 'acetaldehyde',
    'yisuan': 'acetic_acid', 'ys': 'acetic_acid',
    'siqingfunan': 'thf', 'sqfn': 'thf',
    'yibinchun': 'isobutanol', 'ybc': 'isobutanol',
    'zhenjiwan': 'n_hexane', 'zhengjiwan': 'n_hexane', 'zjw': 'n_hexane',
    'b': 'benzaldehyde', 'ben': 'benzaldehyde',
}

# Month directories (partial match)
_MONTH_DIRS = {
    '2025.9': 'M1', '2025.10': 'M2', '2025.11': 'M3',
    '2025.12': 'M4', '2026.01': 'M5',
}


# ── File discovery ─────────────────────────────────────────────────
def discover_files(data_dir, month_label, concentration=None):
    """Walk data_dir for a given month's Machine-4 .txt files.

    Args:
        concentration: if int (30/50/100), only return files of that ppm.
    """
    # Find the month directory
    month_dir = None
    for entry in os.listdir(data_dir):
        full = os.path.join(data_dir, entry)
        if not os.path.isdir(full):
            continue
        if month_label in _MONTH_DIRS.values():
            # Reverse lookup
            for dkey, dval in _MONTH_DIRS.items():
                if dval == month_label and dkey in entry:
                    month_dir = full
                    break
        if month_dir:
            break

    if month_dir is None:
        raise FileNotFoundError(f"Month {month_label} not found in {data_dir}")

    # Find Machine-4 subdirectory
    machine_dir = None
    for entry in os.listdir(month_dir):
        if '样机4' in entry or ' Machine 4' in entry:
            machine_dir = os.path.join(month_dir, entry)
            break
    if machine_dir is None:
        raise FileNotFoundError(f"Machine 4 dir not found in {month_dir}")

    files = []
    for root, dirs, filenames in os.walk(machine_dir):
        for f in filenames:
            if not f.endswith('.txt'):
                continue
            if 'ERROR' in f or 'ERROE' in f:
                continue
            if '副本' in f:
                continue
            if 'huangqin' in f.lower() or 'yangji' in f.lower():
                continue

            basename = f.replace('.txt', '')
            tokens = basename.split('-')

            # Extract concentration
            conc = None
            for t in tokens:
                m = re.match(r'(\d+)ppm', t)
                if m:
                    conc = int(m.group(1))
                    break
            if conc is None:
                continue

            # Optional concentration filter
            if concentration is not None and conc != concentration:
                continue

            # Extract repetition
            rep = None
            for t in reversed(tokens):
                if t in ['1', '2', '3']:
                    rep = int(t)
                    break
            if rep is None:
                continue

            # Extract gas
            gas = None
            for t in tokens:
                tl = t.lower()
                if tl in _GAS_MAP:
                    gas = _GAS_MAP[tl]
                    break
            if gas is None:
                subdir = os.path.basename(root)
                if subdir in _GAS_MAP:
                    gas = _GAS_MAP[subdir]
            if gas is None:
                continue

            files.append((os.path.join(root, f), gas, conc, rep))

    return files


# ── Core loading ───────────────────────────────────────────────────
def load_custom_dataset(data_dir, month_label, scaler=None, feature_mode='temporal',
                        concentration=None):
    """Load all Machine-4 .txt files for a month, extract features, return X, y.

    Args:
        data_dir: path to the 2025校准/ directory
        month_label: 'M1' through 'M5'
        scaler: if None, fit new StandardScaler; else transform with it
        feature_mode: 'temporal' (方案B: raw P15 downsample) or
                      'handcrafted' (方案A: 8 engineered features)
        concentration: if int, only load that ppm (30/50/100); None = all

    Returns:
        X: np.ndarray [N_samples, NUM_SENSORS * NUM_FEATURES_PER_SENSOR]
        y: np.ndarray [N_samples] class labels 0-8
        scaler: StandardScaler (fitted or passed-through)
    """
    if feature_mode not in ('temporal', 'handcrafted'):
        raise ValueError(f"Unknown feature_mode: {feature_mode}")

    files = discover_files(data_dir, month_label, concentration=concentration)
    if not files:
        raise RuntimeError(f"No files found for {month_label} in {data_dir}")

    extract_fn = _extract_temporal_features if feature_mode == 'temporal' \
                 else _extract_handcrafted_features

    X_list, y_list = [], []

    for filepath, gas_name, conc, rep in files:
        label = GAS_CLASSES.index(gas_name)

        # Parse the file into stage-separated data
        try:
            features = extract_fn(filepath)
        except Exception as e:
            print(f"  [SKIP] {os.path.basename(filepath)}: {e}")
            continue

        X_list.append(features)
        y_list.append(label)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    # Replace NaN/Inf with 0
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Z-score normalization
    if scaler is None:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)

    return X, y, scaler


def _extract_temporal_features(filepath):
    """Parse one .txt file, extract P15 temporal features for 49 sensors.

    Returns: np.ndarray [NUM_SENSORS * NUM_TIMESTEPS] flattened.
    """
    # Read all rows, split by stage
    stages = {'P13': [], 'P15': [], 'P18': []}
    with open(filepath, 'r') as f:
        for line in f:
            cols = line.strip().split()
            if len(cols) < max(SENSOR_COLS) + 1:
                continue
            stage = cols[STAGE_COL]
            if stage in stages:
                # Extract only the 49 sensor columns
                sensor_vals = [float(cols[i]) for i in SENSOR_COLS]
                stages[stage].append(sensor_vals)

    if len(stages['P13']) == 0 or len(stages['P15']) == 0:
        raise ValueError("Missing P13 or P15 stage")

    # Compute P13 baseline mean per sensor
    p13_arr = np.array(stages['P13'], dtype=np.float32)  # [T_p13, 49]
    baseline = np.mean(p13_arr, axis=0)  # [49]

    # Downsample P15 to NUM_TIMESTEPS equally-spaced time points
    p15_arr = np.array(stages['P15'], dtype=np.float32)  # [T_p15, 49]
    t_p15 = p15_arr.shape[0]
    indices = np.linspace(0, t_p15 - 1, NUM_TIMESTEPS, dtype=int)
    p15_down = p15_arr[indices, :]  # [NUM_TIMESTEPS, 49]

    # Subtract baseline (drift compensation)
    p15_normalized = p15_down - baseline[np.newaxis, :]  # [8, 49]

    # Flatten: [49 sensors * 8 timesteps] = [392]
    features = p15_normalized.T.flatten()  # sensor-major: s0_t0, s0_t1, ..., s0_t7, s1_t0, ...

    return features


def _extract_handcrafted_features(filepath):
    """Parse one .txt file, extract 8 hand-crafted features for 49 sensors.

    Features (per sensor):
      1. P13 baseline mean
      2. P13 baseline std
      3. P15 response amplitude (max - baseline)
      4. Normalized response (amplitude / baseline)
      5. P15 steady-state (mean of last 30s)
      6. P15 max rising slope
      7. P18 recovery rate (last 30s / baseline)
      8. P15 response AUC (sum of baseline-subtracted signal)

    Returns: np.ndarray [NUM_SENSORS * 8] flattened (sensor-major).
    """
    stages = {'P13': [], 'P15': [], 'P18': []}
    with open(filepath, 'r') as f:
        for line in f:
            cols = line.strip().split()
            if len(cols) < max(SENSOR_COLS) + 1:
                continue
            stage = cols[STAGE_COL]
            if stage in stages:
                sensor_vals = [float(cols[i]) for i in SENSOR_COLS]
                stages[stage].append(sensor_vals)

    if len(stages['P13']) == 0 or len(stages['P15']) == 0:
        raise ValueError("Missing P13 or P15 stage")

    p13 = np.array(stages['P13'], dtype=np.float32)  # [T_p13, 49]
    p15 = np.array(stages['P15'], dtype=np.float32)  # [T_p15, 49]

    baseline_mean = np.mean(p13, axis=0)              # [49]
    baseline_std = np.std(p13, axis=0)                # [49]

    # Feature 3-4: response amplitude
    p15_max = np.max(p15, axis=0)                     # [49]
    delta_r = p15_max - baseline_mean                  # [49]
    norm_response = delta_r / (baseline_mean + 1e-8)  # [49]

    # Feature 5: steady-state (last 30 seconds of P15)
    ss_window = min(30, p15.shape[0])
    steady_state = np.mean(p15[-ss_window:, :], axis=0)  # [49]

    # Feature 6: max rising slope
    slopes = np.diff(p15, axis=0)                     # [T-1, 49]
    max_slope = np.max(slopes, axis=0)                # [49]

    # Feature 7: recovery rate
    has_p18 = len(stages['P18']) > 0
    if has_p18:
        p18 = np.array(stages['P18'], dtype=np.float32)
        rw_window = min(30, p18.shape[0])
        recovery_end = np.mean(p18[-rw_window:, :], axis=0)  # [49]
        recovery_rate = recovery_end / (baseline_mean + 1e-8) # [49]
    else:
        recovery_rate = np.ones(NUM_SENSORS, dtype=np.float32)

    # Feature 8: response AUC (sum of P15 - baseline)
    auc = np.sum(p15 - baseline_mean[np.newaxis, :], axis=0)  # [49]

    # Stack: [49, 8] → flatten sensor-major → [392]
    features = np.stack([
        baseline_mean, baseline_std, delta_r, norm_response,
        steady_state, max_slope, recovery_rate, auc
    ], axis=1)  # [49, 8]

    return features.T.flatten()  # sensor-major: s0_f0...s0_f7, s1_f0...s1_f7, ...


# ── Graph construction ─────────────────────────────────────────────
def build_physical_adjacency_custom(X_source, threshold=0.3):
    """Compute sensor physical adjacency matrix Rs from source data.

    X_source: [N, 392] where 392 = 49 sensors * 8 timesteps.
    For correlation, use the steady-state (last timestep) of each sensor.
    """
    N = X_source.shape[0]
    # Reshape to [N, 49, 8], take last timestep as steady-state
    X_reshaped = X_source.reshape(N, NUM_SENSORS, NUM_TIMESTEPS)
    steady_state = X_reshaped[:, :, -1]  # [N, 49]

    correlation_matrix = np.corrcoef(steady_state.T)
    correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)

    R_s = np.abs(correlation_matrix)
    R_s[R_s < threshold] = 0.0
    np.fill_diagonal(R_s, 1.0)

    return R_s


def convert_to_pyg_graphs_custom(X, y, R_s_tensor, domain_id):
    """Convert numpy arrays to PyG Data objects (49-sensor graphs).

    Args:
        X: [N, 392] feature matrix
        y: [N] labels
        R_s_tensor: [49, 49] physical adjacency matrix
        domain_id: 0 for source, 1 for target

    Returns:
        List of torch_geometric.data.Data
    """
    N = X.shape[0]
    X_reshaped = X.reshape(N, NUM_SENSORS, NUM_TIMESTEPS)  # [N, 49, 8]

    # Build edge index from Rs (only edges where Rs > 0)
    edge_index = []
    edge_attr = []
    for i in range(NUM_SENSORS):
        for j in range(NUM_SENSORS):
            if R_s_tensor[i, j] > 0:
                edge_index.append([i, j])
                edge_attr.append(R_s_tensor[i, j].item())

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32).unsqueeze(-1)

    # 49-dim one-hot sensor identity
    one_hot_id = torch.eye(NUM_SENSORS, dtype=torch.float32)

    graph_list = []
    for i in range(N):
        raw_features = torch.tensor(X_reshaped[i], dtype=torch.float32)  # [49, 8]
        node_features = torch.cat([raw_features, one_hot_id], dim=1)     # [49, 57]

        label = torch.tensor([y[i]], dtype=torch.long)
        domain = torch.tensor([domain_id], dtype=torch.long)
        sample_id = torch.tensor([i], dtype=torch.long)

        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr,
                    y=label, domain=domain, id=sample_id)
        graph_list.append(data)

    return graph_list


def convert_to_pyg_graphs_custom_pure_attention(X, y, domain_id):
    """Convert to PyG graphs with full attention (no edge weights)."""
    N = X.shape[0]
    X_reshaped = X.reshape(N, NUM_SENSORS, NUM_TIMESTEPS)

    # Fully connected graph
    edge_index = []
    for i in range(NUM_SENSORS):
        for j in range(NUM_SENSORS):
            edge_index.append([i, j])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    one_hot_id = torch.eye(NUM_SENSORS, dtype=torch.float32)

    graph_list = []
    for i in range(N):
        raw_features = torch.tensor(X_reshaped[i], dtype=torch.float32)
        node_features = torch.cat([raw_features, one_hot_id], dim=1)

        label = torch.tensor([y[i]], dtype=torch.long)
        domain = torch.tensor([domain_id], dtype=torch.long)
        sample_id = torch.tensor([i], dtype=torch.long)

        data = Data(x=node_features, edge_index=edge_index, y=label, domain=domain, id=sample_id)
        graph_list.append(data)

    return graph_list
