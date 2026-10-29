"""Explicit task schemas, RDKit featurization, and deterministic scaffold splits."""
import csv
import hashlib
import json
import math
from pathlib import Path

import torch
from torch_geometric.data import Data

CLASSIFICATION = {"bbbp", "tox21", "toxcast", "sider", "hiv", "bace", "muv"}
REGRESSION = {"lipophilicity", "freesolv", "esol", "qm7", "qm8", "qm9"}
SINGLE_TARGET = {"bbbp": "p_np", "hiv": "HIV_active", "bace": "Class",
                 "lipophilicity": "exp", "freesolv": "expt",
                 "esol": "measured log solubility in mols per litre", "qm7": "u0_atom"}
QM9_TARGETS = ("mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "u0", "u298",
               "h298", "g298", "cv")
FEATURE_VERSION = "rdkit-dense9-v1-no-bond-features"


def discover(root):
    files = {}
    for directory in sorted(Path(root).iterdir()):
        if not directory.is_dir():
            continue
        candidates = sorted((directory / "raw").glob("*.csv"))
        if directory.name not in CLASSIFICATION | REGRESSION:
            raise ValueError(f"Unsupported dataset directory: {directory.name}")
        if len(candidates) != 1:
            raise ValueError(f"Expected one raw CSV in {directory}")
        files[directory.name] = candidates[0]
    if not files:
        raise ValueError("No dataset directories found")
    return files


def schema(name, header):
    smiles = header.index("mol" if name == "bace" else "smiles")
    if name in SINGLE_TARGET:
        targets = [header.index(SINGLE_TARGET[name])]
    elif name == "qm9":
        targets = [header.index(key) for key in QM9_TARGETS]
    elif name == "qm8":
        # Raw QM8 has duplicate PBE0 column names; use the first occurrence only.
        targets = list(dict.fromkeys(header.index(key) for key in header if key != "smiles"))
    else:
        targets = [i for i, key in enumerate(header) if key not in {"smiles", "mol_id"}]
    return smiles, targets


def molecule_graph(smiles):
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None, None
    x = []
    for atom in mol.GetAtoms():
        x.append([atom.GetAtomicNum()/100, atom.GetDegree()/6, atom.GetFormalCharge()/3,
                  atom.GetTotalNumHs()/4, int(atom.GetHybridization())/8,
                  float(atom.GetIsAromatic()), float(atom.IsInRing()), atom.GetMass()/200,
                  int(atom.GetChiralTag())/3])
    edges = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.extend([(i, j), (j, i)])
    edge_index = (torch.tensor(edges, dtype=torch.long).T.contiguous() if edges
                  else torch.empty(2, 0, dtype=torch.long))
    canonical = Chem.MolToSmiles(mol, canonical=True)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    graph = Data(x=torch.tensor(x, dtype=torch.float32), edge_index=edge_index)
    # Exact duplicates and stereo variants of a scaffold cannot cross splits.
    return graph, {"canonical": canonical, "scaffold": scaffold}


def load_dataset(name, path, limit=None):
    from rdkit import rdBase
    graphs, keys, invalid = [], [], []
    with Path(path).open(newline="") as handle, rdBase.BlockLogs():
        reader = csv.reader(handle)
        header = next(reader)
        smiles_col, target_cols = schema(name, header)
        for line, row in enumerate(reader, start=2):
            if not row:
                continue
            if len(row) != len(header):
                invalid.append({"line": line, "reason": "column_count"})
                continue
            values = []
            for column in target_cols:
                try:
                    value = float(row[column])
                except ValueError:
                    value = float("nan")
                values.append(value if math.isfinite(value) else float("nan"))
            if not any(math.isfinite(v) for v in values):
                invalid.append({"line": line, "reason": "no_labels"})
                continue
            if name in CLASSIFICATION and any(math.isfinite(v) and v not in (0., 1.)
                                               for v in values):
                raise ValueError(f"Nonbinary observed target in {name}, line {line}")
            graph, key = molecule_graph(row[smiles_col])
            if graph is None:
                invalid.append({"line": line, "reason": "invalid_smiles"})
                continue
            graph.y = torch.tensor([values], dtype=torch.float32)
            graphs.append(graph)
            keys.append(key)
            if limit is not None and len(graphs) >= limit:
                break
    if len(graphs) < 3:
        raise ValueError(f"Too few valid molecules in {name}")
    return graphs, keys, {"name": name, "kind": "classification" if name in CLASSIFICATION
                         else "regression", "graphs": len(graphs), "invalid": invalid,
                         "target_columns": [header[i] for i in target_cols],
                         "target_indices": target_cols, "features": FEATURE_VERSION,
                         "limited": limit is not None,
                         "csv_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                         "qm8_duplicate_policy": "first_column_per_label" if name == "qm8" else None}


def scaffold_split(keys, seed=0):
    """Seeded scaffold-group 80/10/10 greedy split, with duplicate protection.

    Empty (acyclic) scaffold is one group; dominant groups may skew fractions.
    This is explicitly not claimed identical to the paper's unprovided split IDs.
    """
    groups = {}
    for i, key in enumerate(keys):
        groups.setdefault(key["scaffold"], []).append(i)
    if len(groups) < 3:
        raise ValueError("Need at least three scaffold groups for a leakage-free split")
    def rank(item):
        return (-len(item[1]), hashlib.sha256(f"{seed}:{item[0]}".encode()).hexdigest())
    ordered = sorted(groups.items(), key=rank)
    splits = [[], [], []]
    capacities = [0.8 * len(keys), 0.1 * len(keys), 0.1 * len(keys)]
    for index, (_, group) in enumerate(ordered):
        remaining = len(ordered) - index
        empty = [i for i, part in enumerate(splits) if not part]
        if remaining == len(empty):
            choice = empty[0]
        else:
            choice = max(range(3), key=lambda i: (capacities[i]-len(splits[i])) / capacities[i])
        splits[choice].extend(group)
    return tuple(sorted(part) for part in splits)


def save_metadata(path, metadata):
    Path(path).write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
