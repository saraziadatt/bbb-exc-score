#!/usr/bin/env python3
"""Remove full-BBBP overlap from an Inxight CSV using pandas and RDKit.

The released ImageMol checkpoint's exact training split is unknown. This screens
against ALL reference rows, not a verified fine-tuning training list. Default
matching uses the neutralized largest organic parent, RDKit's HetAtomTautomerv2
tautomer hash, and
ignores stereochemistry, isotopes, and atom maps. It is deliberately conservative.
For reference SMILES with invalid valence but readable connectivity, compare
heavy-atom element/connectivity graphs, ignoring bond orders and charges; these
broader matches are explicitly identified in the audit. Unreadable reference
SMILES or invalid Inxight SMILES stop the run; nothing is silently skipped.
Original SMILES, labels, IDs, columns, values, and row order are preserved in the
retained/removed CSVs. Neither class balancing nor other dataset filtering occurs.

Reference download:
https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import rdMolHash
from rdkit.Chem.MolStandardize import rdMolStandardize as std

PARAMS = std.CleanupParameters()
PARAMS.preferOrganic = True


def read_csv(path, smiles_col):
    # Preserve text values (including leading-zero IDs and literal NA strings).
    with open(path, encoding="utf-8-sig", newline="") as f:
        header = next(csv.reader(f), [])
    if not header or len(set(header)) != len(header):
        raise ValueError(f"Empty or duplicated CSV headers: {path}")
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if smiles_col not in df or df.empty:
        raise ValueError(f"Need nonempty {path} with a {smiles_col!r} column.")
    return df


def parse_smiles(smiles):
    if not smiles.strip():
        raise ValueError("Empty SMILES")
    with rdBase.BlockLogs():
        raw = Chem.MolFromSmiles(smiles.strip(), sanitize=False)
        if raw is None or not raw.GetNumAtoms():
            raise ValueError("Unreadable SMILES")
        if any(a.GetAtomicNum() == 0 for a in raw.GetAtoms()):
            raise ValueError("Wildcard atoms do not specify a molecule")
        valid = Chem.MolFromSmiles(smiles.strip())
    return raw, valid


def parent_key(mol):
    mol = Chem.Mol(mol)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
        atom.SetIsotope(0)
    Chem.RemoveStereochemistry(mol)
    mol = std.ChargeParent(Chem.RemoveHs(mol), PARAMS)
    # Non-enumerative tautomer grouping avoids incomplete enumeration on complex
    # BBBP molecules. This is a matching key, not a SMILES to feed into the model.
    if not hasattr(rdMolHash.HashFunction, "HetAtomTautomerv2"):
        raise RuntimeError("This RDKit version lacks HetAtomTautomerv2; use a current RDKit release.")
    return rdMolHash.MolHash(mol, rdMolHash.HashFunction.HetAtomTautomerv2)


def connectivity_keys(mol):
    """Largest organic fragment(s), keeping ties; element/adjacency only."""
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    ranks = [(any(a.GetAtomicNum() == 6 for a in f.GetAtoms()), f.GetNumHeavyAtoms()) for f in fragments]
    keys = set()
    for fragment, rank in zip(fragments, ranks):
        if rank != max(ranks) or rank[1] == 0:
            continue
        graph, indices = Chem.RWMol(), {}
        for atom in fragment.GetAtoms():
            if atom.GetAtomicNum() > 1:
                new_atom = Chem.Atom(atom.GetAtomicNum())
                new_atom.SetNoImplicit(True)
                indices[atom.GetIdx()] = graph.AddAtom(new_atom)
        for bond in fragment.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if i in indices and j in indices:
                graph.AddBond(indices[i], indices[j], Chem.BondType.SINGLE)
        keys.add(Chem.MolToSmiles(graph.GetMol(), canonical=True, isomericSmiles=False))
    if not keys:
        raise ValueError("No heavy atoms for connectivity matching")
    return keys


def compare(inxight, reference, inxight_col="SMILES", reference_col="smiles"):
    parents, fallback = defaultdict(list), defaultdict(list)
    reference_audit = []
    for i, smiles in enumerate(reference[reference_col]):
        try:
            raw, valid = parse_smiles(smiles)
            if valid is None:
                keys = sorted(connectivity_keys(raw))
                for key in keys:
                    fallback[key].append(i)
                mode = "connectivity_fallback_invalid_reference"
            else:
                keys = [parent_key(valid)]
                parents[keys[0]].append(i)
                mode = "standardized_parent"
        except Exception as exc:
            raise ValueError(f"BBBP data row {i} (0-based): {smiles!r}: {exc}") from exc
        reference_audit.append({"bbbp_row_position": i, "bbbp_smiles": smiles, "matching_mode": mode, "match_keys": json.dumps(keys)})
    audit = []
    for i, smiles in enumerate(inxight[inxight_col]):
        try:
            raw, valid = parse_smiles(smiles)
            if valid is None:
                raise ValueError("Invalid Inxight SMILES; correct it before filtering")
            key = parent_key(valid)
            direct = set(parents.get(key, []))
            broad = {j for k in connectivity_keys(raw) for j in fallback.get(k, [])} if fallback else set()
        except Exception as exc:
            raise ValueError(f"Inxight data row {i} (0-based): {smiles!r}: {exc}") from exc
        audit.append({"inxight_row_position": i, "inxight_smiles": smiles, "removed": bool(direct or broad), "standardized_parent": key,
                      "parent_match_bbbp_rows": json.dumps(sorted(direct)), "fallback_match_bbbp_rows": json.dumps(sorted(broad)),
                      "matched_bbbp_smiles": json.dumps(reference.iloc[sorted(direct | broad)][reference_col].tolist())})
    return pd.DataFrame(audit), pd.DataFrame(reference_audit)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inxight-csv", type=Path, required=True)
    parser.add_argument("--bbbp-csv", type=Path, required=True)
    parser.add_argument("--smiles-col", default="SMILES")
    parser.add_argument("--bbbp-smiles-col", default="smiles")
    parser.add_argument("--expected-rows", type=int, default=210, help="Expected Inxight rows; 0 disables this check")
    parser.add_argument("--output", type=Path, default=Path("../datasets/inxight_without_bbbp.csv"))
    args = parser.parse_args()
    if args.expected_rows < 0:
        parser.error("--expected-rows must be >= 0")
    source = read_csv(args.inxight_csv, args.smiles_col)
    reference = read_csv(args.bbbp_csv, args.bbbp_smiles_col)
    if args.expected_rows and len(source) != args.expected_rows:
        parser.error(f"Expected {args.expected_rows} Inxight rows, found {len(source)}. Check the input file.")
    outputs = [args.output] + [args.output.with_name(args.output.stem + suffix + ".csv") for suffix in ("_removed", "_audit", "_reference_audit")]
    inputs = {args.inxight_csv.resolve(), args.bbbp_csv.resolve()}
    if any(p.resolve() in inputs or p.exists() for p in outputs):
        parser.error("An output already exists or overlaps an input; choose a new --output path.")
    audit, reference_audit = compare(source, reference, args.smiles_col, args.bbbp_smiles_col)
    removed = audit.removed.to_numpy(dtype=bool)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for df, path in zip((source.loc[~removed], source.loc[removed], audit, reference_audit), outputs):
        df.to_csv(path, index=False)
    print(f"RDKit {rdBase.rdkitVersion}; BBBP reference: {len(reference)} rows; Inxight: {len(source)} rows")
    print(f"Removed {removed.sum()}; retained {(~removed).sum()}")
    print(f"Malformed BBBP rows checked by connectivity: {(reference_audit.matching_mode != 'standardized_parent').sum()}")
    print(f"Inxight rows with connectivity fallback matches: {(audit.fallback_match_bbbp_rows != '[]').sum()}")
    print("Reference scope: full supplied BBBP dataset; checkpoint-specific training membership is unverified.")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
