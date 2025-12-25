# export_md17_lmdb_to_xyz.py
from fairchem.core.datasets.base_dataset import create_dataset
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write
import os

ROOT = "/home/siripab/fairchem/data/rmd17/lmdb/aspirin"
OUT  = "/home/siripab/fairchem/data/rmd17_xyz/aspirin"

for split in ["train", "val", "test"]:
    print(f"Processing {split}...")

    ds = create_dataset(
        config={
            "format": "fairchem.core.datasets.rmd17.MD17",
            "splits": {split: {"src": f"{ROOT}/{split}"}},
        },
        split=split,
    )

    atoms_list = []

    for d in ds:
        atoms = Atoms(
            numbers=d.atomic_numbers.cpu().numpy(),
            positions=d.pos.cpu().numpy(),
        )

        calc = SinglePointCalculator(
            atoms=atoms,
            energy=float(d.y),
            forces=d.force.cpu().numpy(),
        )
        atoms.calc = calc
        atoms.info["sid"] = d.sid.item() if hasattr(d.sid, "item") else d.sid
        atoms_list.append(atoms)

    os.makedirs(OUT, exist_ok=True)
    write(f"{OUT}/{split}.xyz", atoms_list)
    print(f"Saved {OUT}/{split}.xyz")
