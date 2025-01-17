from typing import Union
import numpy as np
from io import StringIO
from biotite.structure import AtomArray
from biotite.structure.io.pdb import PDBFile

import pyrosetta.rosetta.core.pose as pose
from pyrosetta import pose_from_pdb
from pyrosetta.rosetta.core.import_pose import pose_from_pdbstring


RESIDUE_TYPES_1to3 = {"A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN", "E": "GLU", "G": "GLY",
                      "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO", "S": "SER",
                      "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL"}
RESIDUE_TYPES_3to1 = {v: k for k, v in RESIDUE_TYPES_1to3.items()}


def pose_read_pdb(pdb_file, filter_by_CA=True):
    """
    pdb file path, or, esmfold.infer_pdbs(sequence)[0]
    """
    assert isinstance(pdb_file, str)
    if pdb_file.endswith('.pdb'):
        pose_pdb = pose_from_pdb(pdb_file)
    else:
        pose_pdb = pose.Pose()
        pose_from_pdbstring(pose_pdb, pdb_file)

    if not filter_by_CA:
        return pose_pdb
    filtered_pose = pose.Pose()
    for i in range(1, pose_pdb.total_residue() + 1):
        if pose_pdb.residue(i).has("CA"):
            filtered_pose.append_residue_by_bond(pose_pdb.residue(i))

    return filtered_pose


def pdb_file_to_atomarray(pdb_path: Union[str, StringIO]) -> AtomArray:
    return PDBFile.read(pdb_path).get_structure()[0]


def sequence_from_atomarray(atoms: AtomArray) -> str:
    return "".join(
        [RESIDUE_TYPES_3to1[aa] for aa in atoms[atoms.atom_name == "CA"].res_name]
    )


def _is_Nx3(array: np.ndarray) -> bool:
    return len(array.shape) == 2 and array.shape[1] == 3


def pairwise_distances(coordinates: np.ndarray) -> np.ndarray:
    assert _is_Nx3(coordinates), "Coordinates must be Nx3."
    m = coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]
    distance_matrix = np.linalg.norm(m, axis=-1)
    return distance_matrix[np.triu_indices(distance_matrix.shape[0], k=1)]


def adjacent_distances(coordinates: np.ndarray) -> np.ndarray:
    assert _is_Nx3(coordinates), "Coordinates must be Nx3."
    m = coordinates - np.roll(coordinates, shift=1, axis=0)
    return np.linalg.norm(m, axis=-1)


def get_backbone_atoms(atoms: AtomArray) -> AtomArray:
    return atoms[
        (atoms.atom_name == "CA") | (atoms.atom_name == "N") | (atoms.atom_name == "C")
    ]


def get_center_of_mass(coordinates: np.ndarray) -> np.ndarray:
    assert _is_Nx3(coordinates), "Coordinates must be Nx3."
    return coordinates.mean(axis=0).reshape(1, 3)
