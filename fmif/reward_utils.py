import numpy as np
from io import StringIO
from biotite.structure import AtomArray
from biotite.structure.io.pdb import PDBFile

import pyrosetta.rosetta.core.pose as pose
from pyrosetta import pose_from_pdb
from pyrosetta.rosetta.core.import_pose import pose_from_pdbstring


def pose_read_pdb(pdb_file):
    """
    pdb file path, or, esmfold.infer_pdbs(sequence)[0]
    """
    assert isinstance(pdb_file, str)
    if pdb_file.endswith('.pdb'):
        pose_pdb = pose_from_pdb(pdb_file)
    else:
        pose_pdb = pose.Pose()
        pose_from_pdbstring(pose_pdb, pdb_file)

    return pose_pdb


def biotite_read_pdb(pdb_file):
    """
    pdb_file: pdb file path, or, StringIO(esmfold.infer_pdbs(sequence)[0])
    """
    file = PDBFile.read(pdb_file)
    atom_array = file.get_structure()[0]
    return atom_array


def _is_Nx3(array: np.ndarray) -> bool:
    return len(array.shape) == 2 and array.shape[1] == 3


def get_backbone_atoms(atoms: AtomArray) -> AtomArray:
    return atoms[
        (atoms.atom_name == "CA") | (atoms.atom_name == "N") | (atoms.atom_name == "C")
    ]


def get_center_of_mass(coordinates: np.ndarray) -> np.ndarray:
    assert _is_Nx3(coordinates), "Coordinates must be Nx3."
    return coordinates.mean(axis=0).reshape(1, 3)
