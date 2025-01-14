import numpy as np

import pyrosetta
from pyrosetta import rosetta
from biotite.structure import annotate_sse, AtomArray, rmsd, sasa, superimpose
import biotite.structure.io as strucio
from tmtools import tm_align

from fmif.reward_utils import get_backbone_atoms, get_center_of_mass, pose_read_pdb, biotite_read_pdb

pyrosetta.init(options="-mute all")

_HYDROPHOBICS = {"VAL", "ILE", "LEU", "PHE", "MET", "TRP"}


def esm_to_ptm(folding_result: dict):
    """
    folding result = esmfold.infer(sequence)
    """
    return folding_result['ptm'].item()


def esm_to_plddt(folding_result: dict):
    """
    folding result = esmfold.infer(sequence)
    """
    return folding_result['mean_plddt'].item()


def pdb_to_tm(ori_pdb_file, gen_pdb_file):
    """
    maximize tm score
    :param ori_pdb_file / gen_pdb_file: pdb file path, or, esmfold.infer_pdbs(sequence)[0]
    """
    pose_ori_pdb = pose_read_pdb(ori_pdb_file)
    pose_gen_pdb = pose_read_pdb(gen_pdb_file)

    seq_ori = pose_ori_pdb.sequence()
    seq_gen = pose_gen_pdb.sequence()

    ca_coor_ori = []
    for i in range(1, pose_ori_pdb.total_residue() + 1):
        if pose_ori_pdb.residue(i).has("CA"):
            ca_coord = pose_ori_pdb.residue(i).xyz("CA")
            ca_coor_ori.append((ca_coord.x, ca_coord.y, ca_coord.z))
    ca_coor_ori = np.array(ca_coor_ori)

    ca_coor_gen = []
    for i in range(1, pose_gen_pdb.total_residue() + 1):
        if pose_gen_pdb.residue(i).has("CA"):
            ca_coord = pose_gen_pdb.residue(i).xyz("CA")
            ca_coor_gen.append((ca_coord.x, ca_coord.y, ca_coord.z))
    ca_coor_gen = np.array(ca_coor_gen)

    tm_results = tm_align(ca_coor_ori, ca_coor_gen, seq_ori, seq_gen)
    return tm_results.tm_norm_chain1


def pdb_to_rmsd(ori_pdb_file, gen_pdb_file, backbone=True):
    """
    minimize rmsd, if backbone, only consider N,CA,C
    """
    pose_ori_pdb = pose_read_pdb(ori_pdb_file)
    pose_gen_pdb = pose_read_pdb(gen_pdb_file)

    if backbone:
        return rosetta.core.scoring.bb_rmsd(pose_ori_pdb, pose_gen_pdb)
    else:
        return rosetta.core.scoring.all_atom_rmsd(pose_ori_pdb, pose_gen_pdb)


def pdb_to_lddt(ori_pdb_file, gen_pdb_file):
    """
    maximize lddt score
    """
    pose_ori_pdb = pose_read_pdb(ori_pdb_file)
    pose_gen_pdb = pose_read_pdb(gen_pdb_file)

    lddt = rosetta.core.scoring.lddt(pose_ori_pdb, pose_gen_pdb)
    return lddt


def pdb_to_hydrophobic_score(gen_pdb_file, start_residue_index=None, end_residue_index=None):
    """
    Computes ratio of hydrophobic atoms in a biotite AtomArray that are also surface exposed
    Typically, minimize hydrophobic score
    """
    # atom_array = strucio.load_structure(gen_pdb_file)
    atom_array = biotite_read_pdb(gen_pdb_file)

    hydrophobic_mask = np.array([aa in _HYDROPHOBICS for aa in atom_array.res_name])

    if start_residue_index is None and end_residue_index is None:
        selection_mask = np.ones_like(hydrophobic_mask)
    else:
        start_residue_index = 0 if start_residue_index is None else start_residue_index
        end_residue_index = (
            len(hydrophobic_mask) if end_residue_index is None else end_residue_index
        )
        selection_mask = np.array(
            [
                i >= start_residue_index and i < end_residue_index
                for i in range(len(hydrophobic_mask))
            ]
        )

    hydrophobic_surf = np.logical_and(
        selection_mask * hydrophobic_mask, sasa(atom_array)
    )

    return sum(hydrophobic_surf) / sum(selection_mask * hydrophobic_mask)


def pdb_to_match_ss_score(gen_pdb_file, define_sse: str = "a", start=None, end=None):
    """
    whether protein's secondary structure matched predefined class, compute the ratio
    Specify `'a'` for alpha helix, `'b'` for beta sheet, and `'c'` for coils.
    """
    # atom_array = strucio.load_structure(gen_pdb_file)
    atom_array = biotite_read_pdb(gen_pdb_file)

    start = 0 if start is None else start
    end = len(atom_array) if end is None else end

    subprotein = atom_array[
        np.logical_and(
            atom_array.res_id >= start,
            atom_array.res_id < end,
        )
    ]
    sse = annotate_sse(subprotein)

    return np.mean(sse != define_sse)


def pdb_to_surface_expose_score(gen_pdb_file, start=None, end=None):
    """
    maximize surface exposure
    """
    # atom_array = strucio.load_structure(gen_pdb_file)
    atom_array = biotite_read_pdb(gen_pdb_file)

    start = 0 if start is None else start
    end = len(atom_array) if end is None else end

    residue_mask = np.array([res_id in list(range(start, end)) for res_id in atom_array.res_id])
    surface = np.logical_and(residue_mask, sasa(atom_array))

    return 1.0 - sum(surface) / sum(residue_mask)


# def symmetry_score():
#     """
#     protomer is chain
#     """
#     pass


def pdb_to_globularity_score(gen_pdb_file, start=None, end=None):
    """
    maximize globularity score, make it as a ball
    """
    # atom_array = strucio.load_structure(gen_pdb_file)
    atom_array = biotite_read_pdb(gen_pdb_file)

    start = 0 if start is None else start
    end = len(atom_array) if end is None else end

    backbone = get_backbone_atoms(
        atom_array[
            np.logical_and(
                atom_array.res_id >= start,
                atom_array.res_id < end,
            )
        ]
    ).coord

    center_of_mass = get_center_of_mass(backbone)
    m = backbone - center_of_mass
    return float(np.std(np.linalg.norm(m, axis=-1)))


if __name__ == "__main__":
    ori_pdb_file = "/data/xsu2/BioScale/GenProteins/casp15/ori_pdb/T1104-D1.pdb"
    gen_pdb_file = "/data/xsu2/BioScale/GenProteins/casp15/esm3_sm_open_v1/mcts_rollout20_depth2_posk1_sampling10_esm2_8m_esm2_8m/gen_rosettafold2/T1104-D1_idx0/models/model_00_pred.pdb"

