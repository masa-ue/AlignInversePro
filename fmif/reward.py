import numpy as np
import copy
from io import StringIO

import pyrosetta
from pyrosetta import rosetta
from biotite.structure import annotate_sse, AtomArray, rmsd, sasa, superimpose
import biotite.structure.io as strucio
from tmtools import tm_align

from fmif.reward_utils import *

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
            # seq_ori.append(pose_ori_pdb.sequence()[i - 1])
    ca_coor_ori = np.array(ca_coor_ori)
    # seq_ori = ''.join(seq_ori)

    ca_coor_gen = []
    for i in range(1, pose_gen_pdb.total_residue() + 1):
        if pose_gen_pdb.residue(i).has("CA"):
            ca_coord = pose_gen_pdb.residue(i).xyz("CA")
            ca_coor_gen.append((ca_coord.x, ca_coord.y, ca_coord.z))
            # seq_gen.append(pose_gen_pdb.sequence()[i - 1])
    ca_coor_gen = np.array(ca_coor_gen)
    # seq_gen = ''.join(seq_gen)

    tm_results = tm_align(ca_coor_ori, ca_coor_gen, seq_ori, seq_gen)
    return tm_results.tm_norm_chain1


def pdb_to_crmsd(ori_pdb_file, gen_pdb_file, backbone=True):
    """
    minimize rmsd, if backbone, only consider N,CA,C
    """
    pose_ori_pdb = pose_read_pdb(ori_pdb_file)
    pose_gen_pdb = pose_read_pdb(gen_pdb_file)

    if backbone:
        return rosetta.core.scoring.bb_rmsd(pose_ori_pdb, pose_gen_pdb)
    else:
        return rosetta.core.scoring.all_atom_rmsd(pose_ori_pdb, pose_gen_pdb)


def pdb_to_drmsd(ori_pdb_file, gen_pdb_file, backbone=True):
    atom_gen = pdb_file_to_atomarray(gen_pdb_file)
    atom_ori = pdb_file_to_atomarray(ori_pdb_file)

    if backbone:
        atom_gen = get_backbone_atoms(atom_gen)
        atom_ori = get_backbone_atoms(atom_ori)

    dp = pairwise_distances(atom_gen.coord)
    dq = pairwise_distances(atom_ori.coord)

    return float(np.sqrt(((dp - dq) ** 2).mean()))


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
    atom_array = pdb_file_to_atomarray(gen_pdb_file)

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
    atom_array = pdb_file_to_atomarray(gen_pdb_file)

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
    atom_array = pdb_file_to_atomarray(gen_pdb_file)

    start = 0 if start is None else start
    end = len(atom_array) if end is None else end

    residue_mask = np.array([res_id in list(range(start, end)) for res_id in atom_array.res_id])
    surface = np.logical_and(residue_mask, sasa(atom_array))

    return 1.0 - sum(surface) / sum(residue_mask)


def symmetry_score(gen_pdb_file, starts, ends, all_to_all_protomer_symmetry=False):
    """
    starts: start residue index list
    ends: end residue index list
    """
    atom_array = pdb_file_to_atomarray(gen_pdb_file)

    assert len(starts) == len(ends)
    centers_of_mass = []
    for i in range(len(starts)):
        start, end = starts[i], ends[i]
        backbone_coordinates = get_backbone_atoms(
            atom_array[
                np.logical_and(
                    atom_array.res_id >= start,
                    atom_array.res_id < end,
                )
            ]
        ).coord
        centers_of_mass.append(get_center_of_mass(backbone_coordinates))
    centers_of_mass = np.vstack(centers_of_mass)

    return (
        float(np.std(pairwise_distances(centers_of_mass)))
        if all_to_all_protomer_symmetry
        else float(np.std(adjacent_distances(centers_of_mass)))
    )


def pdb_to_globularity_score(gen_pdb_file, start=None, end=None):
    """
    maximize globularity score, make it as a ball
    """
    # atom_array = strucio.load_structure(gen_pdb_file)
    atom_array = pdb_file_to_atomarray(gen_pdb_file)

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
    # ori_pdb_file = "/data/xsu2/BioScale/GenProteins/casp15/ori_pdb/T1104-D1.pdb"
    # gen_pdb_file = "/data/xsu2/BioScale/GenProteins/casp15/esm3_sm_open_v1/mcts_rollout20_depth2_posk1_sampling10_esm2_8m_esm2_8m/gen_rosettafold2/T1104-D1_idx0/models/model_00_pred.pdb"
    from biotite.database.rcsb import fetch
    import esm2

    ALL_RESIDUE_TYPES = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
    RESIDUE_TYPES_WITHOUT_CYSTEINE = copy.deepcopy(ALL_RESIDUE_TYPES)
    RESIDUE_TYPES_WITHOUT_CYSTEINE.remove("C")

    template_pdb_file = fetch("6mrs", format="pdb")
    pdb_value_str = template_pdb_file.getvalue()
    template_atoms: AtomArray = pdb_file_to_atomarray(StringIO(pdb_value_str))
    sequence_length = len(sequence_from_atomarray(template_atoms))

    random_seq = "".join([np.random.choice(RESIDUE_TYPES_WITHOUT_CYSTEINE) for _ in range(sequence_length)])

    # esmfold
    folding_model = esm2.esm.pretrained.esmfold_structure_module_only_3B().eval()
    output = folding_model.infer(random_seq)
    pdbs = folding_model.output_to_pdb(output)

    # metrics
    ptm = esm_to_ptm(output)
    plddt = esm_to_plddt(output)
    tm = pdb_to_tm(pdb_value_str, pdbs[0])
    crmsd = pdb_to_crmsd(pdb_value_str, pdbs[0])
    drmsd = pdb_to_drmsd(StringIO(pdb_value_str), StringIO(pdbs[0]))
    lddt = pdb_to_lddt(pdb_value_str, pdbs[0])
    hydrophobic = pdb_to_hydrophobic_score(StringIO(pdbs[0]))
    match_ss = pdb_to_match_ss_score(StringIO(pdbs[0]))
    surface_expose = pdb_to_surface_expose_score(StringIO(pdbs[0]))
    globularity = pdb_to_globularity_score(StringIO(pdbs[0]))




