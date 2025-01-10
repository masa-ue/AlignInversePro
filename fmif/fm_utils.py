import torch
import copy
# import math
# import functools as fn
import torch.nn.functional as F
from collections import defaultdict
# from multiflow.data import so3_utils, all_atom
# from multiflow.data import utils as du
from fmif import model_utils as mu
import numpy as np
from tqdm import tqdm
# from scipy.spatial.transform import Rotation
# from scipy.optimize import linear_sum_assignment
# from torch import autograd
# from torch.distributions.categorical import Categorical
# from torch.distributions.binomial import Binomial


# def _centered_gaussian(num_batch, num_res, device):
#     noise = torch.randn(num_batch, num_res, 3, device=device)
#     return noise - torch.mean(noise, dim=-2, keepdims=True)


# def _uniform_so3(num_batch, num_res, device):
#     return torch.tensor(
#         Rotation.random(num_batch*num_res).as_matrix(),
#         device=device,
#         dtype=torch.float32,
#     ).reshape(num_batch, num_res, 3, 3)


def _masked_categorical(num_batch, num_res, device):
    return torch.ones(
        num_batch, num_res, device=device) * mu.MASK_TOKEN_INDEX


def _sample_categorical(categorical_probs):
    gumbel_norm = (
        1e-10
        - (torch.rand_like(categorical_probs) + 1e-10).log())
    return (categorical_probs / gumbel_norm).argmax(dim=-1)


def _sample_categorical_gradient(categorical_probs, temp = 1.0):
    gumbel_norm = (
        1e-10
        - (torch.rand_like(categorical_probs) + 1e-10).log())
    # print(categorical_probs)
    # output = torch.log(categorical_probs)
    # output = output - torch.log(gumbel_norm)
    # output = output / temp
    # output = torch.nn.functional.softmax(output, 2)

    output = torch.nn.functional.softmax((torch.log(categorical_probs)-torch.log(gumbel_norm))/temp, 2)
    
    # straight-through estimator leads to exploding gradients
    # output_argmax = (categorical_probs / gumbel_norm).argmax(dim=-1)
    # output_argmax = torch.nn.functional.one_hot(output_argmax, num_classes=categorical_probs.shape[-1])
    # return output - output.detach() + output_argmax.detach()
    return output


# def _trans_diffuse_mask(trans_t, trans_1, diffuse_mask):
#     return trans_t * diffuse_mask[..., None] + trans_1 * (1 - diffuse_mask[..., None])


# def _rots_diffuse_mask(rotmats_t, rotmats_1, diffuse_mask):
#     return (
#         rotmats_t * diffuse_mask[..., None, None]
#         + rotmats_1 * (1 - diffuse_mask[..., None, None])
#     )


# def _aatypes_diffuse_mask(aatypes_t, aatypes_1, diffuse_mask):
#     return aatypes_t * diffuse_mask + aatypes_1 * (1 - diffuse_mask)


class Interpolant:

    def __init__(self, cfg):
        self._cfg = cfg
        # self._rots_cfg = cfg.rots
        # self._trans_cfg = cfg.trans
        # self._aatypes_cfg = cfg.aatypes
        # self._sample_cfg = cfg.sampling
        # self._igso3 = None

        # self.num_tokens = 21 if self._aatypes_cfg.interpolant_type == "masking" else 20
        self.num_tokens = 22
        self.neg_infinity = -1000000.0


    # @property
    # def igso3(self):
    #     if self._igso3 is None:
    #         sigma_grid = torch.linspace(0.1, 1.5, 1000)
    #         self._igso3 = so3_utils.SampleIGSO3(
    #             1000, sigma_grid, cache_dir='.cache')
    #     return self._igso3

    def set_device(self, device):
        self._device = device

    def sample_t(self, num_batch):
        t = torch.rand(num_batch, device=self._device)
        return t * (1 - 2*self._cfg.min_t) + self._cfg.min_t

    # def _corrupt_trans(self, trans_1, t, res_mask, diffuse_mask):
    #     trans_nm_0 = _centered_gaussian(*res_mask.shape, self._device)
    #     trans_0 = trans_nm_0 * du.NM_TO_ANG_SCALE
    #     if self._trans_cfg.batch_ot:
    #         trans_0 = self._batch_ot(trans_0, trans_1, diffuse_mask)
    #     if self._trans_cfg.train_schedule == 'linear':
    #         trans_t = (1 - t[..., None]) * trans_0 + t[..., None] * trans_1
    #     else:
    #         raise ValueError(
    #             f'Unknown trans schedule {self._trans_cfg.train_schedule}')
    #     trans_t = _trans_diffuse_mask(trans_t, trans_1, diffuse_mask)
    #     return trans_t * res_mask[..., None]
    
    # def _batch_ot(self, trans_0, trans_1, res_mask):
    #     num_batch, num_res = trans_0.shape[:2]
    #     noise_idx, gt_idx = torch.where(
    #         torch.ones(num_batch, num_batch))
    #     batch_nm_0 = trans_0[noise_idx]
    #     batch_nm_1 = trans_1[gt_idx]
    #     batch_mask = res_mask[gt_idx]
    #     aligned_nm_0, aligned_nm_1, _ = du.batch_align_structures(
    #         batch_nm_0, batch_nm_1, mask=batch_mask
    #     ) 
    #     aligned_nm_0 = aligned_nm_0.reshape(num_batch, num_batch, num_res, 3)
    #     aligned_nm_1 = aligned_nm_1.reshape(num_batch, num_batch, num_res, 3)
        
    #     # Compute cost matrix of aligned noise to ground truth
    #     batch_mask = batch_mask.reshape(num_batch, num_batch, num_res)
    #     cost_matrix = torch.sum(
    #         torch.linalg.norm(aligned_nm_0 - aligned_nm_1, dim=-1), dim=-1
    #     ) / torch.sum(batch_mask, dim=-1)
    #     noise_perm, gt_perm = linear_sum_assignment(du.to_numpy(cost_matrix))
    #     return aligned_nm_0[(tuple(gt_perm), tuple(noise_perm))]
    
    # def _corrupt_rotmats(self, rotmats_1, t, res_mask, diffuse_mask):
    #     num_batch, num_res = res_mask.shape
    #     noisy_rotmats = self.igso3.sample(
    #         torch.tensor([1.5]),
    #         num_batch*num_res
    #     ).to(self._device)
    #     noisy_rotmats = noisy_rotmats.reshape(num_batch, num_res, 3, 3)
    #     rotmats_0 = torch.einsum(
    #         "...ij,...jk->...ik", rotmats_1, noisy_rotmats)
        
    #     so3_schedule = self._rots_cfg.train_schedule
    #     if so3_schedule == 'exp':
    #         so3_t = 1 - torch.exp(-t*self._rots_cfg.exp_rate)
    #     elif so3_schedule == 'linear':
    #         so3_t = t
    #     else:
    #         raise ValueError(f'Invalid schedule: {so3_schedule}')
    #     rotmats_t = so3_utils.geodesic_t(so3_t[..., None], rotmats_1, rotmats_0)
    #     identity = torch.eye(3, device=self._device)
    #     rotmats_t = (
    #         rotmats_t * res_mask[..., None, None]
    #         + identity[None, None] * (1 - res_mask[..., None, None])
    #     )
    #     return _rots_diffuse_mask(rotmats_t, rotmats_1, diffuse_mask)

    def _corrupt_aatypes(self, aatypes_1, t, res_mask): #, diffuse_mask):
        num_batch, num_res = res_mask.shape
        assert aatypes_1.shape == (num_batch, num_res)
        assert t.shape == (num_batch, 1)
        assert res_mask.shape == (num_batch, num_res)
        # assert diffuse_mask.shape == (num_batch, num_res)

        if self._cfg.interpolant_type == "masking":
            u = torch.rand(num_batch, num_res, device=self._device)
            aatypes_t = aatypes_1.clone()
            corruption_mask = u < (1 - t) # (B, N) # t=1 is clean data

            aatypes_t[corruption_mask] = mu.MASK_TOKEN_INDEX

            aatypes_t = aatypes_t * res_mask + mu.MASK_TOKEN_INDEX * (1 - res_mask)

        # elif self._aatypes_cfg.interpolant_type == "uniform":
        #     u = torch.rand(num_batch, num_res, device=self._device)
        #     aatypes_t = aatypes_1.clone()
        #     corruption_mask = u < (1-t) # (B, N)
        #     uniform_sample = torch.randint_like(aatypes_t, low=0, high=du.NUM_TOKENS)
        #     aatypes_t[corruption_mask] = uniform_sample[corruption_mask]

        #     aatypes_t = aatypes_t * res_mask + mu.MASK_TOKEN_INDEX * (1 - res_mask)
        else:
            raise ValueError(f"Unknown aatypes interpolant type {self._cfg.interpolant_type}")

        return aatypes_t.long()
        # return _aatypes_diffuse_mask(aatypes_t, aatypes_1, diffuse_mask)

    def corrupt_batch(self, batch, t=None):
        noisy_batch = copy.deepcopy(batch)
        X, S, mask, chain_M, residue_idx, chain_encoding_all = noisy_batch
        noisy_batch = {}
        noisy_batch['X'] = X
        noisy_batch['S'] = S
        noisy_batch['mask'] = mask
        noisy_batch['chain_M'] = chain_M
        noisy_batch['residue_idx'] = residue_idx
        noisy_batch['chain_encoding_all'] = chain_encoding_all

        # [B, N, 3]
        # trans_1 = batch['trans_1']  # Angstrom

        # [B, N, 3, 3]
        # rotmats_1 = batch['rotmats_1']

        # [B, N]
        aatypes_1 = S # batch['aatypes_1']

        # [B, N]
        # res_mask = batch['res_mask']
        # diffuse_mask = batch['diffuse_mask']
        num_batch, num_res = aatypes_1.shape

        # [B, 1]
        # if self._cfg.codesign_separate_t:
        #     u = torch.rand((num_batch,), device=self._device)
        #     forward_fold_mask = (u < self._cfg.codesign_forward_fold_prop).float()
        #     inverse_fold_mask = (u < self._cfg.codesign_inverse_fold_prop + self._cfg.codesign_forward_fold_prop).float() * \
        #         (u >= self._cfg.codesign_forward_fold_prop).float()

        #     normal_structure_t = self.sample_t(num_batch)
        #     inverse_fold_structure_t = torch.ones((num_batch,), device=self._device)
        #     normal_cat_t = self.sample_t(num_batch)
        #     forward_fold_cat_t = torch.ones((num_batch,), device=self._device)

        #     # If we are forward folding, then cat_t should be 1
        #     # If we are inverse folding or codesign then cat_t should be uniform
        #     cat_t = forward_fold_mask * forward_fold_cat_t + (1 - forward_fold_mask) * normal_cat_t

        #     # If we are inverse folding, then structure_t should be 1
        #     # If we are forward folding or codesign then structure_t should be uniform
        #     structure_t = inverse_fold_mask * inverse_fold_structure_t + (1 - inverse_fold_mask) * normal_structure_t

        #     so3_t = structure_t[:, None]
        #     r3_t = structure_t[:, None]
        #     cat_t = cat_t[:, None]

        # else:
        if t is None:
            t = self.sample_t(num_batch)[:, None]
        else:
            t = torch.ones((num_batch, 1), device=self._device) * t
        # so3_t = t
        # r3_t = t
        # cat_t = t
        # noisy_batch['so3_t'] = so3_t
        # noisy_batch['r3_t'] = r3_t
        # noisy_batch['cat_t'] = cat_t
        noisy_batch['t'] = t

        # Apply corruptions
        # if self._trans_cfg.corrupt:
        #     trans_t = self._corrupt_trans(
        #         trans_1, r3_t, res_mask, diffuse_mask)
        # else:
        #     trans_t = trans_1
        # if torch.any(torch.isnan(trans_t)):
        #     raise ValueError('NaN in trans_t during corruption')
        # noisy_batch['trans_t'] = trans_t

        # if self._rots_cfg.corrupt:
        #     rotmats_t = self._corrupt_rotmats(rotmats_1, so3_t, res_mask, diffuse_mask)
        # else:
        #     rotmats_t = rotmats_1
        # if torch.any(torch.isnan(rotmats_t)):
        #     raise ValueError('NaN in rotmats_t during corruption')
        # noisy_batch['rotmats_t'] = rotmats_t

        res_mask = mask * chain_M
        # if self._aatypes_cfg.corrupt:
        aatypes_t = self._corrupt_aatypes(aatypes_1, t, res_mask)
        # print(aatypes_t)
        # else:
        #     aatypes_t = aatypes_1
        noisy_batch['S_t'] = aatypes_t
        # noisy_batch['trans_sc'] = torch.zeros_like(trans_1)
        # noisy_batch['aatypes_sc'] = torch.zeros_like(
        #     aatypes_1)[..., None].repeat(1, 1, self.num_tokens)
        return noisy_batch
    
    # def rot_sample_kappa(self, t):
    #     if self._rots_cfg.sample_schedule == 'exp':
    #         return 1 - torch.exp(-t*self._rots_cfg.exp_rate)
    #     elif self._rots_cfg.sample_schedule == 'linear':
    #         return t
    #     else:
    #         raise ValueError(
    #             f'Invalid schedule: {self._rots_cfg.sample_schedule}')

    # def _trans_vector_field(self, t, trans_1, trans_t):
    #     if self._trans_cfg.sample_schedule == 'linear':
    #         trans_vf = (trans_1 - trans_t) / (1 - t)
    #     elif self._trans_cfg.sample_schedule == 'vpsde':
    #         bmin = self._trans_cfg.vpsde_bmin
    #         bmax = self._trans_cfg.vpsde_bmax
    #         bt = bmin + (bmax - bmin) * (1-t) # scalar
    #         alpha_t = torch.exp(- bmin * (1-t) - 0.5 * (1-t)**2 * (bmax - bmin)) # scalar
    #         trans_vf = 0.5 * bt * trans_t + \
    #             0.5 * bt * (torch.sqrt(alpha_t) * trans_1 - trans_t) / (1 - alpha_t)
    #     else:
    #         raise ValueError(
    #             f'Invalid sample schedule: {self._trans_cfg.sample_schedule}'
    #         )
    #     return trans_vf

    # def _trans_euler_step(self, d_t, t, trans_1, trans_t):
    #     assert d_t >= 0
    #     trans_vf = self._trans_vector_field(t, trans_1, trans_t)
    #     return trans_t + trans_vf * d_t

    # def _rots_euler_step(self, d_t, t, rotmats_1, rotmats_t):
    #     if self._rots_cfg.sample_schedule == 'linear':
    #         scaling = 1 / (1 - t)
    #     elif self._rots_cfg.sample_schedule == 'exp':
    #         scaling = self._rots_cfg.exp_rate
    #     else:
    #         raise ValueError(
    #             f'Unknown sample schedule {self._rots_cfg.sample_schedule}')
    #     # TODO: Add in SDE.
    #     return so3_utils.geodesic_t(
    #         scaling * d_t, rotmats_1, rotmats_t)

    def _regularize_step_probs(self, step_probs, aatypes_t):
        batch_size, num_res, S = step_probs.shape
        device = step_probs.device
        assert aatypes_t.shape == (batch_size, num_res)

        step_probs = torch.clamp(step_probs, min=0.0, max=1.0)
        # TODO replace with torch._scatter
        step_probs[
            torch.arange(batch_size, device=device).repeat_interleave(num_res),
            torch.arange(num_res, device=device).repeat(batch_size),
            aatypes_t.long().flatten()
        ] = 0.0
        step_probs[
            torch.arange(batch_size, device=device).repeat_interleave(num_res),
            torch.arange(num_res, device=device).repeat(batch_size),
            aatypes_t.long().flatten()
        ] = 1.0 - torch.sum(step_probs, dim=-1).flatten()
        step_probs = torch.clamp(step_probs, min=0.0, max=1.0)
        return step_probs

    def _aatypes_euler_step(self, d_t, t, logits_1, aatypes_t):
        # S = 22
        batch_size, num_res, S = logits_1.shape
        assert aatypes_t.shape == (batch_size, num_res)
        if self._cfg.interpolant_type == "masking":
            assert S == 22
            device = logits_1.device
            
            mask_one_hot = torch.zeros((S,), device=device)
            mask_one_hot[mu.MASK_TOKEN_INDEX] = 1.0

            logits_1[:, :, mu.MASK_TOKEN_INDEX] = -1e9

            pt_x1_probs = F.softmax(logits_1 / self._cfg.temp, dim=-1) # (B, D, S)

            aatypes_t_is_mask = (aatypes_t == mu.MASK_TOKEN_INDEX).view(batch_size, num_res, 1).float()
            step_probs = d_t * pt_x1_probs * ((1+ self._cfg.noise*t) / ((1 - t))) # (B, D, S)
            step_probs += d_t * (1 - aatypes_t_is_mask) * mask_one_hot.view(1, 1, -1) * self._cfg.noise

            step_probs = self._regularize_step_probs(step_probs, aatypes_t)

            return torch.multinomial(step_probs.view(-1, S), num_samples=1).view(batch_size, num_res)
        else:
            raise ValueError(f"Unknown aatypes interpolant type {self._cfg.interpolant_type}")

    def _aatypes_euler_step_purity(self, d_t, t, logits_1, aatypes_t):
        # TODO: understand this
        batch_size, num_res, S = logits_1.shape
        assert aatypes_t.shape == (batch_size, num_res)
        assert S == 22
        assert self._cfg.interpolant_type == "masking"
        device = logits_1.device

        logits_1_wo_mask = logits_1[:, :, 0:-1] # (B, D, S-1)
        pt_x1_probs = F.softmax(logits_1_wo_mask / self._cfg.temp, dim=-1) # (B, D, S-1)
        # step_probs = (d_t * pt_x1_probs * (1/(1-t))).clamp(max=1) # (B, D, S-1)
        max_logprob = torch.max(torch.log(pt_x1_probs), dim=-1)[0] # (B, D)
        # bias so that only currently masked positions get chosen to be unmasked
        max_logprob = max_logprob - (aatypes_t != mu.MASK_TOKEN_INDEX).float() * 1e9
        sorted_max_logprobs_idcs = torch.argsort(max_logprob, dim=-1, descending=True) # (B, D)

        unmask_probs = (d_t * ( (1 + self._cfg.noise * t) / (1-t)).to(device)).clamp(max=1) # scalar

        number_to_unmask = torch.binomial(count=torch.count_nonzero(aatypes_t == mu.MASK_TOKEN_INDEX, dim=-1).float(),
                                          prob=unmask_probs)
        unmasked_samples = torch.multinomial(pt_x1_probs.view(-1, S-1), num_samples=1).view(batch_size, num_res)

        D_grid = torch.arange(num_res, device=device).view(1, -1).repeat(batch_size, 1)
        mask1 = (D_grid < number_to_unmask.view(-1, 1)).float()
        inital_val_max_logprob_idcs = sorted_max_logprobs_idcs[:, 0].view(-1, 1).repeat(1, num_res)
        masked_sorted_max_logprobs_idcs = (mask1 * sorted_max_logprobs_idcs + (1-mask1) * inital_val_max_logprob_idcs).long()
        mask2 = torch.zeros((batch_size, num_res), device=device)
        mask2.scatter_(dim=1, index=masked_sorted_max_logprobs_idcs, src=torch.ones((batch_size, num_res), device=device))
        unmask_zero_row = (number_to_unmask == 0).view(-1, 1).repeat(1, num_res).float()
        mask2 = mask2 * (1 - unmask_zero_row)
        aatypes_t = aatypes_t * (1 - mask2) + unmasked_samples * mask2

        # re-mask
        u = torch.rand(batch_size, num_res, device=self._device)
        re_mask_mask = (u < d_t * self._cfg.noise).float()
        aatypes_t = aatypes_t * (1 - re_mask_mask) + mu.MASK_TOKEN_INDEX * re_mask_mask

        return aatypes_t


    def sample(
            self,
            model,
            X, mask, chain_M, residue_idx, chain_encoding_all,
        ):

        num_batch, num_res = mask.shape
        aatypes_0 = _masked_categorical(num_batch, num_res, self._device).long()

        logs_traj = defaultdict(list)

        # Set-up time
        # if num_timesteps is None:
        num_timesteps = self._cfg.num_timesteps
        ts = torch.linspace(self._cfg.min_t, 1.0, num_timesteps)
        t_1 = ts[0]
        aatypes_t_1 = aatypes_0 # [bsz, seqlen]
        prot_traj = [aatypes_0.detach().cpu()] 
        clean_traj = []
        for t_2 in ts[1:]:
            d_t = t_2 - t_1
            with torch.no_grad():
                # model_out = model(batch)
               model_out = model(X, aatypes_t_1, mask, chain_M, residue_idx, chain_encoding_all)

            pred_logits_1 = model_out # [bsz, seqlen, 22]
            pred_logits_wo_mask = pred_logits_1.clone()
            pred_logits_wo_mask[:, :, mu.MASK_TOKEN_INDEX] = -1e9
            pred_aatypes_1 = torch.argmax(pred_logits_wo_mask, dim=-1)
            # pred_aatypes_1 = torch.argmax(pred_logits_1, dim=-1)
            clean_traj.append(pred_aatypes_1.detach().cpu())

            if self._cfg.do_purity:
                aatypes_t_2 = self._aatypes_euler_step_purity(d_t, t_1, pred_logits_1, aatypes_t_1)
            else:
                # aatypes_t_2 = self._aatypes_euler_step(d_t, t_1, pred_logits_1, aatypes_t_1)
                
                # change it to the sampling as in the gosai dataset
                pred_logits_1[:, :, mu.MASK_TOKEN_INDEX] = self.neg_infinity
                pred_logits_1 = pred_logits_1 / self._cfg.temp - torch.logsumexp(pred_logits_1 / self._cfg.temp, 
                                                                             dim=-1, keepdim=True)

                # For the logits of the unmasked tokens, set all values
                # to -infinity except for the indices corresponding to
                # the unmasked tokens.
                unmasked_indices = (aatypes_t_1 != mu.MASK_TOKEN_INDEX)
                # print(unmasked_indices.shape)
                pred_logits_1[unmasked_indices] = self.neg_infinity
                pred_logits_1[unmasked_indices, aatypes_t_1[unmasked_indices]] = 0
                
                # pt_x1_probs = F.softmax(pred_logits_1 / self._cfg.temp, dim=-1) # (B, D, S)
                move_chance_t = 1.0 - t_1
                move_chance_s = 1.0 - t_2
                q_xs = pred_logits_1.exp() * d_t
                # print(q_xs.shape, move_chance_s.shape)
                q_xs[:, :, mu.MASK_TOKEN_INDEX] = move_chance_s #[:, :, 0]
                # _x = torch.multinomial(q_xs.view(-1, q_xs.shape[-1]), num_samples=1).view(num_batch, num_res)
                _x = _sample_categorical(q_xs)
                copy_flag = (aatypes_t_1 != mu.MASK_TOKEN_INDEX).to(aatypes_t_1.dtype)
                aatypes_t_2 = aatypes_t_1 * copy_flag + _x * (1 - copy_flag)


            aatypes_t_1 = aatypes_t_2.long()
            prot_traj.append(aatypes_t_2.cpu().detach())

            t_1 = t_2

        # We only integrated to min_t, so need to make a final step
        t_1 = ts[-1]

        # with torch.no_grad():
        #     # model_out = model(batch)
        #     model_out = model(X, aatypes_t_1, mask, chain_M, residue_idx, chain_encoding_all)
        # pred_logits_1 = model_out
        # pred_logits_wo_mask = pred_logits_1.clone()
        # pred_logits_wo_mask[:, :, mu.MASK_TOKEN_INDEX] = -1e9
        # pred_aatypes_1 = torch.argmax(pred_logits_wo_mask, dim=-1)
        # # pred_aatypes_1 = torch.argmax(pred_logits_1, dim=-1)

        # clean_traj.append(pred_aatypes_1.detach().cpu())
        # prot_traj.append(pred_aatypes_1.detach().cpu())
        return pred_aatypes_1, prot_traj, clean_traj


    def sample_gradient(
            self,
            model,
            X, mask, chain_M, residue_idx, chain_encoding_all,
            truncate_steps, gumbel_softmax_temp
        ):
        assert self._cfg.do_purity == False
        num_batch, num_res = mask.shape
        aatypes_0 = _masked_categorical(num_batch, num_res, self._device).long()
        aatypes_0 = F.one_hot(aatypes_0, num_classes=self.num_tokens).float()

        logs_traj = defaultdict(list)

        # Set-up time
        # if num_timesteps is None:
        num_timesteps = self._cfg.num_timesteps
        ts = torch.linspace(self._cfg.min_t, 1.0, num_timesteps)
        t_1 = ts[0]
        aatypes_t_1 = aatypes_0
        prot_traj = [aatypes_0.detach().cpu()] 
        clean_traj = []

        last_x_list = []
        move_chance_t_list = []
        copy_flag_list = []

        for _ts, t_2 in enumerate(ts[1:]):
            d_t = t_2 - t_1
            model_out = model(X, aatypes_t_1, mask, chain_M, residue_idx, chain_encoding_all)
            pred_logits_1 = model_out.clone()
            pred_logits_wo_mask = pred_logits_1.clone()
            pred_logits_wo_mask[:, :, mu.MASK_TOKEN_INDEX] = -1e9
            pred_aatypes_1 = torch.argmax(pred_logits_wo_mask, dim=-1)
            # pred_aatypes_1 = torch.argmax(pred_logits_1, dim=-1)
            clean_traj.append(pred_aatypes_1.detach().cpu())

            pred_logits_1[:, :, mu.MASK_TOKEN_INDEX] = self.neg_infinity
            pred_logits_1 = pred_logits_1 / self._cfg.temp - torch.logsumexp(pred_logits_1 / self._cfg.temp, 
                                                                             dim=-1, keepdim=True)

            # For the logits of the unmasked tokens, set all values
            # to -infinity except for the indices corresponding to
            # the unmasked tokens.
            if aatypes_t_1.ndim > 2 and aatypes_t_1.shape[-1] == self.num_tokens:
                aatypes_t_1_argmax = aatypes_t_1.argmax(dim=-1)
            else:
                aatypes_t_1_argmax = aatypes_t_1
            unmasked_indices = (aatypes_t_1_argmax != mu.MASK_TOKEN_INDEX)
            # print(unmasked_indices.shape)
            pred_logits_1[unmasked_indices] = self.neg_infinity
            pred_logits_1[unmasked_indices, aatypes_t_1_argmax[unmasked_indices]] = 0
            # pt_x1_probs = F.softmax(pred_logits_1 / self._cfg.temp, dim=-1) # (B, D, S)
            move_chance_t = 1.0 - t_1
            move_chance_s = 1.0 - t_2
            q_xs = pred_logits_1.exp() * d_t / move_chance_t
            q_xs[:, :, mu.MASK_TOKEN_INDEX] = move_chance_s / move_chance_t #[:, :, 0]
            if _ts < num_timesteps - truncate_steps:
                _x = _sample_categorical(q_xs)
                _x = F.one_hot(_x, num_classes=self.num_tokens).float()
                copy_flag = (aatypes_t_1.argmax(dim=-1) != mu.MASK_TOKEN_INDEX).to(aatypes_t_1.dtype).unsqueeze(-1)
                aatypes_t_2 = aatypes_t_1 * copy_flag + _x * (1 - copy_flag)

                aatypes_t_2 = aatypes_t_2.detach()
                aatypes_t_1 = aatypes_t_1.detach()

            else:
                q_xs = q_xs + 1e-9
                _x = _sample_categorical_gradient(q_xs, gumbel_softmax_temp)
                copy_flag = 1 - aatypes_t_1[:, :, mu.MASK_TOKEN_INDEX].unsqueeze(-1)
                aatypes_t_2 = aatypes_t_1 * copy_flag + _x * (1 - copy_flag)

            last_x_list.append(aatypes_t_1)
            move_chance_t_list.append(move_chance_t + self._cfg.min_t)
            copy_flag_list.append(copy_flag)

            aatypes_t_1 = aatypes_t_2
            prot_traj.append(aatypes_t_2.cpu().detach())

            t_1 = t_2

        last_x_list.append(aatypes_t_1)
        move_chance_t_list.append(1.0 - t_1 + self._cfg.min_t)
        copy_flag_list.append(1 - aatypes_t_1[:, :, mu.MASK_TOKEN_INDEX].unsqueeze(-1))

        aatypes_t_1_argmax = aatypes_t_1[:, :, :-1].argmax(dim=-1) # to avoid the mask token
        aatypes_t_1_argmax = F.one_hot(aatypes_t_1_argmax, num_classes=self.num_tokens).float()
        return aatypes_t_1 + (aatypes_t_1_argmax - aatypes_t_1).detach(), last_x_list, move_chance_t_list, copy_flag_list

        # We only integrated to min_t, so need to make a final step
        t_1 = ts[-1]

        model_out = model(X, aatypes_t_1, mask, chain_M, residue_idx, chain_encoding_all)
        pred_logits_1 = model_out
        pred_logits_wo_mask = pred_logits_1.clone()
        pred_logits_wo_mask[:, :, mu.MASK_TOKEN_INDEX] = -1e9
        pred_aatypes_1 = torch.argmax(pred_logits_wo_mask, dim=-1)
        pred_aatypes_1 = F.one_hot(pred_aatypes_1, num_classes=self.num_tokens).float()

        last_x_list.append(aatypes_t_1)
        move_chance_t_list.append(1.0 - t_1)
        copy_flag_list.append(1 - aatypes_t_1[:, :, mu.MASK_TOKEN_INDEX].unsqueeze(-1))
        # pred_aatypes_1 = torch.argmax(pred_logits_1, dim=-1)

        clean_traj.append(pred_aatypes_1.detach().cpu())
        prot_traj.append(pred_aatypes_1.detach().cpu())
        # return pred_aatypes_1, prot_traj, clean_traj
        return pred_logits_1 + (pred_aatypes_1 - pred_logits_1).detach(), last_x_list, move_chance_t_list, copy_flag_list

    def sample_controlled_DPS(self,
                              model,
                              X, mask, chain_M, residue_idx, chain_encoding_all,
                              guidance_scale, reward_model):
        num_batch, num_res = mask.shape
        aatypes_0 = _masked_categorical(num_batch, num_res, self._device).long()
        logs_traj = defaultdict(list)

        # Set-up time
        # if num_timesteps is None:
        num_timesteps = self._cfg.num_timesteps
        ts = torch.linspace(self._cfg.min_t, 1.0, num_timesteps)
        t_1 = ts[0]
        aatypes_t_1 = aatypes_0 # [bsz, seqlen]
        prot_traj = [aatypes_0.detach().cpu()] 
        clean_traj = []
        for t_2 in ts[1:]:
            d_t = t_2 - t_1
            with torch.no_grad():
                # model_out = model(batch)
               model_out = model(X, aatypes_t_1, mask, chain_M, residue_idx, chain_encoding_all)

            pred_logits_1 = model_out # [bsz, seqlen, 22]
            pred_logits_wo_mask = pred_logits_1.clone()
            pred_logits_wo_mask[:, :, mu.MASK_TOKEN_INDEX] = -1e9
            pred_aatypes_1 = torch.argmax(pred_logits_wo_mask, dim=-1)
            # pred_aatypes_1 = torch.argmax(pred_logits_1, dim=-1)
            clean_traj.append(pred_aatypes_1.detach().cpu())

            if self._cfg.do_purity:
                raise NotImplementedError
                aatypes_t_2 = self._aatypes_euler_step_purity(d_t, t_1, pred_logits_1, aatypes_t_1)
            else:
                
                # change it to the sampling as in the gosai dataset
                pred_logits_1[:, :, mu.MASK_TOKEN_INDEX] = self.neg_infinity
                pred_logits_1 = pred_logits_1 / self._cfg.temp - torch.logsumexp(pred_logits_1 / self._cfg.temp, 
                                                                             dim=-1, keepdim=True)
                # For the logits of the unmasked tokens, set all values
                # to -infinity except for the indices corresponding to
                # the unmasked tokens.
                unmasked_indices = (aatypes_t_1 != mu.MASK_TOKEN_INDEX)
                pred_logits_1[unmasked_indices] = self.neg_infinity
                pred_logits_1[unmasked_indices, aatypes_t_1[unmasked_indices]] = 0
                
                # pt_x1_probs = F.softmax(pred_logits_1 / self._cfg.temp, dim=-1) # (B, D, S)
                move_chance_t = 1.0 - t_1
                move_chance_s = 1.0 - t_2
                q_xs = pred_logits_1.exp() * d_t

                x_onehot = F.one_hot(aatypes_t_1, num_classes=self.num_tokens).float()
                copy_flag = (aatypes_t_1 != mu.MASK_TOKEN_INDEX).to(aatypes_t_1.dtype)

                x_grad = self.compute_gradient_DPS(model, aatypes_t_1, x_onehot, copy_flag, reward_model, X, mask, chain_M, residue_idx, chain_encoding_all)
                guidance = guidance_scale * (x_grad - x_grad[:, :, mu.MASK_TOKEN_INDEX].unsqueeze(-1))

                # print(q_xs.shape, move_chance_s.shape)
                q_xs[:, :, mu.MASK_TOKEN_INDEX] = move_chance_s #[:, :, 0]
                # _x = torch.multinomial(q_xs.view(-1, q_xs.shape[-1]), num_samples=1).view(num_batch, num_res)
                q_xs = q_xs * guidance.exp()
                
                _x = _sample_categorical(q_xs)
                copy_flag = (aatypes_t_1 != mu.MASK_TOKEN_INDEX).to(aatypes_t_1.dtype)
                aatypes_t_2 = aatypes_t_1 * copy_flag + _x * (1 - copy_flag)


            aatypes_t_1 = aatypes_t_2.long()
            prot_traj.append(aatypes_t_2.cpu().detach())

            t_1 = t_2

        t_1 = ts[-1]
        return pred_aatypes_1, prot_traj, clean_traj


    def compute_gradient_DPS(self, model, aatypes_t_1, x_onehot,copy_flag, reward_model,
                             X, mask, chain_M, residue_idx, chain_encoding_all):
        x_onehot.requires_grad_(True)

        expected_x0 = model(X, x_onehot, mask, chain_M, residue_idx, chain_encoding_all) # Calcualte E[x_0|x_t]
        #improve_x0 = copy_flag[:, :, None] * F.one_hot(aatypes_t_1, num_classes= 22) + (1.0 - copy_flag[:, :, None]) * torch.nn.functional.softmax(expected_x0, dim = 2) 
        scores = reward_model(X, torch.nn.functional.softmax(expected_x0, dim = 2)   , mask, chain_M, residue_idx, chain_encoding_all)
        
        # expected_x0 = self.forward2(x_onehot, x, sigma_s) 
        scores = scores.mean()
        scores.backward()
        x_grad = x_onehot.grad.clone()
        return x_grad

    def sample_controlled_SVDD(
            self,
            model,
            X, mask, chain_M, residue_idx, chain_encoding_all,
            reward_model, reward_name, repeats = 3
        ):

        num_batch, num_res = mask.shape
        aatypes_0 = _masked_categorical(num_batch, num_res, self._device).long()

        logs_traj = defaultdict(list)

        # Set-up time
        # if num_timesteps is None:
        num_timesteps = self._cfg.num_timesteps
        ts = torch.linspace(self._cfg.min_t, 1.0, num_timesteps)
        t_1 = ts[0]
        aatypes_t_1 = aatypes_0 # [bsz, seqlen]
        prot_traj = [aatypes_0.detach().cpu()] 
        clean_traj = []
        for t_2 in tqdm(ts[1:]):
            d_t = t_2 - t_1
            with torch.no_grad():
                # model_out = model(batch)
               model_out = model(X, aatypes_t_1, mask, chain_M, residue_idx, chain_encoding_all)

            pred_logits_1 = model_out # [bsz, seqlen, 22]
            pred_logits_wo_mask = pred_logits_1.clone()
            pred_logits_wo_mask[:, :, mu.MASK_TOKEN_INDEX] = -1e9
            pred_aatypes_1 = torch.argmax(pred_logits_wo_mask, dim=-1)
            # pred_aatypes_1 = torch.argmax(pred_logits_1, dim=-1)
            clean_traj.append(pred_aatypes_1.detach().cpu())

            if self._cfg.do_purity:
                aatypes_t_2 = self._aatypes_euler_step_purity(d_t, t_1, pred_logits_1, aatypes_t_1)
            else:
                # aatypes_t_2 = self._aatypes_euler_step(d_t, t_1, pred_logits_1, aatypes_t_1)
                
                # change it to the sampling as in the gosai dataset
                pred_logits_1[:, :, mu.MASK_TOKEN_INDEX] = self.neg_infinity
                pred_logits_1 = pred_logits_1 / self._cfg.temp - torch.logsumexp(pred_logits_1 / self._cfg.temp, 
                                                                             dim=-1, keepdim=True)

                # For the logits of the unmasked tokens, set all values
                # to -infinity except for the indices corresponding to
                # the unmasked tokens.
                unmasked_indices = (aatypes_t_1 != mu.MASK_TOKEN_INDEX)
                pred_logits_1[unmasked_indices] = self.neg_infinity
                pred_logits_1[unmasked_indices, aatypes_t_1[unmasked_indices]] = 0
                
                move_chance_t = 1.0 - t_1
                move_chance_s = 1.0 - t_2
                q_xs = pred_logits_1.exp() * d_t

                
                q_xs[:, :, mu.MASK_TOKEN_INDEX] = move_chance_s #[:, :, 0]
                # _x = torch.multinomial(q_xs.view(-1, q_xs.shape[-1]), num_samples=1).view(num_batch, num_res)
          
                copy_flag = (aatypes_t_1 != mu.MASK_TOKEN_INDEX).to(aatypes_t_1.dtype)
                categorical_list = [ _sample_categorical(q_xs) for iii in range(repeats) ]
                aatypes_t_2_list = [ aatypes_t_1 * copy_flag + categorical_list[iii] * (1 - copy_flag) for iii in range(repeats) ] 

                scores = []
                improve_hot_x0_list = [] 
                for i in range(repeats): 
                    copy_flag_pes = (aatypes_t_2_list[i] != mu.MASK_TOKEN_INDEX).to(aatypes_t_1.dtype)
                    expected_x0_pes = model(X, aatypes_t_2_list[i], mask, chain_M, residue_idx, chain_encoding_all) # Calcualte E[x_0|x_{t-1}]
                    one_hot_x0 = torch.argmax(expected_x0_pes, dim = 2)
                    improve_hot_x0 = copy_flag_pes * aatypes_t_2_list[i] + (1 - copy_flag_pes) *  one_hot_x0
                    improve_hot_x0_list.append(improve_hot_x0)

                improve_hot_x0 = torch.cat(improve_hot_x0_list) 
                
                if reward_name == 'stability':
                    reward_list = []
                    for seq in improve_hot_x0_list:
                        reward_list.append(reward_model(X, 1.0 * F.one_hot(seq, num_classes= 22) , mask, chain_M, residue_idx, chain_encoding_all))
                    reward = torch.cat(reward_list) 
                elif reward_name == "LDDT": 
                    reward = reward_model.cal_reward(improve_hot_x0)
                elif reward_name == 'scRMSD':
                    reward = reward_model.cal_rmsd_reward(improve_hot_x0)
                    reward  = np.array(reward)
                    reward = -torch.from_numpy(reward).to(self._device)
                elif reward_name == 'stability_rosetta':
                    reward = reward_model.calculate_energy(improve_hot_x0)
                    reward  = np.array(reward)
                    reward = -torch.from_numpy(reward).to(self._device)

                scores = torch.reshape(reward, (repeats, int(len(reward)/repeats)))
                final_sample_indices = torch.argmax(scores, dim=0).squeeze()  # Indices, Shape [batch_size]
                final_samples = [aatypes_t_2_list[final_sample_indices[j]][j,:] for j in range(aatypes_t_1.size(0))]  # Select the chosen samples using gathered indices
                aatypes_t_2 = torch.stack(final_samples, dim=0)                            

            aatypes_t_1 = aatypes_t_2.long()
            prot_traj.append(aatypes_t_2.cpu().detach())

            t_1 = t_2

        # We only integrated to min_t, so need to make a final step
        t_1 = ts[-1]
        print(torch.mean(scores))
        return pred_aatypes_1, prot_traj, clean_traj
    
    def sample_controlled_NestedIS(
            self,
            model,
            X, mask, chain_M, residue_idx, chain_encoding_all,
            reward_model, reward_name, repeats = 20 
        ):

        num_batch, num_res = mask.shape
        aatypes_0 = _masked_categorical(num_batch, num_res, self._device).long()

        logs_traj = defaultdict(list)

        # Set-up time
        # if num_timesteps is None:
        num_timesteps = self._cfg.num_timesteps
        ts = torch.linspace(self._cfg.min_t, 1.0, num_timesteps)
        t_1 = ts[0]
        aatypes_t_1 = aatypes_0 # [bsz, seqlen]
        prot_traj = [aatypes_0.detach().cpu()] 
        clean_traj = []
        for t_2 in tqdm(ts[1:]):
            d_t = t_2 - t_1
            with torch.no_grad():
                # model_out = model(batch)
               model_out = model(X, aatypes_t_1, mask, chain_M, residue_idx, chain_encoding_all)

            pred_logits_1 = model_out # [bsz, seqlen, 22]
            pred_logits_wo_mask = pred_logits_1.clone()
            pred_logits_wo_mask[:, :, mu.MASK_TOKEN_INDEX] = -1e9
            pred_aatypes_1 = torch.argmax(pred_logits_wo_mask, dim=-1)
            # pred_aatypes_1 = torch.argmax(pred_logits_1, dim=-1)
            clean_traj.append(pred_aatypes_1.detach().cpu())

            if self._cfg.do_purity:
                aatypes_t_2 = self._aatypes_euler_step_purity(d_t, t_1, pred_logits_1, aatypes_t_1)
            else:
                # aatypes_t_2 = self._aatypes_euler_step(d_t, t_1, pred_logits_1, aatypes_t_1)
                
                # change it to the sampling as in the gosai dataset
                pred_logits_1[:, :, mu.MASK_TOKEN_INDEX] = self.neg_infinity
                pred_logits_1 = pred_logits_1 / self._cfg.temp - torch.logsumexp(pred_logits_1 / self._cfg.temp, 
                                                                             dim=-1, keepdim=True)

                # For the logits of the unmasked tokens, set all values
                # to -infinity except for the indices corresponding to
                # the unmasked tokens.
                unmasked_indices = (aatypes_t_1 != mu.MASK_TOKEN_INDEX)
                pred_logits_1[unmasked_indices] = self.neg_infinity
                pred_logits_1[unmasked_indices, aatypes_t_1[unmasked_indices]] = 0
                
                move_chance_t = 1.0 - t_1
                move_chance_s = 1.0 - t_2
                q_xs = pred_logits_1.exp() * d_t

                
                q_xs[:, :, mu.MASK_TOKEN_INDEX] = move_chance_s #[:, :, 0]
                # _x = torch.multinomial(q_xs.view(-1, q_xs.shape[-1]), num_samples=1).view(num_batch, num_res)
          
                copy_flag = (aatypes_t_1 != mu.MASK_TOKEN_INDEX).to(aatypes_t_1.dtype)
                categorical_list = [ _sample_categorical(q_xs) for iii in range(repeats) ]
                aatypes_t_2_list = [ aatypes_t_1 * copy_flag + categorical_list[iii] * (1 - copy_flag) for iii in range(repeats) ] 

                scores = []
                for i in range(repeats): 
                    copy_flag_pes = (aatypes_t_2_list[i] != mu.MASK_TOKEN_INDEX).to(aatypes_t_1.dtype)
                    expected_x0_pes = model(X, aatypes_t_2_list[i], mask, chain_M, residue_idx, chain_encoding_all) # Calcualte E[x_0|x_{t-1}]
                    one_hot_x0 = torch.argmax(expected_x0_pes, dim = 2)
                    improve_hot_x0 = copy_flag_pes * aatypes_t_2_list[i] + (1 - copy_flag_pes) *  one_hot_x0
                    if reward_name == 'stability':
                        reward = reward_model(X, 1.0 * F.one_hot(improve_hot_x0, num_classes= 22) , mask, chain_M, residue_idx, chain_encoding_all)
                    elif reward_name == 'scRMSD':
                        reward = reward_model.cal_rmsd_reward(improve_hot_x0)
                        reward  = np.array(reward)
                        reward = -torch.from_numpy(reward).to(self._device)
                    elif reward_name == "LDDT": 
                        reward = reward_model.cal_reward(improve_hot_x0)
                    elif reward_name == 'stability_rosetta':
                        reward = reward_model.calculate_energy(improve_hot_x0)
                        reward  = np.array(reward)
                        reward = -torch.from_numpy(reward).to(self._device)
                    scores.append(reward.squeeze())

                scores = torch.stack(scores, dim=1)
                final_sample_indices = torch.argmax(scores, dim=1).squeeze()  # Indices, Shape [batch_size]
              
                final_samples = [aatypes_t_2_list[final_sample_indices[j]][j,:] for j in range(aatypes_t_1.size(0))]  # Select the chosen samples using gathered indices
                aatypes_t_2 = torch.stack(final_samples, dim=0)  
                
                ### Global resampling
                global_weight = torch.max(scores, dim=1).values  # Indices, Shape [batch_size]
                global_weighthoge = torch.exp(global_weight/torch.max( torch.abs(global_weight )))
                global_weighthoge = global_weighthoge.cpu().detach().numpy()
                final_sample_indices = np.random.choice(aatypes_t_1.shape[0], aatypes_t_1.shape[0], p =  global_weighthoge/global_weighthoge.sum() )                  
                aatypes_t_2 = aatypes_t_2[final_sample_indices]

               
            aatypes_t_1 = aatypes_t_2.long()
            prot_traj.append(aatypes_t_2.cpu().detach())

            t_1 = t_2

        # We only integrated to min_t, so need to make a final step
        t_1 = ts[-1]
        print(torch.mean(scores))
        return pred_aatypes_1, prot_traj, clean_traj
    '''
    def sample_controlled_TDS5(
            self,
            model,
            X, mask, chain_M, residue_idx, chain_encoding_all,
            reward_model, alpha, repeats = 20
        ):

        num_batch, num_res = mask.shape
        aatypes_0 = _masked_categorical(num_batch, num_res, self._device).long()

        logs_traj = defaultdict(list)

        # Set-up time
        # if num_timesteps is None:
        num_timesteps = self._cfg.num_timesteps
        ts = torch.linspace(self._cfg.min_t, 1.0, num_timesteps)
        t_1 = ts[0]
        aatypes_t_1 = aatypes_0 # [bsz, seqlen]
        prot_traj = [] 
        clean_traj = []
        aa_type_t_1_list = [ ]
        for t_2 in ts[1:]:
            d_t = t_2 - t_1
 
            with torch.no_grad():
                # model_out = model(batch)
               model_out = model(X, aatypes_t_1, mask, chain_M, residue_idx, chain_encoding_all)
            
            pred_logits_1 = model_out # [bsz, seqlen, 22]
            pred_logits_wo_mask = pred_logits_1.clone()
            pred_logits_wo_mask[:, :, mu.MASK_TOKEN_INDEX] = -1e9
            pred_aatypes_1 = torch.argmax(pred_logits_wo_mask, dim=-1)
            # pred_aatypes_1 = torch.argmax(pred_logits_1, dim=-1)
            clean_traj.append(aatypes_t_1.detach().cpu())

            if self._cfg.do_purity:
                aatypes_t_2 = self._aatypes_euler_step_purity(d_t, t_1, pred_logits_1, aatypes_t_1)
            else:
                # aatypes_t_2 = self._aatypes_euler_step(d_t, t_1, pred_logits_1, aatypes_t_1)
                
                # change it to the sampling as in the gosai dataset
                pred_logits_1[:, :, mu.MASK_TOKEN_INDEX] = self.neg_infinity
                pred_logits_1 = pred_logits_1 / self._cfg.temp - torch.logsumexp(pred_logits_1 / self._cfg.temp, 
                                                                             dim=-1, keepdim=True)

                # For the logits of the unmasked tokens, set all values
                # to -infinity except for the indices corresponding to
                # the unmasked tokens.
                unmasked_indices = (aatypes_t_1 != mu.MASK_TOKEN_INDEX)
                pred_logits_1[unmasked_indices] = self.neg_infinity
                pred_logits_1[unmasked_indices, aatypes_t_1[unmasked_indices]] = 0
                
                move_chance_t = 1.0 - t_1
                move_chance_s = 1.0 - t_2
                q_xs = pred_logits_1.exp() * d_t

                
                q_xs[:, :, mu.MASK_TOKEN_INDEX] = move_chance_s #[:, :, 0]
                # _x = torch.multinomial(q_xs.view(-1, q_xs.shape[-1]), num_samples=1).view(num_batch, num_res)
          
                copy_flag = (aatypes_t_1 != mu.MASK_TOKEN_INDEX).to(aatypes_t_1.dtype)
                categorical_list = [ _sample_categorical(q_xs) for iii in range(repeats) ]
                aatypes_t_2_list = [ aatypes_t_1 * copy_flag + categorical_list[iii] * (1 - copy_flag) for iii in range(repeats) ] 

                scores = []
                for i in range(repeats):
                    copy_flag_pes = (aatypes_t_2_list[i] != mu.MASK_TOKEN_INDEX).to(aatypes_t_1.dtype)
                    expected_x0_pes = model(X, aatypes_t_2_list[i], mask, chain_M, residue_idx, chain_encoding_all) # Calcualte E[x_0|x_{t-1}]
                    one_hot_x0 = torch.argmax(expected_x0_pes, dim = 2)
                    improve_hot_x0 = copy_flag_pes * aatypes_t_2_list[i] + (1 - copy_flag_pes) *  one_hot_x0
                    reward = reward_model(X, 1.0 * F.one_hot(improve_hot_x0, num_classes= 22) , mask, chain_M, residue_idx, chain_encoding_all)
                    scores.append(reward.squeeze())

                scores = torch.stack(scores, dim=1)
                final_sample_indices = torch.argmax(scores, dim=1).squeeze()  # Shape [batch_size]
                final_samples = [aatypes_t_2_list[final_sample_indices[j]][j,:] for j in range(aatypes_t_1.size(0))]  # Select the chosen samples using gathered indices
                aatypes_t_2 = torch.stack(final_samples, dim=0) 

                                        
            aatypes_t_1 = aatypes_t_2.long()
            prot_traj.append(aatypes_t_1.cpu().detach())

            t_1 = t_2

        # We only integrated to min_t, so need to make a final step
        t_1 = ts[-1]
        print(torch.mean(scores))
        return pred_aatypes_1, prot_traj, clean_traj
    '''
    

    def sample_controlled_SMC(
            self,
            model,
            X, mask, chain_M, residue_idx, chain_encoding_all,
            reward_model, reward_name, alpha = 0.5
        ):

        num_batch, num_res = mask.shape
        aatypes_0 = _masked_categorical(num_batch, num_res, self._device).long()

        logs_traj = defaultdict(list)

        # Set-up time
        # if num_timesteps is None:
        num_timesteps = self._cfg.num_timesteps
        ts = torch.linspace(self._cfg.min_t, 1.0, num_timesteps)
        t_1 = ts[0]
        aatypes_t_1 = aatypes_0 # [bsz, seqlen]
        prot_traj = [aatypes_0.detach().cpu()] 
        clean_traj = []
        for t_2 in ts[1:]:
            d_t = t_2 - t_1
            with torch.no_grad():
                # model_out = model(batch)
               model_out = model(X, aatypes_t_1, mask, chain_M, residue_idx, chain_encoding_all)

            pred_logits_1 = model_out # [bsz, seqlen, 22]
            pred_logits_wo_mask = pred_logits_1.clone()
            pred_logits_wo_mask[:, :, mu.MASK_TOKEN_INDEX] = -1e9
            pred_aatypes_1 = torch.argmax(pred_logits_wo_mask, dim=-1)
            # pred_aatypes_1 = torch.argmax(pred_logits_1, dim=-1)
            clean_traj.append(pred_aatypes_1.detach().cpu())

            if self._cfg.do_purity:
                aatypes_t_2 = self._aatypes_euler_step_purity(d_t, t_1, pred_logits_1, aatypes_t_1)
            else:
                # aatypes_t_2 = self._aatypes_euler_step(d_t, t_1, pred_logits_1, aatypes_t_1)
                
                # change it to the sampling as in the gosai dataset
                pred_logits_1[:, :, mu.MASK_TOKEN_INDEX] = self.neg_infinity
                pred_logits_1 = pred_logits_1 / self._cfg.temp - torch.logsumexp(pred_logits_1 / self._cfg.temp, 
                                                                             dim=-1, keepdim=True)

                # For the logits of the unmasked tokens, set all values
                # to -infinity except for the indices corresponding to
                # the unmasked tokens.
                unmasked_indices = (aatypes_t_1 != mu.MASK_TOKEN_INDEX)
                pred_logits_1[unmasked_indices] = self.neg_infinity
                pred_logits_1[unmasked_indices, aatypes_t_1[unmasked_indices]] = 0
                
                move_chance_t = 1.0 - t_1
                move_chance_s = 1.0 - t_2
                q_xs = pred_logits_1.exp() * d_t

                
                q_xs[:, :, mu.MASK_TOKEN_INDEX] = move_chance_s #[:, :, 0]
                # _x = torch.multinomial(q_xs.view(-1, q_xs.shape[-1]), num_samples=1).view(num_batch, num_res)
                _x = _sample_categorical(q_xs)
                copy_flag = (aatypes_t_1 != mu.MASK_TOKEN_INDEX).to(aatypes_t_1.dtype)
                aatypes_t_2 = aatypes_t_1 * copy_flag + _x * (1 - copy_flag)


                '''
                Calcualte exp(v_{t-1}(x_{t-1})/alpha)
                '''
                expected_x0_pes = model(X, aatypes_t_2, mask, chain_M, residue_idx, chain_encoding_all) # Calcualte E[x_0|x_{t-1}]
                copy_flag = (aatypes_t_1 != mu.MASK_TOKEN_INDEX).to(aatypes_t_2.dtype)
                one_hot_x0 = torch.argmax(expected_x0_pes, dim = 2)
                improve_hot_x0 = copy_flag * aatypes_t_2 + (1 - copy_flag) *  one_hot_x0
                reward_num = reward_model(X, 1.0 * F.one_hot(improve_hot_x0, num_classes= 22) , mask, chain_M, residue_idx, chain_encoding_all)

                if reward_name == 'stability':
                        reward_num = reward_model(X, 1.0 * F.one_hot(improve_hot_x0, num_classes= 22) , mask, chain_M, residue_idx, chain_encoding_all)
                elif reward_name == 'scRMSD':
                    reward_num = reward_model.cal_rmsd_reward(improve_hot_x0)
                    reward_num  = np.array(reward_num)
                    reward_num = -torch.from_numpy(reward_num).to(self._device)
                elif reward_name == 'stability_rosetta':
                    reward_num = reward_model.calculate_energy(improve_hot_x0)
                    reward_num  = np.array(reward_num)
                    reward_num = -torch.from_numpy(reward_num).to(self._device)

          
                '''
                Calcualte exp(v_{t}(x_{t})/alpha)
                '''
                expected_x0_pes = model(X, aatypes_t_1, mask, chain_M, residue_idx, chain_encoding_all) # Calcualte E[x_0|x_t]
                copy_flag = (aatypes_t_1 != mu.MASK_TOKEN_INDEX).to(aatypes_t_1.dtype)
                one_hot_x0 = torch.argmax(expected_x0_pes, dim = 2)
                improve_hot_x0 = copy_flag * aatypes_t_1 + (1 - copy_flag) *  one_hot_x0
                reward_den = reward_model(X, 1.0 * F.one_hot(improve_hot_x0, num_classes= 22) , mask, chain_M, residue_idx, chain_encoding_all)

                if reward_name == 'stability':
                        reward_den = reward_model(X, 1.0 * F.one_hot(improve_hot_x0, num_classes= 22) , mask, chain_M, residue_idx, chain_encoding_all)
                elif reward_name == 'scRMSD':
                    reward_den = reward_model.cal_rmsd_reward(improve_hot_x0)
                    reward_den  = np.array(reward_den)
                    reward_den = -torch.from_numpy(reward_den).to(self._device)
                elif reward_name == 'stability_rosetta':
                    reward_den = reward_model.calculate_energy(improve_hot_x0)
                    reward_den  = np.array(reward_den)
                    reward_den = -torch.from_numpy(reward_den).to(self._device)


                '''
                Calculate ratio and do sampling
                '''
                ratio = torch.exp(1.0/alpha * (reward_num - reward_den)) # Now calculate exp( (v_{t-1}(x_{t-1) -v_{t}(x_{t}) /alpha) 
                ratio = ratio.detach().cpu().numpy()
                final_sample_indices = np.random.choice(reward_num.shape[0], reward_num.shape[0], p =  ratio/ratio.sum() )                  
                aatypes_t_2 = aatypes_t_2[final_sample_indices]

            aatypes_t_1 = aatypes_t_2.long()
            prot_traj.append(aatypes_t_2.cpu().detach())

            t_1 = t_2

        # We only integrated to min_t, so need to make a final step
        t_1 = ts[-1]
        return pred_aatypes_1, prot_traj, clean_traj



def fm_model_step(model, noisy_batch):
    loss_mask = noisy_batch['mask'] * noisy_batch['chain_M']
    loss_denom = torch.sum(loss_mask, dim=-1)
    if torch.any(torch.sum(loss_mask, dim=-1) < 1):
        raise ValueError('Empty batch encountered')
    num_batch, num_res = loss_mask.shape

    # ground truth labels
    gt_aatypes_1 = noisy_batch['S']

    # Timestep used for normalization.
    t = noisy_batch['t']

    # if args.aatypes_loss_use_likelihood_weighting:
    #     cat_norm_scale = 1 - torch.min(
    #         t, torch.tensor(args.t_normalize_clip)) # (B, 1)
    #     assert cat_norm_scale.shape == (num_batch, 1)
    # else:
    #     cat_norm_scale = 1.0

    # Model output predictions.
    X = noisy_batch['X']
    aatypes_t_1 = noisy_batch['S_t']
    mask = noisy_batch['mask']
    chain_M = noisy_batch['chain_M']
    residue_idx = noisy_batch['residue_idx']
    chain_encoding_all = noisy_batch['chain_encoding_all']

    pred_logits = model(X, aatypes_t_1, mask, chain_M, residue_idx, chain_encoding_all)

    # aatypes loss
    # ce_loss = torch.nn.functional.cross_entropy(
    #     pred_logits.reshape(-1, len(mu.ALPHABET_WITH_MASK)),
    #     gt_aatypes_1.flatten().long(),
    #     reduction='none',
    # ).reshape(num_batch, num_res) # / cat_norm_scale
    # aatypes_loss = torch.sum(ce_loss * loss_mask, dim=-1) / loss_denom

    return pred_logits

def get_likelihood(model, batch, num_steps, device, noise_interpolant, eps=1e-5):
    X, S, mask, chain_M, residue_idx, chain_encoding_all = batch # featurize(batch, device)
    timesteps = torch.linspace(
      eps, 1-eps, num_steps + 1, device=device) # t=1 is clean data
    dt = (1 - 2*eps) / num_steps
    log_p_at_time_list = []
    for i in range(num_steps):
        t = timesteps[i]
        multiplier = dt/(1-t) * torch.ones(S.shape[0], device=device)
        noisy_batch = noise_interpolant.corrupt_batch((X, S, mask, chain_M, residue_idx, chain_encoding_all), t=t)
        mask_for_loss = mask*chain_M
    
        log_probs = fm_model_step(model, noisy_batch)
        log_p_s0 = log_probs.gather(-1, S[..., None]).squeeze(-1)
        log_p_s0 = (log_p_s0 * mask_for_loss).sum(dim=-1) * multiplier
        log_p_at_time_list.append(log_p_s0)
    log_p_at_time = torch.stack(log_p_at_time_list, dim=0).sum(dim=0)
    return log_p_at_time
