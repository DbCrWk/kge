import torch
import math

from kge import Config, Dataset
from kge.model.kge_model import RelationalScorer, KgeModel

class SvdScorer(RelationalScorer):
    r"""Implementation of the SVD scorer."""

    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        super().__init__(config, dataset, configuration_key)

    # def score_single_example(self, s_emb, p_emb, o_emb):
    #     """
    #     s_emb is 1-by-K vector, p is (r * (1 + 2K))-by-1 vector, o_emb is K-by-1 vector
    #     """
    #     entity_size = s_emb.size(-1)
    #     p_matrix_decomp = p_emb.view(rank, 1 + 2 * entity_size)                     # r-by-(1 + 2K)
    #     svd_values = p_matrix_decomp[:, :1] * torch.eye(rank)                       # r-by-r
    #     u_matrices = p_matrix_decomp[:, 1:(entity_size + 1)].permute(1, 0)          # K-by-r
    #     v_matrices = p_matrix_decomp[:, (entity_size + 1):(2 * entity_size + 1)]    # r-by-K

    #     left_side = torch.matmul(s_emb, u_matrices)                                 # (1-by-K) * (K-by-r) = (1-by-r)
    #     right_side = torch.matmul(v_matrices, o_emb)                                # (r-by-K) * (K-by-1) = (r-by-1)

    #     left_side_weighted = torch.matmul(left_side, svd_values)                    # (1-by-r) * (r-by-r) = (1-by-r)

    #     out = torch.matmul(left_side_weighted, right_side)                          # (1-by-r) * (r-by-1) = (1-by-1)
    #     return out

    # def score_emb(
    #     self,
    #     s_emb: torch.Tensor,
    #     p_emb: torch.Tensor,
    #     o_emb: torch.Tensor,
    #     combine: str,
    # ):
    #     batch_size = p_emb.size(0)
    #     o_len = o_emb.size(0)
    #     return torch.zeros(batch_size, o_len)

    def score_emb_spo(
        self,
        s_emb,                                                                      # N-by-K
        p_emb,
        o_emb                                                                       # N-by-K
    ):
        rank = self.get_option("rank")

        batch_size = p_emb.size(0)
        entity_size = s_emb.size(-1)
        s_emb_ext = torch.unsqueeze(s_emb, -1)                                      # N-by-K-by-1
        o_emb_ext = torch.unsqueeze(o_emb, -1)                                      # N-by-K-by-1

        p_matrix_decomp = p_emb.view(
            batch_size, rank, 1 + 2 * entity_size)
        svd_values = p_matrix_decomp[:, :, :1] * torch.eye(rank)                    # N-by-r-by-r
        u_matrices = p_matrix_decomp[:, :, 1:(entity_size + 1)]                     # N-by-r-by-K
        v_matrices = p_matrix_decomp[
            :,
            :,
            (entity_size + 1):(2 * entity_size + 1)
        ]                                                                           # N-by-r-by-K

        # See: https://pytorch.org/docs/stable/torch.html#torch.matmul
        # Batched matrix multiplication
        # N-by-r-by-K * N-by-K-by-1 = N-by-r-by-1 -> N-by-1-by-r
        left_side = torch.matmul(u_matrices, s_emb_ext).permute(0, 2, 1)
        
        # N-by-1-by-r * N-by-r-by-r = N-by-1-by-r
        weighted_left_side = torch.matmul(left_side, svd_values)
        
        # N-by-r-by-K * N-by-K-by-1 = N-by-r-by-1
        right_side = torch.matmul(v_matrices, o_emb_ext)
        
        # N-by-1-by-r * N-by-r-by-1 = N-by-1-by-1
        out_raw = torch.matmul(weighted_left_side, right_side)

        # We need to remove the last dimension to return an N-by-1 dimensional object
        out = torch.squeeze(out_raw, dim=2)

        return out



class Svd(KgeModel):
    r"""Implementation of the proposed SVD KGE model."""

    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        self._init_configuration(config, configuration_key)
        svd_set_relation_embedder_dim(
            config, dataset, self.configuration_key
        )
        super().__init__(
            config, dataset, SvdScorer, configuration_key=self.configuration_key
        )


def svd_set_relation_embedder_dim(config, dataset, conf_key):
    """Set the relation embedder dimensionality for SVD in the config.

    If <0, set it to 2 * rank * emb_dim + rank. Else leave
    unchanged.

    """
    rel_emb_conf_key = conf_key + ".relation_embedder"
    rank = config.get_default(conf_key + ".rank")
    dim = config.get_default(rel_emb_conf_key + ".dim")
    if dim < 0:  # autodetect relation embedding dimensionality
        ent_emb_conf_key = rel_emb_conf_key.replace(
            "relation_embedder", "entity_embedder"
        )
        if ent_emb_conf_key == rel_emb_conf_key:
            raise ValueError(
                "Cannot determine relation embedding size; please set manually."
            )
        dim = 2 * config.get_default(ent_emb_conf_key + ".dim") * rank + rank
        config.set(rel_emb_conf_key + ".dim", dim, log=True)
