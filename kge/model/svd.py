import math
import torch
import time

from kge import Config, Dataset
from kge.model.kge_model import KgeEmbedder, KgeModel, RelationalScorer


class SvdScorer(RelationalScorer):
    r"""Implementation of the SVD KGE scorer."""

    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        super().__init__(config, dataset, configuration_key)

    def score_emb(
        self,
        s_emb: torch.Tensor,
        p_emb: torch.Tensor,
        o_emb: torch.Tensor,
        combine: str,
    ):
        rank = self.get_option("rank")
        batch_size = p_emb.size(0)
        entity_size = s_emb.size(-1)

        p_matrix_decomp = p_emb.view(-1, entity_size, 2 * rank)
        u_matrices = p_matrix_decomp[:, :, 0:rank]
        v_matrices = p_matrix_decomp[:, :, rank:(2 * rank)].permute(0, 2, 1)

        if combine == "spo":
            out = (
                s_emb.unsqueeze(1)  # [batch x 1 x entity_size]
                .bmm(u_matrices)  # apply mixing matrices
                .bmm(v_matrices)
                .view(batch_size, entity_size)  # drop dim 1
                * o_emb  # apply object embeddings
            ).sum(
                dim=-1
            )  # and sum to obtain predictions
        elif combine == "sp_":
            out = (
                s_emb.unsqueeze(1)
                .bmm(u_matrices)
                .bmm(v_matrices)
                .view(batch_size, entity_size)
                .mm(o_emb.transpose(0, 1))
            )
        elif combine == "_po":
            out = (
                u_matrices.bmm(v_matrices).bmm(o_emb.unsqueeze(2))
                .view(batch_size, entity_size)
                .mm(s_emb.transpose(0, 1))
            )
        else:
            super().score_emb(s_emb, p_emb, o_emb, combine)

        return out.view(batch_size, -1)

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
        dim = 2 * config.get_default(ent_emb_conf_key + ".dim") * rank
        config.set(rel_emb_conf_key + ".dim", dim, log=True)
