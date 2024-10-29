from typing import Dict, Tuple

import torch

from scaling.core.topology.topology import Topology
from scaling.transformer.context.config import ContrastiveLossFunctionConfig
from scaling.transformer.data.text_dataset_batch import TextDatasetBatch
from scaling.transformer.model.layers.base import TransformerLayerIO


class ContrastiveLoss(torch.nn.Module):
    def __init__(
        self,
        topology: Topology,
        loss_config: ContrastiveLossFunctionConfig,
    ):
        super().__init__()
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction="mean")
        self.nr_samples_per_batch = loss_config.number_of_hard_negatives + 2
        self.nr_passages = loss_config.number_of_hard_negatives + 1
        self.nr_hard_negatives = loss_config.number_of_hard_negatives
        self.log_verbose_metrics = loss_config.log_verbose_metrics
        self.scale = loss_config.scale
        self.topology = topology

    def forward(
        self, output: TransformerLayerIO, batch: TextDatasetBatch
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        activations = self._gather_activations(output)
        queries, passages = self._split_queries_passages(activations)
        scores = self._compute_scores(queries, passages)
        target = self._compute_target(scores, passages, queries)

        loss = self.cross_entropy.forward(scores, target)

        if self.log_verbose_metrics:
            metrics = self._compute_metrics(scores, target)
        else:
            metrics = {}

        return loss, metrics

    def _gather_activations(self, output: TransformerLayerIO) -> torch.Tensor:
        if self.topology.config.data_parallel_size > 1:
            return self._dist_gather_tensor(output.activations)
        return output.activations

    def _split_queries_passages(self, activations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert activations is not None
        queries = activations[:: self.nr_samples_per_batch, :]
        passages_indices = [
            i + j for i in range(1, len(activations), self.nr_passages + 1) for j in range(self.nr_passages)
        ]
        passages = activations[passages_indices]
        return queries, passages

    def _compute_scores(self, queries: torch.Tensor, passages: torch.Tensor) -> torch.Tensor:
        return ContrastiveLoss._cos_sim(queries, passages) * self.scale

    def _compute_target(self, scores: torch.Tensor, passages: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        target *= passages.size(0) // queries.size(0)
        return target

    def _compute_retrieved_targets(self, scores: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        row_indices = torch.arange(target.unsqueeze(1).size(0)).unsqueeze(1).cuda(self.topology.device)
        combined_indices = torch.cat((row_indices.cuda(), target.unsqueeze(1)), dim=1)
        return scores[combined_indices[:, 0], combined_indices[:, 1]]

    @staticmethod
    def _cos_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
        b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
        return torch.mm(a_norm, b_norm.transpose(0, 1))

    def _compute_metrics(self, scores: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        retrieved_targets = self._compute_retrieved_targets(scores, target)
        avg_in_batch_negative_cosine, avg_in_batch_negative_cosine_abs = self._compute_in_batch_negative_cosine(
            scores, target, retrieved_targets
        )
        top_10_map, top_5_map, top_1_map = (
            ContrastiveLoss._calculate_map(scores, target, 10) if self.nr_samples_per_batch >= 10 else None,
            ContrastiveLoss._calculate_map(scores, target, 5) if self.nr_samples_per_batch >= 5 else None,
            ContrastiveLoss._calculate_map(scores, target, 1),
        )
        top_10_acc, top_5_acc, top_1_acc = (
            ContrastiveLoss._compute_top_k_accuracy(scores, target, 10) if self.nr_samples_per_batch >= 10 else None,
            ContrastiveLoss._compute_top_k_accuracy(scores, target, 5) if self.nr_samples_per_batch >= 5 else None,
            ContrastiveLoss._compute_top_k_accuracy(scores, target, 1),
        )

        metrics = {
            "average_cosine": scores.detach().mean(),
            "average_target_cosine_similarity": retrieved_targets.detach().cpu().mean(),
            "average_in_batch_negative_cosine": torch.stack(avg_in_batch_negative_cosine).detach().mean().cpu(),
            "average_absolute_in_batch_negative_cosine": torch.stack(avg_in_batch_negative_cosine_abs)
            .detach()
            .mean()
            .cpu(),
            "top_1_acc": top_1_acc,
            "top_1_map": top_1_map,
        }

        if self.nr_hard_negatives > 0:
            hard_negative_cosine_avg = self._compute_hard_negative_cosine_avg(scores, target)
            positive_hard_negative_delta = self._compute_positive_hard_negative_delta(scores, target, retrieved_targets)
            metrics["positive_hard_negative_delta"] = positive_hard_negative_delta
            metrics["average_hard_negative_cosine_similarity"] = hard_negative_cosine_avg

        if top_10_acc is not None and top_10_map is not None:
            metrics["top_10_acc"] = top_10_acc
            metrics["top_10_map"] = top_10_map

        if top_5_acc is not None and top_5_map is not None:
            metrics["top_5_acc"] = top_5_acc
            metrics["top_5_map"] = top_5_map

        return metrics

    def _compute_hard_negative_cosine_avg(self, scores: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        hard_negative_cosine_list = self._compute_hard_negative_cosine_list(scores, target)
        return torch.tensor(hard_negative_cosine_list).detach().mean().cpu()

    def _compute_hard_negative_cosine_list(self, scores: torch.Tensor, target: torch.Tensor) -> list[list[float]]:
        hard_negative_cosine_list = []
        row_indices = torch.arange(target.unsqueeze(1).size(0)).unsqueeze(1).cuda(self.topology.device)
        combined_indices = torch.cat((row_indices.cuda(), target.unsqueeze(1)), dim=1)
        for row in combined_indices:
            row_scores = [float(scores[row[0]][row[1] + 1 + i].item()) for i in range(self.nr_hard_negatives)]
            hard_negative_cosine_list.append(row_scores)
        return hard_negative_cosine_list

    def _compute_in_batch_negative_cosine(
        self, scores: torch.Tensor, target: torch.Tensor, retrieved_targets: torch.Tensor
    ) -> Tuple[list[torch.Tensor], list[torch.Tensor]]:
        avg_in_batch_negative_cosine = []
        avg_in_batch_negative_cosine_abs = []
        hard_negative_cosine_list = (
            self._compute_hard_negative_cosine_list(scores, target) if self.nr_hard_negatives > 0 else []
        )
        for i, (target_cosine, row) in enumerate(zip(retrieved_targets, scores)):
            in_batch_negative_sum, in_batch_negative_sum_abs = self._compute_in_batch_negative_sums(
                row, target_cosine, hard_negative_cosine_list, i
            )
            avg_in_batch_negative_cosine.append(in_batch_negative_sum)
            avg_in_batch_negative_cosine_abs.append(in_batch_negative_sum_abs)
        return avg_in_batch_negative_cosine, avg_in_batch_negative_cosine_abs

    def _compute_in_batch_negative_sums(
        self, row: torch.Tensor, target_cosine: torch.Tensor, hard_negative_cosine_list: list[list[float]], i: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.nr_hard_negatives > 0:
            in_batch_negative_sum = (row.sum() - target_cosine - sum(hard_negative_cosine_list[i])) / (
                len(row) - 1 - len(hard_negative_cosine_list[i])
            )
            in_batch_negative_sum_abs = (row.abs().sum() - target_cosine - sum(hard_negative_cosine_list[i])) / (
                len(row) - 1 - len(hard_negative_cosine_list[i])
            )
        else:
            in_batch_negative_sum = (row.sum() - target_cosine) / len(row) - 1
            in_batch_negative_sum_abs = (row.abs().sum() - target_cosine) / len(row) - 1
        return in_batch_negative_sum, in_batch_negative_sum_abs

    @staticmethod
    def _calculate_map(scores: torch.Tensor, correct_indices: torch.Tensor, k: int) -> torch.Tensor:
        num_queries = scores.size(0)
        average_precisions = torch.zeros(num_queries)

        for i in range(num_queries):
            query_scores = scores[i]
            correct_index = correct_indices[i]
            sorted_indices = torch.argsort(query_scores, descending=True)
            ranks = torch.where(sorted_indices == correct_index)[0]
            if ranks.size(0) > 0 and ranks[0] < k:
                tp_count = 1
                rank = ranks[0].item()
                precision_at_k = tp_count / (rank + 1)
                average_precisions[i] = precision_at_k

        mean_average_precision = average_precisions.mean()
        return mean_average_precision

    @staticmethod
    def _compute_top_k_accuracy(scores: torch.Tensor, target: torch.Tensor, k: int) -> torch.Tensor:
        _, indices = scores.topk(k)
        top_acc = torch.tensor([target[i] in indices[i] for i in range(target.size(-1))]).detach().float().mean()

        return top_acc

    def _compute_positive_hard_negative_delta(
        self, scores: torch.Tensor, target: torch.Tensor, retrieved_targets: torch.Tensor
    ) -> torch.Tensor:
        hard_negative_cosine_list = self._compute_hard_negative_cosine_list(scores, target)
        hard_negative_cosine = torch.tensor(hard_negative_cosine_list).cuda(self.topology.device)
        retrieved_targets_expanded = retrieved_targets.unsqueeze(-1).repeat((1, self.nr_hard_negatives))
        return torch.abs(retrieved_targets_expanded - hard_negative_cosine).detach().mean()

    def _dist_gather_tensor(self, t: torch.Tensor) -> torch.Tensor:
        t = t.contiguous()
        all_tensors = [torch.empty_like(t) for _ in range(len(self.topology.data_parallel_ranks))]
        # All tensors have the same shape, as pooling already applied to them
        torch.distributed.all_gather(all_tensors, t, group=self.topology.data_parallel_group)
        all_tensors[self.topology.data_parallel_rank] = t

        return torch.cat(all_tensors, dim=0)
