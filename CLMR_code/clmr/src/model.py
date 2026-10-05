from __future__ import annotations

import copy
import math

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel


def masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(hidden.dtype).unsqueeze(-1)
    return (hidden * weight).sum(1) / weight.sum(1).clamp_min(1.0)


def enable_compatible_gradient_checkpointing(encoder: nn.Module) -> None:

    model_type = getattr(getattr(encoder, "config", None), "model_type", "")
    if model_type == "deberta-v2":
        encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    else:
        encoder.gradient_checkpointing_enable()


def corpus_relative_background(
    z_target: torch.Tensor,
    pos_ids: torch.Tensor,
    lemma_ids: torch.Tensor,
    available: torch.Tensor,
    memory_vectors: torch.Tensor,
    memory_pos_ids: torch.Tensor,
    memory_lemma_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    if memory_vectors.shape[0] == 0:
        return (
            torch.zeros(z_target.shape[0], device=z_target.device),
            torch.zeros(z_target.shape[0], device=z_target.device, dtype=torch.bool),
            torch.zeros_like(z_target, dtype=torch.float32),
        )
    similarity = z_target.float() @ memory_vectors.float().T
    candidates = (
        pos_ids[:, None].eq(memory_pos_ids[None, :])
        & ~lemma_ids[:, None].eq(memory_lemma_ids[None, :])
    )
    boundary_available = available & candidates.any(-1)
    masked = similarity.masked_fill(~candidates, -1e4)
    maximum, nearest_index = masked.max(-1)
    maximum = torch.where(
        boundary_available, maximum, torch.zeros_like(maximum)
    )
    nearest_vector = memory_vectors[nearest_index].float()
    nearest_vector = torch.where(
        boundary_available.unsqueeze(-1), nearest_vector,
        torch.zeros_like(nearest_vector),
    )
    return maximum, boundary_available, nearest_vector


class Projection(nn.Module):
    def __init__(self, source: int, target: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(source, target), nn.GELU(),
                                     nn.LayerNorm(target), nn.Dropout(dropout))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


class SingleEncoderTargetClassifier(nn.Module):

    def __init__(self, pretrained_model: str, dropout: float = .1,
                 gradient_checkpointing: bool = True):
        super().__init__()
        self.context_encoder = AutoModel.from_pretrained(pretrained_model)


        self.knowledge_encoder = nn.Identity()
        if gradient_checkpointing:
            enable_compatible_gradient_checkpointing(self.context_encoder)
        hidden = self.context_encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden), nn.Dropout(dropout), nn.Linear(hidden, 2)
        )
        self.relative_boundary = False
        self.dual_expert = False

    def forward(self, batch: dict[str, torch.Tensor], **_: object
                ) -> dict[str, torch.Tensor]:
        hidden = self.context_encoder(
            input_ids=batch["context_input_ids"],
            attention_mask=batch["context_attention_mask"],
        ).last_hidden_state
        target = masked_mean(hidden, batch["target_mask"])
        logits = self.classifier(target)
        zero = logits.new_zeros(())
        batch_zero = logits.new_zeros(logits.shape[0])
        batch_false = torch.zeros(
            logits.shape[0], dtype=torch.bool, device=logits.device
        )
        return {
            "logits": logits,
            "base_expert_logits": logits,
            "boundary_expert_logits": logits,
            "auxiliary_loss": zero,
            "boundary_router_loss": zero,
            "compatibility": batch_zero,
            "manifold_route_gate": batch_zero,
            "boundary_route_gate": batch_zero,
            "boundary_utility": batch_zero,
            "relative_boundary": batch_zero,
            "boundary_available": batch_false,
        }


class EvidenceDCRLayer(nn.Module):

    def __init__(self, hidden_size: int, expansion: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, expansion), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(expansion, hidden_size),
        )
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, evidence: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(evidence)
        return self.norm2(evidence + self.dropout(self.ffn(normalized)))


class KnowledgeInteractionModulator(nn.Module):

    def __init__(self, hidden_size: int, rank: int = 256,
                 conditioner_size: int = 768, dropout: float = .1):
        super().__init__()
        relation_size = hidden_size * 4
        self.conditioner = nn.Sequential(
            nn.LayerNorm(relation_size),
            nn.Linear(relation_size, conditioner_size), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(conditioner_size, rank), nn.Tanh(),
        )
        self.context_projection = nn.Linear(hidden_size, rank, bias=False)
        self.output_projection = nn.Linear(rank, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, contextual: torch.Tensor, literal: torch.Tensor,
                available: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        relation = torch.cat((contextual, literal,
                              torch.abs(contextual-literal), contextual*literal), dim=-1)
        channels = self.conditioner(relation)
        update = self.output_projection(channels*self.context_projection(contextual))
        update = self.dropout(update)*available.unsqueeze(-1).to(update.dtype)
        return contextual+update, update


class LiteralEvidenceSetInteraction(nn.Module):

    def __init__(self, semantic_size: int, hidden_size: int, layers: int = 2,
                 heads: int = 8, expansion: int = 1536,
                 source_types: int = 5, dropout: float = .1):
        super().__init__()


        self.element_projection = nn.Sequential(
            nn.LayerNorm(semantic_size * 6),
            nn.Linear(semantic_size * 6, hidden_size), nn.GELU(),
            nn.Dropout(dropout),
        )
        self.source_embedding = nn.Embedding(source_types, hidden_size)
        layer = lambda: nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=heads, dim_feedforward=expansion,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.set_layers = nn.ModuleList([layer() for _ in range(int(layers))])
        self.query_projection = nn.Linear(semantic_size, hidden_size)
        self.cross_attention = nn.MultiheadAttention(
            hidden_size, heads, dropout=dropout, batch_first=True
        )
        self.output_norm = nn.LayerNorm(hidden_size)
        self.output_projection = nn.Linear(hidden_size, hidden_size)


        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, target: torch.Tensor, literal_word: torch.Tensor,
                literal_context: torch.Tensor, valid: torch.Tensor,
                source_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        expanded_target = target.unsqueeze(1).expand_as(literal_word)
        features = torch.cat((
            literal_word, literal_context,
            torch.abs(expanded_target-literal_word), expanded_target*literal_word,
            torch.abs(expanded_target-literal_context), expanded_target*literal_context,
        ), dim=-1)
        elements = self.element_projection(features)
        elements = elements + self.source_embedding(source_ids.clamp(0, 4))

        safe_valid = valid.clone()
        unavailable = ~safe_valid.any(1)
        safe_valid[unavailable, 0] = True
        elements = torch.where(safe_valid.unsqueeze(-1), elements,
                               torch.zeros_like(elements))
        padding = ~safe_valid
        for layer in self.set_layers:
            elements = layer(elements, src_key_padding_mask=padding)
        query = self.query_projection(target).unsqueeze(1)
        pooled, weights = self.cross_attention(
            query, elements, elements, key_padding_mask=padding,
            need_weights=True, average_attn_weights=False,
        )
        update = self.output_projection(self.output_norm(pooled.squeeze(1)))
        update = update * (~unavailable).unsqueeze(-1).to(update.dtype)
        return update, weights.squeeze(2)


class ContextualMIPRefiner(nn.Module):

    def __init__(self, source_size: int, hidden_size: int = 384,
                 bottleneck: int = 256, dropout: float = .1):
        super().__init__()
        self.context_projection = Projection(source_size, hidden_size, dropout)
        self.gloss_projection = Projection(source_size, hidden_size, dropout)
        self.relation = nn.Sequential(
            nn.LayerNorm(hidden_size * 4),
            nn.Linear(hidden_size * 4, hidden_size * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_size * 2, bottleneck), nn.GELU(),
            nn.Dropout(dropout),
        )
        self.output = nn.Linear(bottleneck, 2)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.compatibility_scale = nn.Parameter(torch.tensor(8.0))
        self.compatibility_boundary = nn.Parameter(torch.tensor(0.25))

    def forward(self, contextual_target: torch.Tensor, basic_gloss: torch.Tensor,
                available: torch.Tensor, labels: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context = self.context_projection(contextual_target)
        gloss = self.gloss_projection(basic_gloss)
        relation = torch.cat((context, gloss, torch.abs(context-gloss), context*gloss), dim=-1)
        delta = self.output(self.relation(relation))
        delta = delta * available.unsqueeze(-1).to(delta.dtype)
        similarity = F.cosine_similarity(context.float(), gloss.float(), dim=-1)
        literal_target = 1.0-labels.float()
        logit = self.compatibility_scale.clamp(1.0,20.0)*(similarity-self.compatibility_boundary)
        loss = (F.binary_cross_entropy_with_logits(
            logit[available], literal_target[available])
            if available.any() else delta.sum()*0)
        return delta, loss, similarity


class FrozenSpaceMIPCalibrator(nn.Module):

    def __init__(self, hidden_size: int = 96, dropout: float = .1):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(6), nn.Linear(6, hidden_size), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_size, hidden_size), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_size, 2),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        self.scale = nn.Parameter(torch.tensor(8.0))
        self.boundary = nn.Parameter(torch.tensor(0.25))

    def forward(self, target: torch.Tensor, gloss: torch.Tensor,
                base_logits: torch.Tensor, available: torch.Tensor,
                labels: torch.Tensor):
        similarity = F.cosine_similarity(target.float(), gloss.float(), dim=-1)
        margin = (base_logits[:,1]-base_logits[:,0]).float()
        present = available.float()
        features = torch.stack((similarity, similarity.square(), margin,
                                similarity*margin, margin.abs(), present), dim=-1)
        delta = self.network(features.to(base_logits.dtype))*present.unsqueeze(-1)
        literal = 1.0-labels.float()
        score = self.scale.clamp(1.0,20.0)*(similarity-self.boundary)
        loss = (F.binary_cross_entropy_with_logits(score[available],literal[available])
                if available.any() else delta.sum()*0)
        return delta,loss,similarity


class LMRExpert(nn.Module):

    def __init__(self, pretrained_model: str, representation_size: int = 384,
                 semantic_size: int = 256, dropout: float = .1,
                 gradient_checkpointing: bool = True,
                 hard_negative_weight: float = 0.0,
                 hard_negative_temperature: float = 0.08,
                 dcr_layers: int = 0,
                 dcr_expansion: int | None = None,
                 kim_rank: int = 0,
                 kim_conditioner_size: int = 768,
                 external_mip: bool = False,
                 external_mip_rank: int = 256,
                 contextual_mip: bool = False,
                 contextual_mip_bottleneck: int = 256,
                 frozen_space_mip: bool = False,
                 ambiguity_routing: bool = False,
                 relative_boundary: bool = False,
                 boundary_loss_weight: float = 0.0,
                 boundary_routing: bool = False,
                 boundary_router_hidden: int = 384,
                 boundary_router_warmup: float = 0.2,
                 boundary_router_loss_weight: float = 0.5,
                 boundary_utility_temperature: float = 0.05,
                 dual_expert: bool = False,
                 set_interaction_layers: int = 0,
                 set_interaction_heads: int = 8,
                 set_interaction_expansion: int = 1536):
        super().__init__()
        self.context_encoder = AutoModel.from_pretrained(pretrained_model)
        self.knowledge_encoder = AutoModel.from_pretrained(pretrained_model)
        if gradient_checkpointing:
            enable_compatible_gradient_checkpointing(self.context_encoder)
            enable_compatible_gradient_checkpointing(self.knowledge_encoder)
        hidden = self.context_encoder.config.hidden_size
        self.target_projection = Projection(hidden, representation_size, dropout)
        self.context_projection = Projection(hidden, representation_size, dropout)
        self.context_relation = Projection(hidden * 2, representation_size, dropout)

        self.context_semantic = Projection(hidden, semantic_size, dropout)
        self.literal_semantic = Projection(hidden, semantic_size, dropout)
        self.literal_relation = Projection(semantic_size * 2 + 4, representation_size, dropout)
        self.ambiguity_routing = bool(ambiguity_routing)
        self.relative_boundary = bool(relative_boundary)
        self.boundary_loss_weight = float(boundary_loss_weight)
        self.boundary_routing = bool(boundary_routing)
        self.boundary_router_warmup = float(boundary_router_warmup)
        self.boundary_router_loss_weight = float(boundary_router_loss_weight)
        self.boundary_utility_temperature = float(boundary_utility_temperature)
        self.dual_expert = bool(dual_expert)
        if self.boundary_routing and not self.relative_boundary:
            raise ValueError("boundary_routing requires relative_boundary=true")
        if self.dual_expert and not self.relative_boundary:
            raise ValueError("dual_expert requires relative_boundary=true")




        self.register_buffer(
            "boundary_memory_vectors",
            torch.empty((0, semantic_size), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "boundary_memory_pos_ids", torch.empty(0, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "boundary_memory_lemma_ids", torch.empty(0, dtype=torch.long),
            persistent=False,
        )
        self.pos_embedding = nn.Embedding(17, representation_size)
        self.classifier = nn.Sequential(
            nn.LayerNorm(representation_size * 4), nn.Dropout(dropout),
            nn.Linear(representation_size * 4, representation_size), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(representation_size, 2),
        )


        self.boundary_classifier = (
            copy.deepcopy(self.classifier) if self.dual_expert else None
        )



        self.routing_pos_embedding = (
            nn.Embedding(17, 8) if self.ambiguity_routing else None
        )
        self.manifold_router = (
            nn.Sequential(
                nn.LayerNorm(14), nn.Linear(14, 64), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(64, 1),
            ) if self.ambiguity_routing else None
        )
        self.boundary_projection = (
            nn.Sequential(
                nn.LayerNorm(4), nn.Linear(4, representation_size), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(representation_size, representation_size),
            ) if self.relative_boundary else None
        )
        if self.boundary_projection is not None:


            nn.init.zeros_(self.boundary_projection[-1].weight)
            nn.init.zeros_(self.boundary_projection[-1].bias)
            self.boundary_scale = nn.Parameter(torch.tensor(8.0))
            self.boundary_threshold = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_parameter("boundary_scale", None)
            self.register_parameter("boundary_threshold", None)





        boundary_router_input = semantic_size * 7 + 4 + 8
        self.boundary_routing_pos_embedding = (
            nn.Embedding(17, 8) if self.boundary_routing else None
        )
        self.boundary_router = (
            nn.Sequential(
                nn.LayerNorm(boundary_router_input),
                nn.Linear(boundary_router_input, int(boundary_router_hidden)),
                nn.GELU(), nn.Dropout(dropout),
                nn.Linear(int(boundary_router_hidden), 128),
                nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 1),
            ) if self.boundary_routing else None
        )
        self.literal_set_interaction = (
            LiteralEvidenceSetInteraction(
                semantic_size=semantic_size, hidden_size=representation_size,
                layers=int(set_interaction_layers),
                heads=int(set_interaction_heads),
                expansion=int(set_interaction_expansion), dropout=dropout,
            ) if int(set_interaction_layers) > 0 else None
        )


        expansion = int(dcr_expansion or representation_size * 6)
        self.evidence_refinement = nn.ModuleList([
            EvidenceDCRLayer(representation_size, expansion, dropout)
            for _ in range(int(dcr_layers))
        ])
        self.knowledge_modulator = (
            KnowledgeInteractionModulator(
                representation_size, rank=int(kim_rank),
                conditioner_size=int(kim_conditioner_size), dropout=dropout,
            ) if int(kim_rank)>0 else None
        )



        self.external_knowledge_encoder = (
            AutoModel.from_pretrained(pretrained_model, add_pooling_layer=False)
            if (external_mip or contextual_mip or frozen_space_mip) else None
        )
        if self.external_knowledge_encoder is not None:
            for parameter in self.external_knowledge_encoder.parameters():
                parameter.requires_grad = False
            self.external_mip_relation = Projection(hidden * 2, representation_size, dropout)
            self.external_mip_modulator = KnowledgeInteractionModulator(
                representation_size, rank=int(external_mip_rank),
                conditioner_size=int(kim_conditioner_size), dropout=dropout,
            )
        else:
            self.external_mip_relation = None
            self.external_mip_modulator = None
        self.contextual_mip_refiner = (
            ContextualMIPRefiner(hidden, representation_size,
                                 int(contextual_mip_bottleneck), dropout)
            if contextual_mip else None
        )
        self.frozen_space_mip_calibrator = (
            FrozenSpaceMIPCalibrator(dropout=dropout) if frozen_space_mip else None)
        self.compatibility_scale = nn.Parameter(torch.tensor(8.0))
        self.compatibility_boundary = nn.Parameter(torch.tensor(0.25))
        self.hard_negative_weight=float(hard_negative_weight)
        self.hard_negative_temperature=float(hard_negative_temperature)
        self.training_progress=1.0

    def load_external_knowledge_encoder(self, path: str) -> None:
        if self.external_knowledge_encoder is None:
            raise RuntimeError("external_mip was not enabled")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        state = payload.get("knowledge_encoder", payload)
        self.external_knowledge_encoder.load_state_dict(state, strict=True)
        self.external_knowledge_encoder.requires_grad_(False)

    def initialize_literal_encoder_from_mip(self, path: str) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        state = payload.get("knowledge_encoder", payload)
        missing, unexpected = self.knowledge_encoder.load_state_dict(state, strict=False)
        allowed_missing = {"pooler.dense.weight", "pooler.dense.bias"}
        if unexpected or set(missing) - allowed_missing:
            raise RuntimeError(
                f"Incompatible MIP initializer: missing={missing}, unexpected={unexpected}")

    def set_training_progress(self,value:float)->None:
        self.training_progress=float(min(max(value,0.0),1.0))

    def classify_without_dropout(self, representation: torch.Tensor) -> torch.Tensor:

        value = self.classifier[0](representation)
        value = self.classifier[2](value)
        value = self.classifier[3](value)
        return self.classifier[5](value)

    @torch.inference_mode()
    def encode_boundary_candidates(
        self, batch: dict[str, torch.Tensor]
    ) -> torch.Tensor:

        evidence_hidden = self.knowledge_encoder(
            input_ids=batch["evidence_input_ids"],
            attention_mask=batch["evidence_attention_mask"],
        ).last_hidden_state
        batch_size, prototype_count = batch["prototype_valid_mask"].shape
        hidden_size = evidence_hidden.shape[-1]
        literal_target = masked_mean(
            evidence_hidden, batch["literal_target_mask"]
        ).reshape(batch_size, prototype_count, hidden_size)
        literal_context = masked_mean(
            evidence_hidden, batch["example_mask"]
        ).reshape(batch_size, prototype_count, hidden_size)
        source = .35 * literal_target + .65 * literal_context
        prototypes = F.normalize(self.literal_semantic(source).float(), dim=-1)
        valid = batch["prototype_valid_mask"].bool()
        centroid = (prototypes * valid.unsqueeze(-1)).sum(1)
        centroid = centroid / valid.sum(1, keepdim=True).clamp_min(1)
        return F.normalize(centroid, dim=-1)

    @torch.inference_mode()
    def set_boundary_memory(
        self,
        vectors: torch.Tensor,
        pos_ids: torch.Tensor,
        lemma_ids: torch.Tensor,
    ) -> None:

        if vectors.ndim != 2 or vectors.shape[1] != self.boundary_memory_vectors.shape[1]:
            raise ValueError(f"Invalid boundary vectors: {tuple(vectors.shape)}")
        if not (vectors.shape[0] == pos_ids.numel() == lemma_ids.numel()):
            raise ValueError("Boundary vector/POS/lemma counts do not match")
        device = next(self.parameters()).device
        self.boundary_memory_vectors = F.normalize(
            vectors.detach().to(device=device, dtype=torch.float32), dim=-1
        )
        self.boundary_memory_pos_ids = pos_ids.detach().to(device=device, dtype=torch.long)
        self.boundary_memory_lemma_ids = lemma_ids.detach().to(device=device, dtype=torch.long)

    def forward(self, batch: dict[str, torch.Tensor],
                disable_dcr: bool = False,
                disable_kim: bool = False,
                disable_external_mip: bool = False,
                disable_contextual_mip: bool = False,
                disable_frozen_space_mip: bool = False,
                disable_set_interaction: bool = False,
                route_override: float | None = None) -> dict[str, torch.Tensor]:
        context_hidden = self.context_encoder(
            input_ids=batch["context_input_ids"],
            attention_mask=batch["context_attention_mask"],
        ).last_hidden_state
        evidence_hidden = self.knowledge_encoder(
            input_ids=batch["evidence_input_ids"],
            attention_mask=batch["evidence_attention_mask"],
        ).last_hidden_state

        batch_size, prototype_count = batch["prototype_valid_mask"].shape
        hidden_size = context_hidden.shape[-1]
        target = masked_mean(context_hidden, batch["target_mask"])
        context_mask = batch["context_attention_mask"].bool() & ~batch["target_mask"]
        context = masked_mean(context_hidden, context_mask)
        literal_target = masked_mean(evidence_hidden, batch["literal_target_mask"]).reshape(
            batch_size, prototype_count, hidden_size)
        literal_context = masked_mean(evidence_hidden, batch["example_mask"]).reshape(
            batch_size, prototype_count, hidden_size)

        z_target = F.normalize(self.context_semantic(target).float(), dim=-1)
        z_literal_word = F.normalize(self.literal_semantic(literal_target).float(), dim=-1)
        z_literal_context = F.normalize(self.literal_semantic(literal_context).float(), dim=-1)

        prototype_source = .35 * literal_target + .65 * literal_context
        z_prototype = F.normalize(self.literal_semantic(prototype_source).float(), dim=-1)
        similarity = torch.einsum("bd,bkd->bk", z_target, z_prototype)
        valid = batch["prototype_valid_mask"].bool()
        available = valid.any(1)
        safe = valid.clone(); unavailable = ~safe.any(1); safe[unavailable, 0] = True
        masked_similarity = similarity.masked_fill(~safe, -1e4)
        attention = torch.softmax(masked_similarity / .08, dim=-1)
        attention = attention * valid.to(attention.dtype)
        attention = attention / attention.sum(-1, keepdim=True).clamp_min(1.0)
        prototype = torch.einsum("bk,bkd->bd", attention, z_prototype)

        maximum = masked_similarity.max(-1).values
        maximum = torch.where(unavailable, torch.zeros_like(maximum), maximum)
        mean = (similarity * valid).sum(-1) / valid.sum(-1).clamp_min(1)
        variance = ((similarity - mean.unsqueeze(-1)).pow(2) * valid).sum(-1)
        variance = variance / valid.sum(-1).clamp_min(1)
        entropy = -(attention * attention.clamp_min(1e-8).log()).sum(-1)


        statistics = torch.stack((maximum, mean, variance.clamp_min(1e-6).sqrt(), entropy), dim=-1)
        manifold_relation = self.literal_relation(torch.cat(
            (torch.abs(z_target - prototype), z_target * prototype, statistics), dim=-1))
        set_update = manifold_relation.new_zeros(manifold_relation.shape)
        set_attention = attention.unsqueeze(1)
        if self.literal_set_interaction is not None and not disable_set_interaction:
            set_update, set_attention = self.literal_set_interaction(
                z_target, z_literal_word, z_literal_context, valid,
                batch["prototype_source_ids"],
            )
            manifold_relation = manifold_relation + set_update

        valid_count = valid.sum(-1)
        if prototype_count >= 2:
            top_values = masked_similarity.topk(k=2, dim=-1).values
            top_gap = top_values[:, 0] - top_values[:, 1]
            top_gap = torch.where(valid_count.ge(2), top_gap, torch.zeros_like(top_gap))
        else:
            top_gap = torch.zeros_like(maximum)
        route_gate = torch.ones_like(maximum)
        literal_relation = manifold_relation
        if self.manifold_router is not None:
            basic = z_prototype[:, 0]
            basic_similarity = (z_target * basic).sum(-1)
            basic_statistics = torch.stack((
                basic_similarity, basic_similarity,
                torch.zeros_like(basic_similarity), torch.zeros_like(basic_similarity),
            ), dim=-1)
            basic_relation = self.literal_relation(torch.cat((
                torch.abs(z_target - basic), z_target * basic, basic_statistics,
            ), dim=-1))
            routing_features = torch.cat((
                statistics, top_gap.unsqueeze(-1),
                valid_count.float().div(float(prototype_count)).unsqueeze(-1),
                self.routing_pos_embedding(batch["pos_ids"]),
            ), dim=-1)
            route_gate = torch.sigmoid(self.manifold_router(routing_features)).squeeze(-1)
            route_gate = torch.where(available, route_gate, torch.zeros_like(route_gate))
            if route_override is not None:
                fixed = torch.full_like(route_gate, float(route_override))
                route_gate = torch.where(available, fixed, torch.zeros_like(fixed))
            literal_relation = (route_gate.unsqueeze(-1) * manifold_relation +
                                (1.0-route_gate).unsqueeze(-1) * basic_relation)





        relative_boundary_value = torch.zeros_like(maximum)
        background_maximum = torch.zeros_like(maximum)
        boundary_available = torch.zeros_like(available)
        background_prototype = torch.zeros_like(z_target)
        boundary_route_gate = torch.ones_like(maximum)
        boundary_route_logit = torch.zeros_like(maximum)
        boundary_router_loss = maximum.sum() * 0
        boundary_utility = torch.zeros_like(maximum)
        boundary_relation_base = literal_relation
        boundary_relation_full = literal_relation
        boundary_loss = maximum.sum() * 0
        if self.boundary_projection is not None:
            if self.boundary_memory_vectors.shape[0] > 0:
                (background_maximum, boundary_available,
                 background_prototype) = corpus_relative_background(
                    z_target,
                    batch["pos_ids"],
                    batch["lemma_ids"],
                    available,
                    self.boundary_memory_vectors,
                    self.boundary_memory_pos_ids,
                    self.boundary_memory_lemma_ids,
                )
            relative_boundary_value = maximum - background_maximum
            boundary_features = torch.stack((
                relative_boundary_value, maximum, background_maximum, entropy,
            ), dim=-1)
            boundary_update = self.boundary_projection(boundary_features)
            boundary_relation_full = boundary_relation_base + boundary_update
            if self.boundary_router is not None:
                boundary_routing_features = torch.cat((
                    z_target, prototype, background_prototype,
                    torch.abs(z_target - prototype), z_target * prototype,
                    torch.abs(z_target - background_prototype),
                    z_target * background_prototype, boundary_features,
                    self.boundary_routing_pos_embedding(batch["pos_ids"]),
                ), dim=-1)
                boundary_route_logit = self.boundary_router(
                    boundary_routing_features).squeeze(-1)
                boundary_route_gate = torch.sigmoid(boundary_route_logit)
                boundary_route_gate = torch.where(
                    boundary_available, boundary_route_gate,
                    torch.zeros_like(boundary_route_gate),
                )



                if (self.training and
                        self.training_progress <= self.boundary_router_warmup):
                    boundary_route_gate = torch.where(
                        boundary_available, torch.ones_like(boundary_route_gate),
                        torch.zeros_like(boundary_route_gate),
                    )
                boundary_update = boundary_update * boundary_route_gate.unsqueeze(-1)
            literal_relation = boundary_relation_base + boundary_update

        contextual_relation_base = self.context_relation(torch.cat(
            (torch.abs(target - context), target * context), dim=-1))
        contextual_relation = contextual_relation_base
        kim_update = contextual_relation.new_zeros(contextual_relation.shape)
        if self.knowledge_modulator is not None and not disable_kim:
            contextual_relation,kim_update=self.knowledge_modulator(
                contextual_relation,literal_relation,available)
        external_update = contextual_relation.new_zeros(contextual_relation.shape)
        external_word = external_gloss = external_context_target = None
        if self.external_knowledge_encoder is not None:


            self.external_knowledge_encoder.eval()
            with torch.no_grad():
                external_hidden = self.external_knowledge_encoder(
                    input_ids=batch["external_input_ids"],
                    attention_mask=batch["external_attention_mask"],
                ).last_hidden_state
                external_vector = masked_mean(external_hidden, batch["external_attention_mask"])
                if self.frozen_space_mip_calibrator is not None:
                    external_context_hidden = self.external_knowledge_encoder(
                        input_ids=batch["context_input_ids"],
                        attention_mask=batch["context_attention_mask"],
                    ).last_hidden_state
                    external_context_target = masked_mean(
                        external_context_hidden,batch["target_mask"])
            external_word, external_gloss = external_vector.chunk(2, dim=0)
        if self.external_mip_modulator is not None and not disable_external_mip:
            external_relation = self.external_mip_relation(torch.cat((
                torch.abs(external_word-external_gloss), external_word*external_gloss,
            ), dim=-1))
            contextual_relation,external_update = self.external_mip_modulator(
                contextual_relation, external_relation, batch["external_available"].bool())
        target_evidence = self.target_projection(target) + self.pos_embedding(batch["pos_ids"])
        context_evidence = self.context_projection(context)
        evidence_units = torch.stack((
            target_evidence, context_evidence, contextual_relation, literal_relation,
        ), dim=1)
        base_representation = evidence_units.flatten(1)
        if (self.boundary_router is not None and
                self.training_progress > self.boundary_router_warmup and
                boundary_available.any()):



            with torch.no_grad():
                route_base = torch.stack((
                    target_evidence, context_evidence, contextual_relation,
                    boundary_relation_base,
                ), dim=1).flatten(1)
                route_full = torch.stack((
                    target_evidence, context_evidence, contextual_relation,
                    boundary_relation_full,
                ), dim=1).flatten(1)
                base_route_logits = self.classify_without_dropout(route_base)
                full_route_logits = self.classify_without_dropout(route_full)
                base_route_loss = F.cross_entropy(
                    base_route_logits, batch["labels"], reduction="none")
                full_route_loss = F.cross_entropy(
                    full_route_logits, batch["labels"], reduction="none")
                boundary_utility = base_route_loss - full_route_loss
                route_target = boundary_utility.gt(0).to(boundary_route_logit.dtype)
                route_weight = (boundary_utility.abs() /
                                self.boundary_utility_temperature).clamp(max=1.0)
                route_weight = route_weight * boundary_available.to(route_weight.dtype)
            if self.training:
                per_item_route_loss = F.binary_cross_entropy_with_logits(
                    boundary_route_logit, route_target, reduction="none")
                boundary_router_loss = (
                    (per_item_route_loss * route_weight).sum() /
                    route_weight.sum().clamp_min(1e-6)
                )
        refined = evidence_units
        if not disable_dcr:
            for layer in self.evidence_refinement:
                refined = layer(refined)
        if self.dual_expert:


            base_expert_units = torch.stack((
                target_evidence, context_evidence, contextual_relation,
                manifold_relation,
            ), dim=1)
            base_expert_refined = base_expert_units
            if not disable_dcr:
                for layer in self.evidence_refinement:
                    base_expert_refined = layer(base_expert_refined)
            base_expert_logits = self.classifier(base_expert_refined.flatten(1))
            boundary_expert_logits = self.boundary_classifier(refined.flatten(1))


            logits = torch.logaddexp(
                F.log_softmax(base_expert_logits.float(), dim=-1),
                F.log_softmax(boundary_expert_logits.float(), dim=-1),
            ) - math.log(2.0)
        else:
            logits = self.classifier(refined.flatten(1))
            base_expert_logits = logits
            boundary_expert_logits = logits
        contextual_mip_delta = logits.new_zeros(logits.shape)
        contextual_mip_loss = logits.sum()*0
        contextual_mip_similarity = logits.new_zeros((logits.shape[0],))
        if self.contextual_mip_refiner is not None and not disable_contextual_mip:
            contextual_mip_delta,contextual_mip_loss,contextual_mip_similarity = (
                self.contextual_mip_refiner(
                    target, external_gloss, batch["external_available"].bool(), batch["labels"]))
            logits = logits + contextual_mip_delta
        frozen_space_delta = logits.new_zeros(logits.shape)
        frozen_space_loss = logits.sum()*0
        frozen_space_similarity = logits.new_zeros((logits.shape[0],))
        if self.frozen_space_mip_calibrator is not None and not disable_frozen_space_mip:
            frozen_space_delta,frozen_space_loss,frozen_space_similarity = (
                self.frozen_space_mip_calibrator(
                    external_context_target,external_gloss,logits,
                    batch["external_available"].bool(),batch["labels"]))
            logits = logits+frozen_space_delta


        if self.training or (not self.evidence_refinement and self.knowledge_modulator is None):
            base_logits = logits.detach()
        elif disable_dcr and (disable_kim or self.knowledge_modulator is None):
            base_logits = logits
        else:
            unmodulated_units=torch.stack((
                evidence_units[:,0],evidence_units[:,1],contextual_relation_base,evidence_units[:,3]
            ),dim=1)
            base_logits=self.classifier(unmodulated_units.flatten(1))


        compatible = 1.0 - batch["labels"].float()
        compatibility_logit = self.compatibility_scale.clamp(1.0, 20.0) * (
            maximum - self.compatibility_boundary)
        compatibility_loss = F.binary_cross_entropy_with_logits(
            compatibility_logit[available], compatible[available]) if available.any() else logits.sum() * 0
        if self.boundary_projection is not None and boundary_available.any():
            boundary_logit = self.boundary_scale.clamp(1.0, 20.0) * (
                relative_boundary_value - self.boundary_threshold)
            boundary_loss = F.binary_cross_entropy_with_logits(
                boundary_logit[boundary_available], compatible[boundary_available])
        matching_loss=logits.sum()*0
        if self.hard_negative_weight>0 and "lemma_ids" in batch:



            matching=(z_target@prototype.float().T)/self.hard_negative_temperature
            same_pos=batch["pos_ids"][:,None].eq(batch["pos_ids"][None,:])
            same_lemma=batch["lemma_ids"][:,None].eq(batch["lemma_ids"][None,:])
            candidates=same_pos&available[None,:]
            positives=candidates&same_lemma
            has_negative=(candidates&~same_lemma).any(1)
            queries=batch["labels"].eq(0)&available&has_negative
            if queries.any():
                denominator=torch.logsumexp(matching.masked_fill(~candidates,-1e4),dim=1)
                numerator=torch.logsumexp(matching.masked_fill(~positives,-1e4),dim=1)
                matching_loss=(denominator-numerator)[queries].mean()
        curriculum=self.training_progress
        auxiliary=(compatibility_loss+contextual_mip_loss+frozen_space_loss+
                   self.boundary_loss_weight*boundary_loss+
                   self.boundary_router_loss_weight*boundary_router_loss+
                   self.hard_negative_weight*curriculum*matching_loss)
        return {"logits": logits, "base_logits": base_logits,
                "base_expert_logits":base_expert_logits,
                "boundary_expert_logits":boundary_expert_logits,
                "dcr_logit_delta": (logits-base_logits).detach(),
                "kim_update_norm":kim_update.float().norm(dim=-1).detach(),
                "external_mip_update_norm":external_update.float().norm(dim=-1).detach(),
                "set_interaction_update_norm":set_update.float().norm(dim=-1).detach(),
                "set_interaction_attention":set_attention.detach(),
                "contextual_mip_logit_norm":contextual_mip_delta.float().norm(dim=-1).detach(),
                "contextual_mip_similarity":contextual_mip_similarity.detach(),
                "frozen_space_mip_logit_norm":frozen_space_delta.float().norm(dim=-1).detach(),
                "frozen_space_mip_similarity":frozen_space_similarity.detach(),
                "auxiliary_loss": auxiliary,
                "compatibility": maximum, "prototype_attention": attention,
                "matching_loss":matching_loss.detach(),
                "manifold_route_gate":route_gate.detach(),
                "boundary_route_gate":boundary_route_gate.detach(),
                "boundary_router_loss":boundary_router_loss.detach(),
                "boundary_utility":boundary_utility.detach(),
                "relative_boundary":relative_boundary_value.detach(),
                "boundary_available":boundary_available.detach(),
                "boundary_loss":boundary_loss.detach()}
