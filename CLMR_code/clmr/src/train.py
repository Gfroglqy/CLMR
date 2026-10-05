from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

HERE = Path(__file__).resolve().parent


sys.path.insert(0, str(HERE))
from ark_dcr.data import (LiteralPrototypeMemory, PrototypeEvidenceCollator,
                          PrototypeEvidenceDataset, PrototypeExample,
                          canonical_lemma, load_examples)
from ark_dcr.metrics import binary_metrics
from model import LMRExpert, SingleEncoderTargetClassifier
from external_mip import ExternalMIPCollator


def symmetric_kl(first_logits: torch.Tensor, second_logits: torch.Tensor) -> torch.Tensor:
    first_log = F.log_softmax(first_logits.float(), dim=-1)
    second_log = F.log_softmax(second_logits.float(), dim=-1)
    first_probability = first_log.exp()
    second_probability = second_log.exp()
    forward = F.kl_div(first_log, second_probability, reduction="none").sum(dim=-1)
    reverse = F.kl_div(second_log, first_probability, reduction="none").sum(dim=-1)
    return 0.5 * (forward + reverse).mean()


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def move(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()}


def shuffled_prototypes(items: list[PrototypeExample], seed: int) -> list[PrototypeExample]:
    rng=np.random.default_rng(seed);replacement=[list(item.prototypes) for item in items]
    positions={}
    for row,item in enumerate(items):
        for slot in range(len(item.prototypes)):
            positions.setdefault((item.example.pos_tag.upper(),slot),[]).append(row)
    for (pos,slot),rows in positions.items():
        if len(rows)<2:continue
        order=rng.permutation(np.asarray(rows));sources=np.roll(order,1)
        for destination,source in zip(order,sources):
            replacement[int(destination)][slot]=items[int(source)].prototypes[slot]
    return [replace(item,prototypes=tuple(prototypes)) for item,prototypes in zip(items,replacement)]


def query_only_prototypes(items: list[PrototypeExample]) -> list[PrototypeExample]:
    query_sources={LiteralPrototypeMemory.QUERY_GLOSS,LiteralPrototypeMemory.QUERY_EXAMPLE}
    return [replace(item,prototypes=tuple(
        prototype for prototype in item.prototypes if prototype.source_id in query_sources))
        for item in items]


def select_boundary_memory_items(
    items: list[PrototypeExample],
) -> list[PrototypeExample]:

    selected: dict[tuple[str, str], PrototypeExample] = {}
    for item in sorted(items, key=lambda value: value.example.row_id):
        example = item.example
        if example.label != 0 or not item.prototypes:
            continue
        key = (canonical_lemma(example.target_word, example.pos_tag),
               example.pos_tag.upper())
        selected.setdefault(key, item)
    return [selected[key] for key in sorted(selected)]


@torch.inference_mode()
def refresh_boundary_memory(model, loader, device) -> int:

    was_training = model.training
    model.eval()
    vectors: list[torch.Tensor] = []
    pos_ids: list[torch.Tensor] = []
    lemma_ids: list[torch.Tensor] = []
    for batch in tqdm(loader, desc="refresh boundary memory", leave=False):
        batch = move(batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            vectors.append(model.encode_boundary_candidates(batch).float().cpu())
        pos_ids.append(batch["pos_ids"].cpu())
        lemma_ids.append(batch["lemma_ids"].cpu())
    if not vectors:
        raise RuntimeError("No training-only literal items available for boundary memory")
    model.set_boundary_memory(
        torch.cat(vectors), torch.cat(pos_ids), torch.cat(lemma_ids)
    )
    model.train(was_training)
    return sum(value.shape[0] for value in vectors)


@torch.inference_mode()
def evaluate(model, loader, device, disable_dcr: bool = False,
             disable_kim: bool = False, disable_external_mip: bool = False,
             disable_contextual_mip: bool = False,
             disable_frozen_space_mip: bool = False,
             disable_set_interaction: bool = False,
             route_override: float | None = None):
    model.eval(); labels=[]; probabilities=[]; row_ids=[]; compatibilities=[]
    base_expert_probabilities=[];boundary_expert_probabilities=[]
    route_gates=[]; boundary_route_gates=[]; boundary_utilities=[]
    relative_boundaries=[]; boundary_available=[]
    for batch in tqdm(loader, desc="evaluate", leave=False):
        batch=move(batch,device)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            output=model(batch,disable_dcr=disable_dcr,disable_kim=disable_kim,
                         disable_external_mip=disable_external_mip,
                         disable_contextual_mip=disable_contextual_mip,
                         disable_frozen_space_mip=disable_frozen_space_mip,
                         disable_set_interaction=disable_set_interaction,
                         route_override=route_override)
        probabilities.extend(output["logits"].float().softmax(-1)[:,1].cpu().tolist())
        base_expert_probabilities.extend(
            output["base_expert_logits"].float().softmax(-1)[:,1].cpu().tolist())
        boundary_expert_probabilities.extend(
            output["boundary_expert_logits"].float().softmax(-1)[:,1].cpu().tolist())
        compatibilities.extend(output["compatibility"].float().cpu().tolist())
        route_gates.extend(output["manifold_route_gate"].float().cpu().tolist())
        boundary_route_gates.extend(output["boundary_route_gate"].float().cpu().tolist())
        boundary_utilities.extend(output["boundary_utility"].float().cpu().tolist())
        relative_boundaries.extend(output["relative_boundary"].float().cpu().tolist())
        boundary_available.extend(output["boundary_available"].bool().cpu().tolist())
        labels.extend(batch["labels"].cpu().tolist());row_ids.extend(batch["row_ids"].cpu().tolist())
    frame=pd.DataFrame({"row_id":row_ids,"label":labels,"probability":probabilities,
                        "base_expert_probability":base_expert_probabilities,
                        "boundary_expert_probability":boundary_expert_probabilities,
                        "compatibility":compatibilities,
                        "manifold_route_gate":route_gates,
                        "boundary_route_gate":boundary_route_gates,
                        "boundary_utility":boundary_utilities,
                        "relative_boundary":relative_boundaries,
                        "boundary_available":boundary_available}).sort_values("row_id")
    metrics=binary_metrics(frame.label,frame.probability)
    metrics["base_expert_f1"]=binary_metrics(
        frame.label,frame.base_expert_probability)["f1"]
    metrics["boundary_expert_f1"]=binary_metrics(
        frame.label,frame.boundary_expert_probability)["f1"]
    base_prediction=frame.base_expert_probability.ge(.5)
    boundary_prediction=frame.boundary_expert_probability.ge(.5)
    label_boolean=frame.label.astype(bool)
    base_correct=base_prediction.eq(label_boolean)
    boundary_correct=boundary_prediction.eq(label_boolean)
    metrics["expert_disagreement_rate"]=float(
        base_prediction.ne(boundary_prediction).mean())
    metrics["base_expert_unique_correct_rate"]=float(
        (base_correct & ~boundary_correct).mean())
    metrics["boundary_expert_unique_correct_rate"]=float(
        (boundary_correct & ~base_correct).mean())
    metrics["literal_compatibility_mean"]=float(frame.loc[frame.label.eq(0),"compatibility"].mean())
    metrics["metaphor_compatibility_mean"]=float(frame.loc[frame.label.eq(1),"compatibility"].mean())
    metrics["compatibility_gap"]=metrics["literal_compatibility_mean"]-metrics["metaphor_compatibility_mean"]
    if frame.boundary_available.any():
        selected=frame.loc[frame.boundary_available]
        metrics["boundary_coverage"]=float(frame.boundary_available.mean())
        metrics["literal_boundary_mean"]=float(
            selected.loc[selected.label.eq(0),"relative_boundary"].mean())
        metrics["metaphor_boundary_mean"]=float(
            selected.loc[selected.label.eq(1),"relative_boundary"].mean())
        metrics["relative_boundary_gap"]=(metrics["literal_boundary_mean"]-
                                          metrics["metaphor_boundary_mean"])
        metrics["manifold_route_gate_mean"]=float(selected.manifold_route_gate.mean())
        metrics["boundary_route_gate_mean"]=float(selected.boundary_route_gate.mean())
        metrics["literal_boundary_route_gate_mean"]=float(
            selected.loc[selected.label.eq(0),"boundary_route_gate"].mean())
        metrics["metaphor_boundary_route_gate_mean"]=float(
            selected.loc[selected.label.eq(1),"boundary_route_gate"].mean())
        useful=selected.boundary_utility.abs().gt(1e-8)
        if useful.any():
            routed=selected.loc[useful]
            metrics["boundary_utility_positive_rate"]=float(
                routed.boundary_utility.gt(0).mean())
            metrics["boundary_route_utility_agreement"]=float((
                routed.boundary_route_gate.ge(0.5)==
                routed.boundary_utility.gt(0)).mean())
    return metrics,frame


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--seed",type=int,default=13);parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--data-dir",type=Path,help="Override config data_dir for canonical folds")
    parser.add_argument("--additional-train-dir",type=Path,action="append",default=[],
                        help="Append another corpus train.csv during source-stage training")
    parser.add_argument("--additional-memory-dir",type=Path,action="append",default=[],
                        help="Use another corpus only to enrich literal prototypes; do not train its labels")
    parser.add_argument("--epochs",type=int)
    parser.add_argument("--eval-batch-size",type=int,
                        help="Override evaluation batch size for invariance audits")
    parser.add_argument("--knowledge-mode",choices=("real","shuffled"),default="real")
    parser.add_argument("--query-only-prototypes",action="store_true",
                        help="Ablate retrieved train usages while retaining gloss/example anchors")
    parser.add_argument("--evaluate-checkpoint",type=Path)
    parser.add_argument("--evaluation-split",choices=("train","validation","test"),default="validation",
                        help="Split used by --evaluate-checkpoint")
    parser.add_argument("--teacher-probabilities",type=Path,
                        help="CSV with row_id and teacher_probability for ensemble distillation")
    parser.add_argument("--distill-weight",type=float,default=0.0)
    parser.add_argument("--distill-temperature",type=float,default=2.0)
    parser.add_argument("--initial-checkpoint",type=Path,
                        help="Optional LMR checkpoint used before task-specific fine-tuning")
    parser.add_argument("--initial-encoder-checkpoint",type=Path,
                        help="Load only context/knowledge encoders from a V1 LMR checkpoint")
    parser.add_argument("--overlap-checkpoint",type=Path,
                        help="Warm-start all shape-compatible parameters")
    parser.add_argument("--train-external-mip-only",action="store_true",
                        help="Freeze CLMR and train only the external MIP residual")
    parser.add_argument("--train-contextual-mip-only",action="store_true",
                        help="Freeze CLMR and train only contextual MIP alignment")
    parser.add_argument("--train-frozen-space-mip-only",action="store_true")
    parser.add_argument("--train-boundary-expert-only",action="store_true",
                        help="Freeze the warm-started base path and train only boundary modules")
    parser.add_argument("--train-set-interaction-only",action="store_true",
                        help="Freeze a warm-started CLMR and train only the literal set adapter")
    parser.add_argument("--evaluate-test",action="store_true",
                        help="After validation selection, evaluate the saved best model on test.csv")
    parser.add_argument("--freeze-knowledge-encoder",action="store_true")
    parser.add_argument("--freeze-context-encoder",action="store_true")
    parser.add_argument("--compact-checkpoint",action="store_true",
                        help="Save only trainable parameters (for frozen-encoder transfer runs)")
    parser.add_argument("--disable-dcr",action="store_true",
                        help="Evaluate a DCR checkpoint with evidence refinement disabled")
    parser.add_argument("--disable-kim",action="store_true",
                        help="Evaluate a KIM checkpoint with knowledge modulation disabled")
    parser.add_argument("--disable-external-mip",action="store_true",
                        help="Evaluate with the external MIP modulation disabled")
    parser.add_argument("--disable-contextual-mip",action="store_true",
                        help="Evaluate with contextual MIP alignment disabled")
    parser.add_argument("--disable-frozen-space-mip",action="store_true")
    parser.add_argument("--disable-set-interaction",action="store_true",
                        help="Same-checkpoint ablation of the literal evidence set layer")
    parser.add_argument("--route-override",type=float,
                        help="Validation-only intervention: replace learned manifold route gate")
    args=parser.parse_args();config_path=args.config.resolve();config=json.loads(config_path.read_text(encoding="utf-8"))
    if args.epochs is not None:config["epochs"]=args.epochs
    if args.eval_batch_size is not None:config["eval_batch_size"]=args.eval_batch_size
    resolve=lambda value:str((config_path.parent/Path(value)).resolve())
    pretrained=resolve(config["pretrained_model"])
    data_dir=args.data_dir.resolve() if args.data_dir else Path(resolve(config["data_dir"]))
    args.output.mkdir(parents=True,exist_ok=True);(args.output/"checkpoints").mkdir(exist_ok=True)
    seed_all(args.seed);device=torch.device("cuda");torch.cuda.reset_peak_memory_stats()
    tokenizer=AutoTokenizer.from_pretrained(
        pretrained,use_fast=True,add_prefix_space=True,local_files_only=True,
        fix_mistral_regex=True)
    raw_train=load_examples(data_dir/"train.csv")
    for additional_dir in args.additional_train_dir:
        raw_train.extend(load_examples(additional_dir.resolve()/"train.csv"))


    raw_train=[replace(example,row_id=index) for index,example in enumerate(raw_train)]
    evaluation_file=({"train":"train.csv","validation":"val.csv","test":"test.csv"}
                     [args.evaluation_split] if args.evaluate_checkpoint else "val.csv")
    raw_valid=load_examples(data_dir/evaluation_file)
    memory_train=list(raw_train)
    for additional_dir in args.additional_memory_dir:
        extra=load_examples(additional_dir.resolve()/"train.csv")
        offset=len(memory_train)
        memory_train.extend(replace(example,row_id=offset+index)
                            for index,example in enumerate(extra))



    memory=LiteralPrototypeMemory(memory_train,max_candidates=config["max_memory_candidates"],anchor_first=True)
    train_items=memory.materialize(raw_train,config["prototype_count"],training=True)
    valid_items=memory.materialize(
        raw_valid,config["prototype_count"],
        training=bool(args.evaluate_checkpoint and args.evaluation_split=="train"))
    if args.knowledge_mode=="shuffled":
        train_items=shuffled_prototypes(train_items,args.seed+3000)
        valid_items=shuffled_prototypes(valid_items,args.seed+4000)
    if args.query_only_prototypes:
        train_items=query_only_prototypes(train_items)
        valid_items=query_only_prototypes(valid_items)
    train_set=PrototypeEvidenceDataset(train_items);valid_set=PrototypeEvidenceDataset(valid_items)
    collator=PrototypeEvidenceCollator(tokenizer,config["prototype_count"],config["max_context_length"],config["max_evidence_length"])
    if (config.get("external_mip",False) or config.get("contextual_mip",False)
            or config.get("frozen_space_mip",False)):
        collator=ExternalMIPCollator(
            collator,tokenizer,Path(resolve(config["external_gloss_path"])),
            config.get("external_mip_pos","VERB"),
            config.get("external_mip_mode","real"),
            config.get("external_mip_source","csv"))
    generator=torch.Generator().manual_seed(args.seed)
    train_loader=DataLoader(train_set,batch_size=config["batch_size"],shuffle=True,collate_fn=collator,
        num_workers=config["num_workers"],pin_memory=True,generator=generator)
    valid_loader=DataLoader(valid_set,batch_size=config["eval_batch_size"],shuffle=False,collate_fn=collator,
        num_workers=config["num_workers"],pin_memory=True)
    boundary_memory_items = select_boundary_memory_items(train_items)
    boundary_memory_loader = DataLoader(
        PrototypeEvidenceDataset(boundary_memory_items),
        batch_size=config["eval_batch_size"], shuffle=False, collate_fn=collator,
        num_workers=config["num_workers"], pin_memory=True,
    )
    if config.get("model_type") == "single_target":
        model=SingleEncoderTargetClassifier(
            pretrained,dropout=config["dropout"],
            gradient_checkpointing=config.get("gradient_checkpointing",True)
        ).to(device)
    else:
        model=LMRExpert(pretrained,config["representation_size"],config["semantic_size"],config["dropout"],
            gradient_checkpointing=config.get("gradient_checkpointing",True),
            hard_negative_weight=config.get("hard_negative_weight",0.0),
            hard_negative_temperature=config.get("hard_negative_temperature",0.08),
            dcr_layers=config.get("dcr_layers",0),
            dcr_expansion=config.get("dcr_expansion"),
            kim_rank=config.get("kim_rank",0),
            kim_conditioner_size=config.get("kim_conditioner_size",768),
            external_mip=config.get("external_mip",False),
            external_mip_rank=config.get("external_mip_rank",256),
            contextual_mip=config.get("contextual_mip",False),
            contextual_mip_bottleneck=config.get("contextual_mip_bottleneck",256),
            frozen_space_mip=config.get("frozen_space_mip",False),
            ambiguity_routing=config.get("ambiguity_routing",False),
            relative_boundary=config.get("relative_boundary",False),
            boundary_loss_weight=config.get("boundary_loss_weight",0.0),
            boundary_routing=config.get("boundary_routing",False),
            boundary_router_hidden=config.get("boundary_router_hidden",384),
            boundary_router_warmup=config.get("boundary_router_warmup",0.2),
            boundary_router_loss_weight=config.get("boundary_router_loss_weight",0.5),
            boundary_utility_temperature=config.get("boundary_utility_temperature",0.05),
            dual_expert=config.get("dual_expert",False),
            set_interaction_layers=config.get("set_interaction_layers",0),
            set_interaction_heads=config.get("set_interaction_heads",8),
            set_interaction_expansion=config.get("set_interaction_expansion",1536)).to(device)
    if (config.get("external_mip",False) or config.get("contextual_mip",False)
            or config.get("frozen_space_mip",False)):
        model.load_external_knowledge_encoder(resolve(config["external_mip_checkpoint"]))
    if config.get("literal_mip_initializer"):
        model.initialize_literal_encoder_from_mip(resolve(config["literal_mip_initializer"]))
    if args.overlap_checkpoint:
        payload=torch.load(args.overlap_checkpoint,map_location="cpu",weights_only=True)
        source=payload.get("model",payload)
        target=model.state_dict()
        compatible={key:value for key,value in source.items()
                    if key in target and target[key].shape==value.shape}
        missing,unexpected=model.load_state_dict(compatible,strict=False)
        if unexpected or not compatible:
            raise RuntimeError(f"Invalid overlap checkpoint: loaded={len(compatible)}, unexpected={unexpected}")
    if args.train_boundary_expert_only:
        if not model.dual_expert or model.boundary_classifier is None:
            raise RuntimeError("--train-boundary-expert-only requires dual_expert=true")
        if model.boundary_projection is None or model.manifold_router is None:
            raise RuntimeError(
                "Boundary-only training requires relative_boundary and ambiguity_routing")


        model.boundary_classifier.load_state_dict(model.classifier.state_dict())
        trainable_prefixes=(
            "routing_pos_embedding.", "manifold_router.",
            "boundary_projection.", "boundary_classifier.",
        )
        trainable_scalars={"boundary_scale", "boundary_threshold"}
        for name,parameter in model.named_parameters():
            parameter.requires_grad=(
                name.startswith(trainable_prefixes) or name in trainable_scalars)
    if args.train_set_interaction_only:
        if model.literal_set_interaction is None:
            raise RuntimeError(
                "--train-set-interaction-only requires set_interaction_layers > 0")
        if not args.overlap_checkpoint:
            raise RuntimeError(
                "--train-set-interaction-only requires a warm-start --overlap-checkpoint")
        for name,parameter in model.named_parameters():
            parameter.requires_grad=name.startswith("literal_set_interaction.")
    if args.train_external_mip_only:
        if model.external_mip_modulator is None:
            raise RuntimeError("--train-external-mip-only requires external_mip=true")
        for name,parameter in model.named_parameters():
            parameter.requires_grad=(name.startswith("external_mip_relation.") or
                                     name.startswith("external_mip_modulator."))
    if args.train_contextual_mip_only:
        if model.contextual_mip_refiner is None:
            raise RuntimeError("--train-contextual-mip-only requires contextual_mip=true")
        for name,parameter in model.named_parameters():
            parameter.requires_grad=name.startswith("contextual_mip_refiner.")
    if args.train_frozen_space_mip_only:
        if model.frozen_space_mip_calibrator is None:
            raise RuntimeError("frozen_space_mip=true is required")
        for name,parameter in model.named_parameters():
            parameter.requires_grad=name.startswith("frozen_space_mip_calibrator.")
    if args.initial_checkpoint:
        initial=torch.load(args.initial_checkpoint,map_location="cpu",weights_only=True)
        model.load_state_dict(initial["model"],strict=True)
    if args.initial_encoder_checkpoint:
        initial=torch.load(args.initial_encoder_checkpoint,map_location="cpu",weights_only=True)["model"]
        encoder_state={key:value for key,value in initial.items()
                       if key.startswith("context_encoder.") or key.startswith("knowledge_encoder.")}
        missing,unexpected=model.load_state_dict(encoder_state,strict=False)
        if unexpected or not encoder_state:
            raise RuntimeError(f"Invalid encoder checkpoint: unexpected={unexpected}, keys={len(encoder_state)}")
    if args.freeze_knowledge_encoder:
        for parameter in model.knowledge_encoder.parameters(): parameter.requires_grad=False
    if args.freeze_context_encoder:
        for parameter in model.context_encoder.parameters(): parameter.requires_grad=False
    if args.evaluate_checkpoint:
        checkpoint=torch.load(args.evaluate_checkpoint,map_location="cpu",weights_only=True)
        model.load_state_dict(checkpoint["model"],strict=not checkpoint.get("compact",False))
        boundary_memory_entries = (refresh_boundary_memory(
            model,boundary_memory_loader,device) if model.relative_boundary else 0)
        metrics,frame=evaluate(model,valid_loader,device,disable_dcr=args.disable_dcr,
                               disable_kim=args.disable_kim,
                               disable_external_mip=args.disable_external_mip,
                               disable_contextual_mip=args.disable_contextual_mip,
                               disable_frozen_space_mip=args.disable_frozen_space_mip,
                               disable_set_interaction=args.disable_set_interaction,
                               route_override=args.route_override)
        suffix = ""
        for enabled, candidate in (
            (args.query_only_prototypes, "_query_only"),
            (args.disable_dcr, "_dcr_off"),
            (args.disable_kim, "_kim_off"),
            (args.disable_external_mip, "_external_mip_off"),
            (args.disable_contextual_mip, "_contextual_mip_off"),
            (args.disable_frozen_space_mip, "_frozen_space_mip_off"),
            (args.disable_set_interaction, "_set_interaction_off"),
        ):
            if enabled:
                suffix = candidate
                break
        if not suffix and args.route_override is not None:
            suffix = f"_route_{args.route_override:g}"
        prediction_name=f"{args.evaluation_split}_predictions{suffix}.csv"
        frame.to_csv(args.output/prediction_name,index=False)
        summary={"seed":args.seed,"validation_only":args.evaluation_split=="validation",
                 "evaluation_split":args.evaluation_split,"knowledge_mode":args.knowledge_mode,
                 "query_only_prototypes":args.query_only_prototypes,
                 "dcr_disabled":args.disable_dcr,"kim_disabled":args.disable_kim,
                 "external_mip_disabled":args.disable_external_mip,
                 "contextual_mip_disabled":args.disable_contextual_mip,
                 "frozen_space_mip_disabled":args.disable_frozen_space_mip,
                 "set_interaction_disabled":args.disable_set_interaction,
                 "route_override":args.route_override,
                 "boundary_memory_entries":boundary_memory_entries,
                 "checkpoint":str(args.evaluate_checkpoint.resolve()),args.evaluation_split:metrics}
        metrics_name=f"metrics{suffix}.json"
        (args.output/metrics_name).write_text(json.dumps(summary,indent=2),encoding="utf-8")
        print(json.dumps(summary,indent=2));return
    context_ids={id(p) for p in model.context_encoder.parameters()}
    knowledge_ids={id(p) for p in model.knowledge_encoder.parameters()}
    context=[p for p in model.parameters() if id(p) in context_ids and p.requires_grad]
    knowledge=[p for p in model.parameters() if id(p) in knowledge_ids and p.requires_grad]
    head=[p for p in model.parameters()
          if id(p) not in context_ids and id(p) not in knowledge_ids and p.requires_grad]
    optimizer=AdamW([{"params":context,"lr":config["encoder_lr"]},
                     {"params":knowledge,"lr":config.get("knowledge_encoder_lr",config["encoder_lr"])},
                     {"params":head,"lr":config["head_lr"]}],
                    weight_decay=config["weight_decay"])
    updates=(len(train_loader)+config["gradient_accumulation"]-1)//config["gradient_accumulation"]
    scheduler=get_linear_schedule_with_warmup(optimizer,int(updates*config["epochs"]*config["warmup_ratio"]),updates*config["epochs"])
    loss_fn=nn.CrossEntropyLoss(weight=torch.tensor([1.,config["class_weight"]],device=device))
    teacher_probability=None
    if args.teacher_probabilities:
        teacher_frame=pd.read_csv(args.teacher_probabilities).sort_values("row_id")
        expected=np.arange(len(raw_train))
        if not np.array_equal(teacher_frame["row_id"].to_numpy(),expected):
            raise ValueError("Teacher probabilities must cover every training row exactly once")
        teacher_probability=torch.tensor(
            teacher_frame["teacher_probability"].to_numpy(),dtype=torch.float32,device=device)
    history=[];best=-1.;best_record=None
    selection_metric=config.get("selection_metric","f1")
    optimizer.zero_grad(set_to_none=True)
    trainable_names={name for name,parameter in model.named_parameters() if parameter.requires_grad}
    boundary_memory_entries = (refresh_boundary_memory(
        model,boundary_memory_loader,device) if model.relative_boundary else 0)
    for epoch in range(1,config["epochs"]+1):
        if hasattr(model,"set_training_progress"):model.set_training_progress(epoch/config["epochs"])
        model.train();running=0.;running_kl=0.;running_router_loss=0.
        running_expert_loss=0.;started=time.time()
        progress=tqdm(train_loader,desc=f"LMR seed={args.seed} epoch={epoch}/{config['epochs']}")
        for step,batch in enumerate(progress,1):
            batch=move(batch,device)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                output=model(batch)
                if config.get("rdrop_weight",0.0)>0:
                    second_output=model(batch)
                    ce=0.5*(loss_fn(output["logits"],batch["labels"])+
                            loss_fn(second_output["logits"],batch["labels"]))
                    first_expert_ce=0.5*(
                        loss_fn(output["base_expert_logits"],batch["labels"])+
                        loss_fn(output["boundary_expert_logits"],batch["labels"]))
                    second_expert_ce=0.5*(
                        loss_fn(second_output["base_expert_logits"],batch["labels"])+
                        loss_fn(second_output["boundary_expert_logits"],batch["labels"]))
                    expert_ce=0.5*(first_expert_ce+second_expert_ce)
                    auxiliary=0.5*(output["auxiliary_loss"]+second_output["auxiliary_loss"])
                    router_loss=0.5*(output["boundary_router_loss"]+
                                     second_output["boundary_router_loss"])
                    kl=symmetric_kl(output["logits"],second_output["logits"])
                else:
                    ce=loss_fn(output["logits"],batch["labels"])
                    expert_ce=0.5*(
                        loss_fn(output["base_expert_logits"],batch["labels"])+
                        loss_fn(output["boundary_expert_logits"],batch["labels"]))
                    auxiliary=output["auxiliary_loss"]
                    router_loss=output["boundary_router_loss"]
                    kl=output["logits"].new_zeros(())
                distill=output["logits"].new_zeros(())
                if teacher_probability is not None and args.distill_weight>0:
                    temperature=float(args.distill_temperature)
                    soft=teacher_probability[batch["row_ids"]].clamp(1e-5,1-1e-5)
                    teacher_logit=torch.logit(soft)/temperature
                    student_logit=(output["logits"][:,1]-output["logits"][:,0])/temperature
                    first_kd=F.binary_cross_entropy_with_logits(
                        student_logit,torch.sigmoid(teacher_logit))*(temperature**2)
                    if config.get("rdrop_weight",0.0)>0:
                        second_logit=(second_output["logits"][:,1]-second_output["logits"][:,0])/temperature
                        second_kd=F.binary_cross_entropy_with_logits(
                            second_logit,torch.sigmoid(teacher_logit))*(temperature**2)
                        distill=.5*(first_kd+second_kd)
                    else:
                        distill=first_kd
                loss=(ce+config.get("expert_supervision_weight",0.0)*expert_ce+
                      config["auxiliary_weight"]*auxiliary+
                      config.get("rdrop_weight",0.0)*kl+
                      args.distill_weight*distill)/config["gradient_accumulation"]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch={epoch} step={step}")
            loss.backward();running+=float(loss.detach())*config["gradient_accumulation"]
            running_kl+=float(kl.detach())
            running_router_loss+=float(router_loss.detach())
            running_expert_loss+=float(expert_ce.detach())
            if step%config["gradient_accumulation"]==0 or step==len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True)
            progress.set_postfix(loss=f"{running/step:.4f}")
        if model.relative_boundary:
            boundary_memory_entries=refresh_boundary_memory(
                model,boundary_memory_loader,device)
        metrics,frame=evaluate(model,valid_loader,device);record={"epoch":epoch,"train_loss":running/len(train_loader),
            "train_symmetric_kl":running_kl/len(train_loader),
            "train_boundary_router_loss":running_router_loss/len(train_loader),
            "train_expert_loss":running_expert_loss/len(train_loader),
            "seconds":time.time()-started,**metrics};history.append(record);print(json.dumps(record),flush=True)
        if selection_metric not in metrics:
            raise KeyError(f"Unknown selection_metric={selection_metric}")
        if metrics[selection_metric]>best:
            best=metrics[selection_metric]
            best_record=record
            state=model.state_dict()
            if args.compact_checkpoint:
                state={name:value.detach().cpu().clone() for name,value in state.items()
                       if name in trainable_names}
            torch.save({"model":state,"epoch":epoch,"metrics":metrics,
                        "compact":args.compact_checkpoint},args.output/"checkpoints"/"best.pt")
            frame.to_csv(args.output/"validation_predictions.csv",index=False)
        (args.output/"history.json").write_text(json.dumps(history,indent=2),encoding="utf-8")
    test_metrics=None
    if args.evaluate_test:
        raw_test=load_examples(data_dir/"test.csv")
        test_items=memory.materialize(raw_test,config["prototype_count"],training=False)
        if args.knowledge_mode=="shuffled":test_items=shuffled_prototypes(test_items,args.seed+5000)
        test_loader=DataLoader(PrototypeEvidenceDataset(test_items),batch_size=config["eval_batch_size"],shuffle=False,
            collate_fn=collator,num_workers=config["num_workers"],pin_memory=True)
        best_checkpoint=torch.load(args.output/"checkpoints"/"best.pt",map_location="cpu",weights_only=True)
        model.load_state_dict(best_checkpoint["model"],strict=not best_checkpoint.get("compact",False))
        if model.relative_boundary:
            boundary_memory_entries=refresh_boundary_memory(
                model,boundary_memory_loader,device)
        test_metrics,test_frame=evaluate(model,test_loader,device)
        test_frame.to_csv(args.output/"test_predictions.csv",index=False)
    summary={"seed":args.seed,"validation_only":not args.evaluate_test,
        "selection_metric":selection_metric,"best_selection_score":best,
        "best_f1":float(best_record["f1"]),"best_epoch":int(best_record["epoch"]),
        "parameters":sum(p.numel() for p in model.parameters()),"trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),
        "peak_gpu_memory_gb":torch.cuda.max_memory_allocated()/1024**3,
        "boundary_memory_entries":boundary_memory_entries,
        "initial_checkpoint":str(args.initial_checkpoint.resolve()) if args.initial_checkpoint else None,
        "initial_encoder_checkpoint":str(args.initial_encoder_checkpoint.resolve()) if args.initial_encoder_checkpoint else None,
        "compact_checkpoint":args.compact_checkpoint,
        "test":test_metrics,"config":config}
    (args.output/"metrics.json").write_text(json.dumps(summary,indent=2),encoding="utf-8");print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
