import torch
import wandb
import pprint
import vitdet
import lightning.pytorch as pl
import torch.nn.functional as F

from torch import nn
from functools import partial
from navie_decoder import NaiveDecoder
from torchmetrics.classification import (
    BinaryJaccardIndex,
    F1Score,
    BinaryPrecisionRecallCurve,
)
from segment_anything.modeling.mask_decoder import MaskDecoder
from segment_anything.modeling.prompt_encoder import PromptEncoder
from segment_anything.modeling.transformer import TwoWayTransformer
from segment_anything.modeling.image_encoder import ImageEncoderViT


class SAMRoadplus(pl.LightningModule):
    """This is the RelationFormer module that performs object detection"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        assert config.SAM_VERSION in {"vit_b", "vit_l", "vit_h"}
        if config.SAM_VERSION == "vit_b":
            ### SAM config (B)
            encoder_embed_dim = 768
            encoder_depth = 12
            encoder_num_heads = 12
            encoder_global_attn_indexes = [2, 5, 8, 11]
        elif config.SAM_VERSION == "vit_l":
            ### SAM config (L)
            encoder_embed_dim = 1024
            encoder_depth = 24
            encoder_num_heads = 16
            encoder_global_attn_indexes = [5, 11, 17, 23]
        elif config.SAM_VERSION == "vit_h":
            ### SAM config (H)
            encoder_embed_dim = 1280
            encoder_depth = 32
            encoder_num_heads = 16
            encoder_global_attn_indexes = [7, 15, 23, 31]

        prompt_embed_dim = 256
        # SAM default is 1024
        image_size = config.PATCH_SIZE
        self.image_size = image_size
        vit_patch_size = 16

        image_embedding_size = image_size // vit_patch_size
        encoder_output_dim = prompt_embed_dim

        self.register_buffer(  # SAM 논문에서 ImageNet 기반 ViT 모델에 맞춰서 사용된 값
            "pixel_mean", torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1), False
        )
        self.register_buffer(  # 학습 시 gradient 계산 X, GPU/CPU 자동 이동
            "pixel_std", torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1), False
        )
        if self.config.NO_SAM:
            ### im1k + mae pre-trained vitb
            self.image_encoder = vitdet.VITBEncoder(
                image_size=image_size, output_feature_dim=prompt_embed_dim
            )
            self.matched_param_names = self.image_encoder.matched_param_names
        else:
            ### SAM vitb
            self.image_encoder = ImageEncoderViT(
                depth=encoder_depth,
                embed_dim=encoder_embed_dim,
                img_size=image_size,
                mlp_ratio=4,
                norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                num_heads=encoder_num_heads,
                patch_size=vit_patch_size,
                qkv_bias=True,
                use_rel_pos=True,
                global_attn_indexes=encoder_global_attn_indexes,
                window_size=14,
                out_chans=prompt_embed_dim,
            )
        if self.config.USE_SAM_DECODER:
            # SAM DECODER
            # Not used, just produce null embeddings
            self.prompt_encoder = PromptEncoder(
                embed_dim=prompt_embed_dim,
                image_embedding_size=(image_embedding_size, image_embedding_size),
                input_image_size=(image_size, image_size),
                mask_in_chans=16,
            )
            for param in self.prompt_encoder.parameters():
                param.requires_grad = False
            self.mask_decoder = MaskDecoder(
                num_multimask_outputs=3,  # keypoint, road
                transformer=TwoWayTransformer(
                    depth=2,
                    embedding_dim=prompt_embed_dim,
                    mlp_dim=2048,
                    num_heads=8,
                ),
                transformer_dim=prompt_embed_dim,
                iou_head_depth=3,
                iou_head_hidden_dim=256,
            )
        else:
            #### Naive decoder
            out_channels = 2 if not self.config.ALEATORIC else 3
            self.decoder_list = nn.ModuleList(
                [
                    NaiveDecoder(out_channels=out_channels)
                    for _ in range(self.config.DECODER_COUNT)
                ]
            )

        reduction = "mean" if not self.config.ALEATORIC else "none"
        self.mask_criterion = torch.nn.BCEWithLogitsLoss(reduction=reduction)

        #### Metrics
        self.keypoint_iou = BinaryJaccardIndex(threshold=0.5)
        self.road_iou = BinaryJaccardIndex(threshold=0.5)
        self.topo_f1 = F1Score(task="binary", threshold=0.5, ignore_index=-1)

        # testing only, not used in training
        self.keypoint_pr_curve = BinaryPrecisionRecallCurve(ignore_index=-1)
        self.road_pr_curve = BinaryPrecisionRecallCurve(ignore_index=-1)
        self.topo_pr_curve = BinaryPrecisionRecallCurve(ignore_index=-1)

        with open(config.SAM_CKPT_PATH, "rb") as f:
            ckpt_state_dict = torch.load(f)

            ## Resize pos embeddings, if needed
            if image_size != 1024:
                new_state_dict = self.resize_sam_pos_embed(
                    ckpt_state_dict,
                    image_size,
                    vit_patch_size,
                    encoder_global_attn_indexes,
                )
                ckpt_state_dict = new_state_dict

            matched_names = []
            mismatch_names = []
            state_dict_to_load = {}
            for k, v in self.named_parameters():
                if k in ckpt_state_dict and v.shape == ckpt_state_dict[k].shape:
                    matched_names.append(k)
                    state_dict_to_load[k] = ckpt_state_dict[k]
                else:
                    mismatch_names.append(k)
            print("###### Matched params ######")
            pprint.pprint(matched_names)
            print("###### Mismatched params ######")
            pprint.pprint(mismatch_names)

            self.matched_param_names = set(matched_names)
            self.load_state_dict(state_dict_to_load, strict=False)

    def resize_sam_pos_embed(
        self, state_dict, image_size, vit_patch_size, encoder_global_attn_indexes
    ):
        new_state_dict = {k: v for k, v in state_dict.items()}
        pos_embed = new_state_dict["image_encoder.pos_embed"]
        token_size = int(image_size // vit_patch_size)
        if pos_embed.shape[1] != token_size:
            # Copied from SAMed
            # resize pos embedding, which may sacrifice the performance, but I have no better idea
            pos_embed = pos_embed.permute(0, 3, 1, 2)  # [b, c, h, w]
            pos_embed = F.interpolate(
                pos_embed,
                (token_size, token_size),
                mode="bilinear",
                align_corners=False,
            )
            pos_embed = pos_embed.permute(0, 2, 3, 1)  # [b, h, w, c]
            new_state_dict["image_encoder.pos_embed"] = pos_embed
            rel_pos_keys = [k for k in state_dict.keys() if "rel_pos" in k]
            global_rel_pos_keys = [
                k
                for k in rel_pos_keys
                if any([str(i) in k for i in encoder_global_attn_indexes])
            ]
            for k in global_rel_pos_keys:
                rel_pos_params = new_state_dict[k]
                h, w = rel_pos_params.shape
                rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)
                rel_pos_params = F.interpolate(
                    rel_pos_params,  # [1, 1, h, w]
                    (token_size * 2 - 1, w),
                    mode="bilinear",
                    align_corners=False,
                )
                new_state_dict[k] = rel_pos_params[0, 0, ...]
        return new_state_dict

    def forward(self, rgb, graph_points, pairs, valid):
        # rgb: [B, H, W, C]
        # graph_points: [B, N_points, 2]
        # pairs: [B, N_samples, N_pairs, 2]
        # valid: [B, N_samples, N_pairs]

        x = rgb.permute(0, 3, 1, 2)  # [B, C, H, W]
        x = (x - self.pixel_mean) / self.pixel_std

        # [B, |256, image_size / vit_patch_size, image_size / vit_patch_size]
        image_embeddings = self.image_encoder(x)

        # mask_logits, mask_scores: [B, 2, H, W]
        if self.config.USE_SAM_DECODER:
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None, boxes=None, masks=None
            )
            low_res_logits, iou_predictions = self.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
            )
            mask_logits = F.interpolate(
                low_res_logits,
                (self.image_encoder.img_size, self.image_encoder.img_size),
                mode="bilinear",
                align_corners=False,
            )
            mask_scores = torch.sigmoid(mask_logits)
        else:
            mask_logits_list = [
                decoder(image_embeddings) for decoder in self.decoder_list
            ]
            mask_scores_list = [torch.sigmoid(logits) for logits in mask_logits_list]

        # image embedding + mask를 통해 graph points 주변 feature 샘플링
        # target_feature, target_point, source_feature
        # [B, 2, H, W]

        mask_logits_list = list(map(lambda x: x.permute(0, 2, 3, 1), mask_logits_list))
        mask_scores_list = list(map(lambda x: x.permute(0, 2, 3, 1), mask_scores_list))
        return mask_logits_list, mask_scores_list

    def training_step(self, batch, batch_idx):
        # masks: [B, H, W]
        rgb, keypoint_mask, road_mask = (
            batch["rgb"],
            batch["keypoint_mask"],
            batch["road_mask"],
        )
        graph_points, pairs, valid = (
            batch["graph_points"],
            batch["pairs"],
            batch["valid"],
        )
        # [B, H, W, 2]
        mask_logits_list, _ = self(rgb, graph_points, pairs, valid)
        gt_masks = torch.stack([keypoint_mask, road_mask], dim=3)

        mask_loss_list = []
        for i, logits in enumerate(mask_logits_list):
            if self.config.ALEATORIC:  # 3채널은 분산정보
                new_logits = logits[..., :2]
                uncertainty = logits[..., 2].unsqueeze(-1)

                tmp_mask_loss = self.mask_criterion(new_logits, gt_masks)
                tmp_loss = tmp_mask_loss * torch.exp(-uncertainty) + uncertainty
                mask_loss = tmp_loss.mean()
            else:
                mask_loss = self.mask_criterion(logits, gt_masks)

            if torch.any(torch.isnan(mask_loss)):
                print(f"NaN detected in mask_loss_{i}. Using default loss value.")
                mask_loss = torch.tensor(0.0, device=mask_loss.device)

            self.log(
                f"train_mask_loss_{i}",
                mask_loss,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )

            mask_loss_list.append(mask_loss)

        total_mask_loss = torch.stack(mask_loss_list).mean()
        total_loss = total_mask_loss

        if self.config.COMBINE_LOSS:  # True면 우선 l2 loss사용
            l2_loss_list = []
            for i in range(self.config.DECODER_COUNT):  # Decoder 순으로 pair 구성
                for j in range(i + 1, self.config.DECODER_COUNT):  # normalization 적용
                    l2_loss = torch.mean(
                        (mask_logits_list[i] - mask_logits_list[j]) ** 2
                    )
                    self.log(
                        f"train_l2_loss_{i}_{j}",
                        l2_loss,
                        on_step=True,
                        on_epoch=False,
                        prog_bar=True,
                    )
                    l2_loss_list.append(l2_loss)

            total_l2_loss = torch.stack(l2_loss_list).mean()
            self.log(
                "train_total_l2_loss",
                total_l2_loss,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )

            total_loss += self.config.L2_LOSS_WEIGHT * total_l2_loss

        self.log(
            "train_total_mask_loss",
            total_mask_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )

        self.log(
            "train_total__loss",
            total_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )

        return total_loss

    def validation_step(self, batch, batch_idx):
        # masks: [B, H, W]
        rgb, keypoint_mask, road_mask = (
            batch["rgb"],
            batch["keypoint_mask"],
            batch["road_mask"],
        )
        graph_points, pairs, valid = (
            batch["graph_points"],
            batch["pairs"],
            batch["valid"],
        )
        # masks: [B, H, W, 2] topo: [B, N_samples, N_pairs, 1]
        mask_logits_list, mask_scores_list = self(rgb, graph_points, pairs, valid)
        gt_masks = torch.stack([keypoint_mask, road_mask], dim=3)

        mask_loss_list = []
        for i, logits in enumerate(mask_logits_list):
            if self.config.ALEATORIC:
                new_logits = logits[..., :2]
                uncertainty = logits[..., 2].unsqueeze(-1)

                tmp_mask_loss = self.mask_criterion(new_logits, gt_masks)
                tmp_loss = tmp_mask_loss * torch.exp(-uncertainty) + uncertainty
                mask_loss = tmp_loss.mean()
            else:
                mask_loss = self.mask_criterion(logits, gt_masks)

            if torch.any(torch.isnan(mask_loss)):
                print(f"NaN detected in mask_loss_{i}. Using default loss value.")
                mask_loss = torch.tensor(0.0, device=mask_loss.device)

            self.log(
                f"val_mask_loss_{i}",
                mask_loss,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )

            mask_loss_list.append(mask_loss)

        total_mask_loss = torch.stack(mask_loss_list).mean()
        total_loss = total_mask_loss

        if self.config.COMBINE_LOSS:  # True면 우선 l2 loss사용
            l2_loss_list = []
            for i in range(self.config.DECODER_COUNT):  # Decoder 순으로 pair 구성
                for j in range(i + 1, self.config.DECODER_COUNT):
                    if self.config.LOGITS_NORMALIZATION:  # normalization 적용
                        f1 = torch.tanh(mask_logits_list[i])
                        f2 = torch.tanh(mask_logits_list[j])
                        l2_loss = ((f1 - f2) ** 2).mean()
                    else:
                        l2_loss = torch.mean(
                            (mask_logits_list[i] - mask_logits_list[j]) ** 2
                        )
                    self.log(
                        f"val_l2_loss_{i}_{j}",
                        l2_loss,
                        on_step=True,
                        on_epoch=False,
                        prog_bar=True,
                    )
                    l2_loss_list.append(l2_loss)

            total_l2_loss = torch.stack(l2_loss_list).mean()
            self.log(
                "val_total_l2_loss",
                total_l2_loss,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )

            total_loss += self.config.L2_LOSS_WEIGHT * total_l2_loss

        self.log(
            "val_total_mask_loss",
            total_mask_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )

        self.log(
            "val_total__loss",
            total_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )

        # Log images
        if batch_idx == 0:
            max_viz_num = 4
            viz_rgb = rgb[:max_viz_num, :, :]

            viz_pred_road_list = [
                mask_score[:max_viz_num, :, :, 1] for mask_score in mask_scores_list
            ]
            viz_pred_keypoint_list = [
                mask_score[:max_viz_num, :, :, 0] for mask_score in mask_scores_list
            ]

            viz_gt_keypoint = keypoint_mask[:max_viz_num, ...]
            viz_gt_road = road_mask[:max_viz_num, ...]

            pred_road_names = [
                f"pred_road{i}" for i in range(1, len(mask_scores_list) + 1)
            ]
            pred_keypoint_names = [
                f"pred_keypoint{i}" for i in range(1, len(mask_scores_list) + 1)
            ]

            columns = [
                "rgb",
                "gt_keypoint",
                "gt_road",
                *pred_road_names,
                *pred_keypoint_names,
            ]
            data = [
                [wandb.Image(x.cpu().numpy()) for x in row]
                for row in list(
                    zip(
                        viz_rgb,
                        viz_gt_keypoint,
                        viz_gt_road,
                        *viz_pred_road_list,
                        *viz_pred_keypoint_list,
                    )
                )
            ]
            self.logger.log_table(key="viz_table", columns=columns, data=data)

    def on_validation_epoch_end(self):
        return
        # keypoint_iou = self.keypoint_iou.compute()
        # road_iou = self.road_iou.compute()
        # topo_f1 = self.topo_f1.compute()
        # self.log("keypoint_iou", keypoint_iou)
        # self.log("road_iou", road_iou)
        # self.log("topo_f1", topo_f1)
        # self.keypoint_iou.reset()
        # self.road_iou.reset()
        # self.topo_f1.reset()

    def test_step(self, batch, batch_idx):
        # masks: [B, H, W]
        rgb, keypoint_mask, road_mask = (
            batch["rgb"],
            batch["keypoint_mask"],
            batch["road_mask"],
        )
        graph_points, pairs, valid = (
            batch["graph_points"],
            batch["pairs"],
            batch["valid"],
        )
        # masks: [B, H, W, 2] topo: [B, N_samples, N_pairs, 1]
        mask_logits, mask_scores, topo_logits, topo_scores = self(
            rgb, graph_points, pairs, valid
        )
        topo_gt, topo_loss_mask = batch["connected"].to(torch.int32), valid.to(
            torch.float32
        )
        self.keypoint_pr_curve.update(
            mask_scores[..., 0], keypoint_mask.to(torch.int32)
        )
        self.road_pr_curve.update(mask_scores[..., 1], road_mask.to(torch.int32))
        valid = valid.to(torch.int32)
        topo_gt = (1 - valid) * -1 + valid * topo_gt
        self.topo_pr_curve.update(topo_scores, topo_gt.unsqueeze(-1).to(torch.int32))

    def on_test_end(self):
        def find_best_threshold(pr_curve_metric, category):
            print(f"======= {category} ======")
            precision, recall, thresholds = pr_curve_metric.compute()
            f1_scores = 2 * (precision * recall) / (precision + recall)
            best_threshold_index = torch.argmax(f1_scores)
            best_threshold = thresholds[best_threshold_index]
            best_precision = precision[best_threshold_index]
            best_recall = recall[best_threshold_index]
            best_f1 = f1_scores[best_threshold_index]
            print(
                f"Best threshold {best_threshold}, P={best_precision} R={best_recall} F1={best_f1}"
            )

        print("======= Finding best thresholds ======")
        find_best_threshold(self.keypoint_pr_curve, "keypoint")
        find_best_threshold(self.road_pr_curve, "road")
        find_best_threshold(self.topo_pr_curve, "topo")

    def configure_optimizers(self):
        param_dicts = []
        if not self.config.FREEZE_ENCODER and not self.config.ENCODER_LORA:
            encoder_params = {
                "params": [
                    p
                    for k, p in self.image_encoder.named_parameters()
                    if "image_encoder." + k in self.matched_param_names
                ],
                "lr": self.config.BASE_LR * self.config.ENCODER_LR_FACTOR,
            }
            param_dicts.append(encoder_params)
        if self.config.ENCODER_LORA:
            # LoRA params only
            encoder_params = {
                "params": [
                    p
                    for k, p in self.image_encoder.named_parameters()
                    if "qkv.linear_" in k
                ],
                "lr": self.config.BASE_LR,
            }
            param_dicts.append(encoder_params)

        if self.config.USE_SAM_DECODER:
            matched_decoder_params = {
                "params": [
                    p
                    for k, p in self.mask_decoder.named_parameters()
                    if "mask_decoder." + k in self.matched_param_names
                ],
                "lr": self.config.BASE_LR * 0.1,
            }
            fresh_decoder_params = {
                "params": [
                    p
                    for k, p in self.mask_decoder.named_parameters()
                    if "mask_decoder." + k not in self.matched_param_names
                ],
                "lr": self.config.BASE_LR,
            }
            decoder_params = [matched_decoder_params, fresh_decoder_params]
        else:
            decoder_params = [
                {
                    "params": [
                        p for decoder in self.decoder_list for p in decoder.parameters()
                    ],
                    "lr": self.config.BASE_LR,
                }
            ]
        param_dicts += decoder_params

        for i, param_dict in enumerate(param_dicts):
            param_num = sum([int(p.numel()) for p in param_dict["params"]])
            print(f"optim param dict {i} params num: {param_num}")

        optimizer = torch.optim.AdamW(
            param_dicts, lr=self.config.BASE_LR, betas=(0.9, 0.999), weight_decay=0.1
        )
        step_lr = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[
                9,
            ],
            gamma=0.1,
        )
        return {"optimizer": optimizer, "lr_scheduler": step_lr}
