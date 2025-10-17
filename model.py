import math
import torch
import wandb
import pprint
import vitdet
import numpy as np
import torchvision
import torch.nn.functional as F
import lightning.pytorch as pl

from torch import nn
from functools import partial
from torchmetrics.classification import (
    BinaryJaccardIndex,
    F1Score,
    BinaryPrecisionRecallCurve,
)
from segment_anything.modeling.common import LayerNorm2d
from segment_anything.modeling.transformer import TwoWayTransformer
from segment_anything.modeling.mask_decoder import MaskDecoder
from segment_anything.modeling.image_encoder import ImageEncoderViT
from segment_anything.modeling.prompt_encoder import PromptEncoder

# import copy
# import matplotlib.pyplot as plt
# from torchmetrics.classification import MulticlassJaccardIndex


def find_highest_mask_point(x, y, mask, device="cuda"):
    """
    주어진 좌표 (x, y) 주변에서 mask(마스크) 값이 가장 높은 지점을 찾는 함수

    Args:
        x (int or float): 기준 x 좌표
        y (int or float): 기준 y 좌표
        mask (Tensor): [C, H, W] 차원의 마스크 텐서
        device (str): torch device

    Returns:
        (int, int): 가장 높은 mask 값을 갖는 좌표 (x_final, y_final)
    """
    # mask shape을 가져옴
    H, W, D = mask.shape

    # x, y 좌표 제한 (이미지 범위 밖으로 나가지 않게 clamp)
    x = torch.clamp(x, 0, W)
    y = torch.clamp(y, 0, D)
    x = int(x)
    y = int(y)

    radius = torch.tensor(2)  # 반경 2 픽셀 내에서 탐색

    # 좌표 범위 설정 (반경 내에서만 탐색)
    x_min = max(0, x - radius)
    x_max = min(W, x + radius)
    y_min = max(0, y - radius)
    y_max = min(D, y + radius)

    # 탐색 범위 내 mask 영역 추출, device에 할당
    mask_region = mask[:, x_min:x_max, y_min:y_max].to(device)

    # x, y 좌표 meshgrid 생성
    x_coords = (
        torch.arange(x_min, x_max, device=device)
        .view(-1, 1)
        .expand(x_max - x_min, y_max - y_min)
    )
    y_coords = (
        torch.arange(y_min, y_max, device=device)
        .view(1, -1)
        .expand(x_max - x_min, y_max - y_min)
    )

    # 기준점에서의 거리 계산 (유클리드 거리)
    distances = torch.sqrt((x_coords - x) ** 2 + (y_coords - y) ** 2)

    # 지정 반경 안의 마스크만 사용
    within_radius = (distances <= radius).to(device)

    # 두 개의 채널 mask를 더한 점수 계산 (with radius 필터 적용)
    mask_scores = mask_region[1] * within_radius + mask_region[0] * within_radius

    # mask score가 존재할 때
    if mask_scores.numel() > 0:
        mask_max = torch.max(mask_scores)  # 최대값 찾기
        max_pos = torch.nonzero(mask_scores == mask_max)  # 최대값 위치 추출
        if len(max_pos) > 0:
            x_final = max_pos[0][0] + x_min  # mask_region이 일부 영역 추출이므로
            y_final = max_pos[0][1] + y_min  # x, y의 min값 더해줌
        else:
            x_final, y_final = x, y
    else:
        x_final, y_final = x, y  # 없으면 제자리

    return x_final, y_final


def extract_point(x1, y1, x2, y2, image, num_points):
    """
    두 점 (x1, y1), (x2, y2) 사이를 num_points 개수만큼 균등하게 샘플링하여
    이미지 내 좌표를 추출하는 함수

    Args:
        x1, y1: 시작점 좌표 (Tensor)
        x2, y2: 끝점 좌표 (Tensor)
        image: 이미지 텐서, 최소 2D 이상 (ex. [B, D, H, W] 또는 [C, H, W])
        num_points: 추출할 포인트 개수

    Returns:
        x_final, y_final: 각 포인트의 x, y 좌표 Tensor
    """
    H, W = image.shape[-2:]  # 이미지 높이, 너비 추출

    # 0~1 구간에서 num_points 개수만큼 균등 분포한 값을 만든다 (sampling factor로 사용)
    x_values = (
        torch.linspace(0, 1, steps=num_points)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(image.device)
    )
    y_values = (
        torch.linspace(0, 1, steps=num_points)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(image.device)
    )

    # 각 좌표(x1, y1)~(x2, y2)에 대해 구간 선형 보간
    x_interp = x1.unsqueeze(-1) + (x2 - x1).unsqueeze(-1) * x_values
    y_interp = y1.unsqueeze(-1) + (y2 - y1).unsqueeze(-1) * y_values

    # 이미지 범위 밖으로 안나가도록 clamp 및 정수형 변환
    x_interp = torch.clamp(x_interp.long(), min=0, max=W - 1)
    y_interp = torch.clamp(y_interp.long(), min=0, max=H - 1)

    # 좀 더 다양한 샘플링을 위해 x, y 좌표를 +1 한 경우도 만듦
    x_plus_1 = torch.clamp(x_interp + 1, max=W - 1)
    y_plus_1 = torch.clamp(y_interp + 1, max=H - 1)

    # 세 가지 좌표(x, y), (x, y + 1), (x + 1, y) 조합으로 feature 추출을 위한 조합 반환
    x_final = torch.cat([x_interp, x_interp, x_plus_1], dim=-1)
    y_final = torch.cat([y_interp, y_plus_1, y_interp], dim=-1)

    return (x_final, y_final)  # 샘플링 좌표 반환


def extendline(points1, points2, image):
    """
    두 점 집합(points1, points2)에 대해 각 쌍을 양 끝으로 연장한 선분 위의 픽셀 값을 샘플링
    배치 단위로 동작하며, 선분의 내외부 여러 지점들의 특징(feature)을 추출

    Args:
        points1 (Tensor): [B, N, 2] 시작 점 좌표 집합, (x, y) 형식
        points2 (Tensor): [B, N, 2] 끝 점 좌표 집합, (x, y) 형식
        image (Tensor): [B, ... H, W], 최소 2차원 이상, 입력 이미지(또는 feature map)

    Returns:
        features (Tensor): [B, N, 전체 샘플 수, ...] 연장선 상 샘플 좌표의 이미지값 or 특징 벡터

    동작 과정:
    1. 두 점 A, B 사이의 선을 양쪽으로 extend_length(8픽셀)만큼 연장
    2. 연장된 두 점(extended_A, extended_B) - 원래 점(A, B) - 를 세 구간으로 나눠 각각 추출
       · [extended_A ~ A], [A ~ B], [extended_B ~ B]
    3. 추출된 좌표를 이용해 이미지(feature map)로부터 값 추출, 전부 이어붙여 반환
    """
    B, N, _ = points1.shape  # B: 배치 크기, N: 쌍 개수
    H, W = image.shape[-2:]  # 이미지 높이, 너비
    height, width = H, W
    extend_length = 8  # 연장 픽셀 수

    batch_A = points1
    batch_B = points2

    # 방향 벡터 (B, N, 2)
    directions = batch_B - batch_A

    # 각 선분의 길이(norm) 계산, 0은 아주 작은 수로 대체
    lengths = torch.norm(directions, dim=2, keepdim=True)
    lengths = lengths.masked_fill(lengths == 0, 1e-8)
    directions_norm = directions / lengths  # 정규화된 방향 벡터

    # 선분 양 끝을 연장 (양쪽 각각 extend_length만큼)
    extended_A = batch_A - directions_norm * extend_length
    extended_B = batch_B + directions_norm * extend_length

    # 실수 좌표를 정수로 반올림
    extended_A = torch.round(extended_A).long()
    extended_B = torch.round(extended_B).long()

    # 이미지 범위 내로 clamp
    extended_A[:, 0] = extended_A[:, 0].clamp(0, width - 1)
    extended_A[:, 1] = extended_A[:, 1].clamp(0, height - 1)
    extended_B[:, 0] = extended_B[:, 0].clamp(0, width - 1)
    extended_B[:, 1] = extended_B[:, 1].clamp(0, height - 1)

    # 연장 좌표와 원본 점에서 각 x, y 분리
    extend_x1, extend_y1 = extended_A[..., 0], extended_A[..., 1]
    extend_x2, extend_y2 = extended_B[..., 0], extended_B[..., 1]

    x1, y1 = points1[..., 0], points1[..., 1]
    x2, y2 = points2[..., 0], points2[..., 1]

    # 1. 연장구간(extended_A ~ A)에서 num_points=15개 균등샘플 추출
    x_final_1, y_final_1 = extract_point(
        extend_x1, extend_y1, x1, y1, image, num_points=15
    )
    # 2. 두 점(A ~ B) 사이에서 num_points=20개 균등샘플 추출
    x_final, y_final = extract_point(x1, y1, x2, y2, image, num_points=20)
    # 3. 연장구간(extended_B ~ B)에서 num_points=15개 균등샘플 추출
    x_final_2, y_final_2 = extract_point(
        extend_x2, extend_y2, x2, y2, image, num_points=15
    )

    # np.arange(B): 배치 인덱스 용, 좌표대로 이미지 값 추출
    features1 = image[np.arange(B)[:, None, None], x_final_1, y_final_1]  # 연장구간1
    features = image[np.arange(B)[:, None, None], x_final, y_final]  # 구간(AB)
    features2 = image[np.arange(B)[:, None, None], x_final_2, y_final_2]  # 연장구간2

    # 세 가지 구간 샘플링 결과를 (feature/channel 차원에 대해) 2번 axis로 이어붙임
    # 좌표가 아닌 해당 좌표값에 맞는 point들을 연결하기 때문에 dim=2
    features = torch.concat([features1, features, features2], dim=2)

    return features


class BilinearSampler(nn.Module):
    def __init__(self, config):
        super(BilinearSampler, self).__init__()
        self.config = config

    def forward(self, feature_maps, sample_points, mask_scores):
        """
        Args:
            feature_maps (Tensor): The input feature tensor of shape [B, D, H, W].
            sample_points (Tensor): The 2D sample points of shape [B, N_points, 2],
                                    each point in the range [-1, 1], format (x, y).
        Returns:
            Tensor: Sampled feature vectors of shape [B, N_points, D].
        """
        B, D, H, W = feature_maps.shape  # [16, 256, H, W]
        batch_size, N_points, _ = sample_points.shape

        target_new_points = torch.zeros_like(sample_points).cuda()
        for batch_index in range(batch_size):
            for point_index in range(N_points):
                x, y = sample_points[batch_index, point_index]
                if (x.item(), y.item()) == (0, 0):
                    target_new_points[batch_index, point_index] = torch.tensor([x, y])
                else:
                    current_mask = mask_scores[batch_index]  # torch.Size([2, H, W]
                    x_new, y_new = find_highest_mask_point(x, y, current_mask)
                    target_new_points[batch_index, point_index] = torch.tensor(
                        [x_new, y_new], dtype=torch.float32
                    )
        point = target_new_points

        # Target Node Feature
        # [-1, 1] 정규화
        target_new_points = (target_new_points / self.config.PATCH_SIZE) * 2.0 - 1.0
        target_new_points = target_new_points.unsqueeze(2)  # [B, N_points, 1, 2]
        sampled_features = F.grid_sample(  # output -> [B, D, N_points, 1]
            feature_maps, target_new_points, mode="bilinear", align_corners=False
        )
        sampled_features_target = sampled_features.squeeze(dim=-1).permute(
            0, 2, 1
        )  # [B, N_points, D]

        # Source Node Feature
        sample_points = (sample_points / self.config.PATCH_SIZE) * 2.0 - 1.0
        sample_points = sample_points.unsqueeze(2)
        sampled_features_o = F.grid_sample(
            feature_maps, sample_points, mode="bilinear", align_corners=False
        )
        sampled_features_source = sampled_features_o.squeeze(dim=-1).permute(0, 2, 1)

        return sampled_features_target, point, sampled_features_source


class TopoNet(nn.Module):
    def __init__(self, config, feature_dim):  # 256
        super(TopoNet, self).__init__()
        self.config = config
        self.hidden_dim = 128
        self.heads = 4
        self.num_attn_layers = 3
        self.feature_proj = nn.Linear(feature_dim, self.hidden_dim)  # in 256, out 128

        # 152는 extendline에서 n=15, m=20으로 수행 -> extract_point에서 3배수
        # 15 * 2 * 3 + 20 * 3 = 150, offset_x 값 2
        self.pair_proj = nn.Linear(2 * self.hidden_dim + 152, self.hidden_dim)
        # Create Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.heads,
            dim_feedforward=self.hidden_dim,
            dropout=0.1,
            activation="relu",
            batch_first=True,  # Input format is [batch size, sequence length, features]
        )

        # Stack the Transformer Encoder Layers
        if self.config.TOPONET_VERSION != "no_transformer":
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=self.num_attn_layers
            )
        self.output_proj = nn.Linear(self.hidden_dim, 1)

    def forward(
        self,
        points,
        point_features,
        graph_points,
        point_features_o,
        pairs,
        pairs_valid,
        mask_scores,
    ):
        # points: [B, N_points, 2]
        # point_features: [B, N_points, D]
        # pairs: [B, N_samples, N_pairs, 2]
        # pairs_valid: [B, N_samples, N_pairs]
        # mask scores:[B, 2, 512, 512]
        B, _, H, W = mask_scores.shape

        # 설정 차원으로 변환 -> 이후 비교/조합에 용이
        point_features = F.relu(self.feature_proj(point_features))
        point_features_o = F.relu(self.feature_proj(point_features_o))

        # gathers pairs
        batch_size, n_samples, n_pairs, _ = pairs.shape
        pairs = pairs.view(batch_size, -1, 2)  # [B, N_samples * N_pairs, 2]
        batch_indices = (
            torch.arange(batch_size).view(-1, 1).expand(-1, n_samples * n_pairs)
        )

        # [B, N_samples * N_pairs, D]
        src_features = point_features_o[batch_indices, pairs[:, :, 0]]  # 출발 Node
        tgt_features = point_features[batch_indices, pairs[:, :, 1]]  # 도착 노드

        # [B, N_samples * N_pairs, 2] 좌표 값
        src_points = graph_points[batch_indices, pairs[:, :, 0]]
        tgt_points = points[batch_indices, pairs[:, :, 1]]

        _, N, _ = tgt_points.shape
        mask_road_dim = mask_scores[:, 1, :, :]
        line_features = extendline(src_points, tgt_points, mask_road_dim)  # [B, N, 150]
        offset_x = tgt_points - src_points  # 상대 위치 벡터

        # [B, N_samples * N_pairs, 2D + 2]
        if self.config.TOPONET_VERSION == "no_tgt_features":
            pair_features = torch.concat(
                [src_features, torch.zeros_like(tgt_features), offset_x], dim=2
            )
        if self.config.TOPONET_VERSION == "no_offset":
            pair_features = torch.concat(
                [src_features, tgt_features, torch.zeros_like(offset_x)], dim=2
            )
        else:
            # pair_features = torch.concat([line_features,offset_x], dim=2)
            pair_features = torch.concat(
                [src_features, tgt_features, line_features, offset_x], dim=2
            )

        # in [B, N, 2 * 128 + 152], out -> [B, N, 128]
        pair_features = F.relu(self.pair_proj(pair_features))

        pair_features = pair_features.view(batch_size * n_samples, n_pairs, -1)
        pairs_valid = pairs_valid.view(batch_size * n_samples, n_pairs)

        # [B * N_samples, 1]
        # 유효한 pair가 존재하는지 확인하기 위해 sum
        # 0과 비교를 통해 bool Tensor 생성, True 라면 해당 행은 모든 pair invalid
        all_invalid_pair_mask = torch.eq(torch.sum(pairs_valid, dim=-1), 0).unsqueeze(
            -1
        )

        # 모든 pair가 invalid한 행의 원소를 valid하게 변경 -> Transformer 연산에서 Nan값 방지
        pairs_valid = torch.logical_or(pairs_valid, all_invalid_pair_mask)
        padding_mask = (
            ~pairs_valid
        )  # bool 반전, True -> valid, False -> padding 즉, 무시

        if self.config.TOPONET_VERSION != "no_transformer":
            pair_features = self.transformer_encoder(
                pair_features, src_key_padding_mask=padding_mask
            )

        _, n_pairs, _ = pair_features.shape
        pair_features = pair_features.view(batch_size, n_samples, n_pairs, -1)
        # [B, N_samples, N_pairs, 1]
        logits = self.output_proj(pair_features)
        scores = torch.sigmoid(logits)
        return logits, scores


class _LoRA_qkv(nn.Module):
    """In Sam it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    """

    def __init__(
        self,
        qkv: nn.Module,
        linear_a_q: nn.Module,
        linear_b_q: nn.Module,
        linear_a_v: nn.Module,
        linear_b_v: nn.Module,
    ):
        super().__init__()
        # self.qkv = qkv
        self.weight = qkv.weight
        self.bias = qkv.bias
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x):
        # qkv = self.qkv(x)  # B,N,N,3*org_C
        qkv = F.linear(x, self.weight, self.bias)
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, :, : self.dim] += new_q
        qkv[:, :, :, -self.dim :] += new_v
        return qkv


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
            activation = nn.GELU
            self.map_decoder = nn.Sequential(
                nn.ConvTranspose2d(encoder_output_dim, 128, kernel_size=2, stride=2),
                LayerNorm2d(128),
                activation(),
                nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
                activation(),
                nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
                activation(),
                nn.ConvTranspose2d(32, 2, kernel_size=2, stride=2),
            )

        #### TOPONet
        self.bilinear_sampler = BilinearSampler(config)
        self.topo_net = TopoNet(config, 256)

        #### LORA
        if config.ENCODER_LORA:
            r = self.config.LORA_RANK
            lora_layer_selection = None
            assert r > 0
            if lora_layer_selection:
                self.lora_layer_selection = lora_layer_selection
            else:
                self.lora_layer_selection = list(
                    range(len(self.image_encoder.blocks))
                )  # Only apply lora to the image encoder by default
            # create for storage, then we can init them or load weights
            self.w_As = []  # These are linear layers
            self.w_Bs = []
            # lets freeze first
            for param in self.image_encoder.parameters():
                param.requires_grad = False
            # Here, we do the surgery
            for t_layer_i, blk in enumerate(self.image_encoder.blocks):
                # If we only want few lora layer instead of all
                if t_layer_i not in self.lora_layer_selection:
                    continue
                w_qkv_linear = blk.attn.qkv
                dim = w_qkv_linear.in_features
                w_a_linear_q = nn.Linear(dim, r, bias=False)
                w_b_linear_q = nn.Linear(r, dim, bias=False)
                w_a_linear_v = nn.Linear(dim, r, bias=False)
                w_b_linear_v = nn.Linear(r, dim, bias=False)
                self.w_As.append(w_a_linear_q)
                self.w_Bs.append(w_b_linear_q)
                self.w_As.append(w_a_linear_v)
                self.w_Bs.append(w_b_linear_v)
                blk.attn.qkv = _LoRA_qkv(
                    w_qkv_linear,
                    w_a_linear_q,
                    w_b_linear_q,
                    w_a_linear_v,
                    w_b_linear_v,
                )
            # Init LoRA params
            for w_A in self.w_As:
                nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
            for w_B in self.w_Bs:
                nn.init.zeros_(w_B.weight)

        #### Losses
        if self.config.FOCAL_LOSS:
            self.mask_criterion = partial(
                torchvision.ops.sigmoid_focal_loss, reduction="mean"
            )
        else:
            self.mask_criterion = torch.nn.BCEWithLogitsLoss()

        self.topo_criterion = torch.nn.BCEWithLogitsLoss(reduction="none")

        #### Metrics
        self.keypoint_iou = BinaryJaccardIndex(threshold=0.5)
        self.road_iou = BinaryJaccardIndex(threshold=0.5)
        self.topo_f1 = F1Score(task="binary", threshold=0.5, ignore_index=-1)

        # testing only, not used in training
        self.keypoint_pr_curve = BinaryPrecisionRecallCurve(ignore_index=-1)
        self.road_pr_curve = BinaryPrecisionRecallCurve(ignore_index=-1)
        self.topo_pr_curve = BinaryPrecisionRecallCurve(ignore_index=-1)
        if self.config.NO_SAM:
            return

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

        # [B, 256, image_size / vit_patch_size, image_size / vit_patch_size]
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
            mask_logits = self.map_decoder(image_embeddings)  # Navie Decoder
            mask_scores = torch.sigmoid(mask_logits)  # 0 -> keypoint, 1 -> road

        # image embedding + mask를 통해 graph points 주변 feature 샘플링
        # target_feature, target_point, source_feature
        point_features, newpoint, point_features_o = self.bilinear_sampler(
            image_embeddings, graph_points, mask_scores
        )  # [B, N_points, D], [B, N_points, 2], [B, N_points, D]

        # [B, N_sample, N_pair, 1]
        topo_logits, topo_scores = self.topo_net(
            newpoint,
            point_features,
            graph_points,
            point_features_o,
            pairs,
            valid,
            mask_scores,
        )
        # [B, 2, H, W]
        mask_logits = mask_logits.permute(0, 2, 3, 1)  # [B, H, W, 2]
        mask_scores = mask_scores.permute(0, 2, 3, 1)  # [B, H, W, 2]
        return mask_logits, mask_scores, topo_logits, topo_scores

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
        # [B, H, W, 3]
        mask_logits, mask_scores, topo_logits, topo_scores = self(
            rgb, graph_points, pairs, valid
        )
        gt_masks = torch.stack([keypoint_mask, road_mask], dim=3)
        mask_loss = self.mask_criterion(mask_logits, gt_masks)
        topo_gt, topo_loss_mask = batch["connected"].to(torch.int32), valid.to(
            torch.float32
        )
        # [B, N_samples, N_pairs, 1]
        topo_loss = self.topo_criterion(
            topo_logits, topo_gt.unsqueeze(-1).to(torch.float32)
        )

        topo_loss *= topo_loss_mask.unsqueeze(-1)
        topo_loss = torch.nansum(torch.nansum(topo_loss) / topo_loss_mask.sum())

        loss = mask_loss + topo_loss

        if torch.any(torch.isnan(loss)):
            print("NaN detected in loss. Using default loss value.")
            loss = torch.tensor(0.0, device=loss.device)
        self.log(
            "train_mask_loss", mask_loss, on_step=True, on_epoch=False, prog_bar=True
        )
        self.log(
            "train_topo_loss", topo_loss, on_step=True, on_epoch=False, prog_bar=True
        )
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

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
        mask_logits, mask_scores, topo_logits, topo_scores = self(
            rgb, graph_points, pairs, valid
        )
        gt_masks = torch.stack([keypoint_mask, road_mask], dim=3)
        mask_loss = self.mask_criterion(mask_logits, gt_masks)
        topo_gt, topo_loss_mask = batch["connected"].to(torch.int32), valid.to(
            torch.float32
        )
        # [B, N_samples, N_pairs, 1]
        topo_loss = self.topo_criterion(
            topo_logits, topo_gt.unsqueeze(-1).to(torch.float32)
        )
        topo_loss *= topo_loss_mask.unsqueeze(-1)
        topo_loss = topo_loss.sum() / topo_loss_mask.sum()
        loss = mask_loss + topo_loss
        self.log(
            "val_mask_loss", mask_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "val_topo_loss", topo_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        # Log images
        if batch_idx == 0:
            max_viz_num = 4
            viz_rgb = rgb[:max_viz_num, :, :]
            viz_pred_keypoint = mask_scores[:max_viz_num, :, :, 0]
            viz_pred_road = mask_scores[:max_viz_num, :, :, 1]
            viz_gt_keypoint = keypoint_mask[:max_viz_num, ...]
            viz_gt_road = road_mask[:max_viz_num, ...]

            columns = ["rgb", "gt_keypoint", "gt_road", "pred_keypoint", "pred_road"]
            data = [
                [wandb.Image(x.cpu().numpy()) for x in row]
                for row in list(
                    zip(
                        viz_rgb,
                        viz_gt_keypoint,
                        viz_gt_road,
                        viz_pred_road,
                        viz_pred_keypoint,
                    )
                )
            ]
            self.logger.log_table(key="viz_table", columns=columns, data=data)

        road_mask = (road_mask > 0.5).float()
        self.keypoint_iou.update(mask_scores[..., 0], keypoint_mask)
        self.road_iou.update(mask_scores[..., 1], road_mask)
        valid = valid.to(torch.int32)
        topo_gt = (1 - valid) * -1 + valid * topo_gt
        self.topo_f1.update(topo_scores, topo_gt.unsqueeze(-1))

    def on_validation_epoch_end(self):
        keypoint_iou = self.keypoint_iou.compute()
        road_iou = self.road_iou.compute()
        topo_f1 = self.topo_f1.compute()
        self.log("keypoint_iou", keypoint_iou)
        self.log("road_iou", road_iou)
        self.log("topo_f1", topo_f1)
        self.keypoint_iou.reset()
        self.road_iou.reset()
        self.topo_f1.reset()

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
                    "params": [p for p in self.map_decoder.parameters()],
                    "lr": self.config.BASE_LR,
                }
            ]
        param_dicts += decoder_params

        topo_net_params = [
            {
                "params": [p for p in self.topo_net.parameters()],
                "lr": self.config.BASE_LR,
            }
        ]
        param_dicts += topo_net_params

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
