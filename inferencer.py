"""
학습된 SAM-Road++ 체크포인트로 전체 위성 이미지(2048x2048 등)에 대해 도로 그래프를 추론하는 스크립트.

이미지 한 장이 모델의 입력 패치 크기(PATCH_SIZE)보다 크기 때문에, 다음 2-pass 절차로 진행한다.

1. Pass 1 (`infer_one_img` 앞부분): 이미지를 겹치는 패치들로 잘라 배치로 모델에 통과시켜
   keypoint/road 마스크를 얻고, 겹치는 영역은 평균을 내어(fused_*) 이미지 전체 크기의
   마스크로 합친다. 이때 각 패치의 이미지 임베딩(img_features)도 저장해 둔다 (2-pass에서 재사용).
2. graph_extraction.extract_graph_points로 합쳐진 마스크에서 그래프 노드(교차점/도로점) 후보를 뽑는다.
3. Pass 2: 노드들을 다시 패치 단위로 묶어, 저장해둔 이미지 임베딩과 함께 TopoNet에 넣어
   각 노드 쌍이 연결되어 있을 확률(topo_scores)을 구한다. 여러 패치에 걸쳐 중복 예측된
   엣지는 점수를 평균 내고, TOPO_THRESHOLD를 넘는 엣지만 최종 그래프로 채택한다.

실행: `python inferencer.py --config=<학습에 쓴 config.yaml> --checkpoint=<ckpt 경로>`
결과는 `--output_dir`(또는 timestamp 기반 디렉터리) 아래 mask/viz/graph 3개 폴더에 저장된다.
"""

import os
import cv2
import time
import rtree
import torch
import scipy
import pickle
import triage
import graph_utils
import numpy as np
import os.path as osp
import graph_extraction

from utils import load_config, create_output_dir_and_save_config
from dataset_refactor import read_rgb_img, get_patch_info_one_img
from dataset_refactor import (
    spacenet_data_partition,
    cityscale_data_partition,
    globalscale_data_partition,
)
from argparse import ArgumentParser
from modelinfer import SAMRoadplus as OriginSAMRoadplus
from probit_modelinfer import SAMRoadplus as ProbitSAMRoadplus
from collections import defaultdict


parser = ArgumentParser()
parser.add_argument("--checkpoint", default="", help="checkpoint of the model to test.")
parser.add_argument("--config", default="", help="model config.")
parser.add_argument(
    "--output_dir",
    default="",
    help="Name of the output dir, if not specified will use timestamp",
)
parser.add_argument("--data_root", default="/home/work/data/RoadGraph")
parser.add_argument("--device", default="cuda", help="device to use for training")
parser.add_argument("--probit", default="False", action="store_true")
args = parser.parse_args()


def get_img_paths(root_dir, image_indices):
    img_paths = []

    for ind in image_indices:
        img_paths.append(osp.join(root_dir, f"region_{ind}_sat.png"))
    return img_paths


def crop_img_patch(img, x0, y0, x1, y1):
    return img[y0:y1, x0:x1, :]


def get_batch_img_patches(img, batch_patch_info):
    """dataset.get_patch_info_one_img로 만든 패치 좌표 목록에 따라 실제 이미지를 잘라
    [B, H, W, C] 텐서 배치로 쌓는다."""
    patches = []
    for _, (x0, y0), (x1, y1) in batch_patch_info:
        patch = crop_img_patch(img, x0, y0, x1, y1)
        patches.append(torch.tensor(patch, dtype=torch.float32))
    batch = torch.stack(patches, 0).contiguous()
    return batch


def infer_one_img(net, img, config):
    """이미지 한 장에 대해 위 모듈 docstring에 설명한 2-pass 추론을 수행한다.

    Returns:
        tuple:
            pred_nodes (np.ndarray): [N, 2] 예측된 노드 좌표, (row, col) 순서.
            pred_edges (np.ndarray): [E, 2] 예측된 엣지, 노드 인덱스 쌍.
            fused_keypoint_mask (np.ndarray): [H, W] uint8, 합쳐진 keypoint 마스크.
            fused_road_mask (np.ndarray): [H, W] uint8, 합쳐진 road 마스크.
    """
    # TODO(congrui): centralize these configs
    image_size = img.shape[0]
    batch_size = config.INFER_BATCH_SIZE
    # list of (i, (x_begin, y_begin), (x_end, y_end))
    all_patch_info = get_patch_info_one_img(
        0,
        image_size,
        config.SAMPLE_MARGIN,
        config.PATCH_SIZE,
        config.INFER_PATCHES_PER_EDGE,
    )
    patch_num = len(all_patch_info)
    batch_num = (
        patch_num // batch_size
        if patch_num % batch_size == 0
        else patch_num // batch_size + 1
    )
    # [IMG_H, IMG_W]
    fused_keypoint_mask = torch.zeros(img.shape[0:2], dtype=torch.float32).to(
        args.device, non_blocking=False
    )
    fused_road_mask = torch.zeros(img.shape[0:2], dtype=torch.float32).to(
        args.device, non_blocking=False
    )
    pixel_counter = torch.zeros(img.shape[0:2], dtype=torch.float32).to(
        args.device, non_blocking=False
    )
    # stores img embeddings for toponet
    # list of [B, D, h, w], len=batch_num
    img_features = list()
    img_mask = list()
    ## Pass 1: 패치 단위로 마스크와 이미지 임베딩을 추론하고, 겹치는 영역을 평균내어 합친다.
    for batch_index in range(batch_num):
        offset = batch_index * batch_size
        batch_patch_info = all_patch_info[offset : offset + batch_size]
        # tensor [B, H, W, C]
        batch_img_patches = get_batch_img_patches(img, batch_patch_info)

        with torch.no_grad():
            batch_img_patches = batch_img_patches.to(args.device, non_blocking=False)
            # [B, H, W, 2]
            mask_scores, patch_img_features = net.infer_masks_and_img_features(
                batch_img_patches
            )
            img_features.append(patch_img_features)

            # 디버깅: 마스크 점수 확인
            print(f"Batch {batch_index}: mask_scores shape = {mask_scores.shape}")
            print(
                f"mask_scores range: [{mask_scores.min():.4f}, {mask_scores.max():.4f}]"
            )
            print(f"mask_scores mean: {mask_scores.mean():.4f}")

            mask_scores11 = mask_scores.permute(0, 3, 1, 2)  # (B, 2, H, W)

            img_mask.append(mask_scores11)
        # Aggregate masks
        for patch_index, patch_info in enumerate(batch_patch_info):
            _, (x0, y0), (x1, y1) = patch_info
            keypoint_patch, road_patch = (
                mask_scores[patch_index, :, :, 0],
                mask_scores[patch_index, :, :, 1],
            )
            fused_keypoint_mask[y0:y1, x0:x1] += keypoint_patch
            fused_road_mask[y0:y1, x0:x1] += road_patch
            pixel_counter[y0:y1, x0:x1] += torch.ones(
                road_patch.shape[0:2], dtype=torch.float32, device=args.device
            )

    fused_keypoint_mask /= pixel_counter
    fused_road_mask /= pixel_counter

    # range 0-1 -> 0-255
    fused_keypoint_mask = (fused_keypoint_mask * 255).to(torch.uint8).cpu().numpy()
    fused_road_mask = (fused_road_mask * 255).to(torch.uint8).cpu().numpy()

    print(f"After scaling to 0-255:")
    print(
        f"keypoint_mask range: [{fused_keypoint_mask.min()}, {fused_keypoint_mask.max()}]"
    )
    print(f"road_mask range: [{fused_road_mask.min()}, {fused_road_mask.max()}]")
    print(f"road_mask shape: {fused_road_mask.shape}")
    graph_points = graph_extraction.extract_graph_points(
        fused_keypoint_mask, fused_road_mask, config
    )
    if graph_points.shape[0] == 0:
        # 노드 후보가 하나도 추출되지 않은 경우(빈 이미지 등) 빈 그래프로 조기 반환
        print(1)
        print(graph_points)
        return (
            graph_points,
            np.zeros((0, 2), dtype=np.int32),
            fused_keypoint_mask,
            fused_road_mask,
        )
    # for box query
    graph_rtree = rtree.index.Index()
    for i, v in enumerate(graph_points):
        x, y = v
        # hack to insert single points
        graph_rtree.insert(i, (x, y, x, y))
    ## Pass 2: 저장해둔 이미지 임베딩을 이용해 TopoNet으로 노드 쌍의 연결 확률(topology)을 추론한다.
    edge_scores = defaultdict(float)
    edge_counts = defaultdict(float)
    for batch_index in range(batch_num):
        offset = batch_index * batch_size
        batch_patch_info = all_patch_info[offset : offset + batch_size]

        topo_data = {
            "points": [],
            "pairs": [],
            "valid": [],
        }
        idx_maps = []
        # prepares pairs queries
        for patch_info in batch_patch_info:
            _, (x0, y0), (x1, y1) = patch_info
            patch_point_indices = list(graph_rtree.intersection((x0, y0, x1, y1)))
            idx_patch2all = {
                patch_idx: all_idx
                for patch_idx, all_idx in enumerate(patch_point_indices)
            }
            patch_point_num = len(patch_point_indices)
            # normalize into patch
            patch_points = graph_points[patch_point_indices, :] - np.array(
                [[x0, y0]], dtype=graph_points.dtype
            )
            # for knn and circle query
            patch_kdtree = scipy.spatial.KDTree(patch_points)
            # k+1 because the nearest one is always self
            # idx is to the patch subgraph
            knn_d, knn_idx = patch_kdtree.query(
                patch_points,
                k=config.MAX_NEIGHBOR_QUERIES + 1,
                distance_upper_bound=config.NEIGHBOR_RADIUS,
            )
            # [patch_point_num, n_nbr]
            knn_idx = knn_idx[:, 1:]  # removes self
            # [patch_point_num, n_nbr] idx is to the patch subgraph
            src_idx = np.tile(
                np.arange(patch_point_num)[:, np.newaxis],
                (1, config.MAX_NEIGHBOR_QUERIES),
            )
            valid = knn_idx < patch_point_num
            tgt_idx = np.where(valid, knn_idx, src_idx)
            # [patch_point_num, n_nbr, 2]
            pairs = np.stack([src_idx, tgt_idx], axis=-1)

            topo_data["points"].append(patch_points)
            topo_data["pairs"].append(pairs)
            topo_data["valid"].append(valid)
            idx_maps.append(idx_patch2all)
        # collate
        collated = {}
        for key, x_list in topo_data.items():
            length = max([x.shape[0] for x in x_list])
            collated[key] = np.stack(
                [
                    np.pad(
                        x, [(0, length - x.shape[0])] + [(0, 0)] * (len(x.shape) - 1)
                    )
                    for x in x_list
                ],
                axis=0,
            )
        # skips this batch if there's no points
        if collated["points"].shape[1] == 0:
            continue
        # infer toponet
        # [B, D, h, w]
        batch_features = img_features[batch_index]
        batch_mask = img_mask[batch_index]
        # [B, N_sample, N_pair, 2]
        batch_points = torch.tensor(collated["points"], device=args.device)
        batch_pairs = torch.tensor(collated["pairs"], device=args.device)
        batch_valid = torch.tensor(collated["valid"], device=args.device)
        with torch.no_grad():
            # [B, N_samples, N_pairs, 1]
            topo_scores = net.infer_toponet(
                batch_features, batch_points, batch_pairs, batch_valid, batch_mask
            )
        # all-invalid (padded, no neighbors) queries returns nan scores
        # [B, N_samples, N_pairs]
        topo_scores = (
            torch.where(torch.isnan(topo_scores), -100.0, topo_scores)
            .squeeze(-1)
            .cpu()
            .numpy()
        )
        # aggregate edge scores
        batch_size, n_samples, n_pairs = topo_scores.shape
        for bi in range(batch_size):
            for si in range(n_samples):
                for pi in range(n_pairs):
                    if not collated["valid"][bi, si, pi]:
                        continue
                    # idx to the full graph
                    src_idx_patch, tgt_idx_patch = collated["pairs"][bi, si, pi, :]
                    src_idx_all, tgt_idx_all = (
                        idx_maps[bi][src_idx_patch],
                        idx_maps[bi][tgt_idx_patch],
                    )
                    edge_score = topo_scores[bi, si, pi]
                    assert 0.0 <= edge_score <= 1.0
                    edge_scores[(src_idx_all, tgt_idx_all)] += edge_score
                    edge_counts[(src_idx_all, tgt_idx_all)] += 1.0
    # 같은 엣지가 여러 패치에서 중복 예측될 수 있으므로 점수를 평균 내고,
    # TOPO_THRESHOLD를 넘는 엣지만 최종 채택한다.
    pred_edges = []
    for edge, score_sum in edge_scores.items():
        score = score_sum / edge_counts[edge]
        if score > config.TOPO_THRESHOLD:
            pred_edges.append(edge)
    pred_edges = np.array(pred_edges).reshape(-1, 2)
    pred_nodes = graph_points[:, ::-1]  # to rc

    return pred_nodes, pred_edges, fused_keypoint_mask, fused_road_mask


if __name__ == "__main__":
    config = load_config(args.config)
    # 평가용 모델 생성 및 체크포인트 로드
    device = torch.device("cuda") if args.device == "cuda" else torch.device("cpu")
    # Good when model architecture/input shape are fixed.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    net = ProbitSAMRoadplus(config) if args.probit else OriginSAMRoadplus(config)

    # load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    print(f"##### Loading Trained CKPT {args.checkpoint} #####")
    net.load_state_dict(checkpoint["state_dict"], strict=True)
    net.eval()
    net.to(device)

    if config.DATASET == "cityscale":
        _, _, test_img_indices = cityscale_data_partition()
        rgb_pattern = osp.join(args.data_root, "cityscale/20cities/region_{}_sat.png")

    elif config.DATASET == "globalscale_outdomain":
        _, _, _, test_img_indices = globalscale_data_partition()
        rgb_pattern = "../region_{}_sat.png"

    elif config.DATASET == "globalscale":
        _, _, test_img_indices, _ = globalscale_data_partition()
        rgb_pattern = "../region_{}_sat.png"

    elif config.DATASET == "spacenet":
        _, _, test_img_indices = spacenet_data_partition()
        rgb_pattern = osp.joint(args.data_root, "spacenet/RGB_1.0_meter/{}__rgb.png")

    output_dir_prefix = "./save/infer_"
    # if args.output_dir:
    #     output_dir = create_output_dir_and_save_config(
    #         output_dir_prefix,
    #         config,
    #         specified_dir=f"./outputs/{config.WANDB_PROJECT_NAME}/{config.WANDB_EXPERIMENT_NAME}/save",
    #     )
    # else:
    output_dir = create_output_dir_and_save_config(
        output_dir_prefix,
        config,
        specified_dir=f"./outputs/{config.WANDB_PROJECT_NAME}/{config.WANDB_EXPERIMENT_NAME}/save",
    )

    total_inference_seconds = 0.0

    # 테스트 이미지들을 순회하며 그래프 추론 -> 마스크/시각화/그래프(pickle) 저장
    for img_id in test_img_indices:
        print(f"Processing {img_id}")
        # [H, W, C] RGB
        img = read_rgb_img(rgb_pattern.format(img_id))
        start_seconds = time.time()
        # coords in (r, c)
        pred_nodes, pred_edges, itsc_mask, road_mask = infer_one_img(net, img, config)
        end_seconds = time.time()
        total_inference_seconds += end_seconds - start_seconds

        # RGB already
        viz_img = np.copy(img)
        img_size = viz_img.shape[0]

        # visualizes fused masks
        mask_save_dir = osp.join(output_dir, "mask")
        if not osp.exists(mask_save_dir):
            os.makedirs(mask_save_dir)
        cv2.imwrite(osp.join(mask_save_dir, f"{img_id}_road.png"), road_mask)
        cv2.imwrite(osp.join(mask_save_dir, f"{img_id}_itsc.png"), itsc_mask)

        viz_save_dir = osp.join(output_dir, "viz")
        if not osp.exists(viz_save_dir):
            os.makedirs(viz_save_dir)
        viz_img = triage.visualize_image_and_graph(
            viz_img, pred_nodes / img_size, pred_edges, viz_img.shape[0]
        )
        cv2.imwrite(osp.join(viz_save_dir, f"{img_id}.png"), viz_img)

        # Saves the large map
        if config.DATASET == "spacenet":
            # r, c -> ???
            pred_nodes = np.stack([400 - pred_nodes[:, 0], pred_nodes[:, 1]], axis=1)
        large_map_sat2graph_format = graph_utils.convert_to_sat2graph_format(
            pred_nodes, pred_edges
        )
        graph_save_dir = osp.join(output_dir, "graph")
        if not osp.exists(graph_save_dir):
            os.makedirs(graph_save_dir)
        graph_save_path = osp.join(graph_save_dir, f"{img_id}.p")
        with open(graph_save_path, "wb") as file:
            pickle.dump(large_map_sat2graph_format, file)

        print(f"Done for {img_id}.")

    # log inference time
    time_txt = (
        f"Inference completed for {args.config} in {total_inference_seconds} seconds."
    )
    print(time_txt)
    with open(osp.join(output_dir, "inference_time.txt"), "w") as f:
        f.write(time_txt)
