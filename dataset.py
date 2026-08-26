"""
학습/검증용 PyTorch Dataset 정의.

세 데이터셋(cityscale, globalscale, spacenet)을 공통 인터페이스로 다룬다. 각 원본 위성
이미지(수천 x 수천 픽셀)에서 학습 시에는 무작위로, 평가 시에는 격자 형태로 PATCH_SIZE
크기의 패치를 잘라내고, 그 패치 영역에 해당하는 keypoint/road 마스크와 함께 GraphLabelGenerator로
생성한 그래프 노드/토폴로지 라벨을 반환한다.

핵심 클래스:
    - GraphLabelGenerator: 이미지 전체의 GT 도로 그래프로부터, 주어진 패치 영역 안에서
      학습에 쓸 노드 좌표와 (노드 쌍, 연결 여부) 샘플을 뽑아내는 라벨 생성기.
    - SatMapDataset: 위 라벨 생성기를 이미지별로 들고 있다가, __getitem__에서 패치를 잘라
      RGB/마스크/그래프 라벨을 함께 텐서로 반환하는 Dataset.
"""

import cv2
import math
import json
import torch
import rtree
import scipy
import pickle
import graph_utils
import numpy as np
import os.path as osp

from torch.utils.data import Dataset

# import os
# import addict


def read_rgb_img(path):
    """
    RGB 이미지를 불러오는 함수

    Args:
        path (str): 이미지 파일 경로

    Returns:
        np.ndarray: RGB 이미지 배열
    """
    bgr = cv2.imread(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def cityscale_data_partition():
    """
    Cityscale 데이터셋의 인덱스를 train, validation, test로 분할하는 함수.

    총 180개의 인덱스를 아래와 같이 분할한다:
      - train: 각 10개 그룹 중 앞 8개 (x % 10 < 8)
      - validation: 각 20개 그룹(0 ~ 19, 20 ~ 39...)에서 18번째 인덱스 (x % 20 == 18)
      - test: 각 10개 그룹 중 9번째 (x % 10 == 9) 및 각 20개 그룹 중 8번째 (x % 20 == 8)

    Returns:
        tuple: (train 인덱스 리스트, validation 인덱스 리스트, test 인덱스 리스트)
    """
    indrange_train = []
    indrange_test = []
    indrange_validation = []

    for x in range(180):
        # train: 각 10개 중 0~7번 인덱스
        if x % 10 < 8:
            indrange_train.append(x)

        # test: 각 10개 중 9번 + 각 20개 중 8번 인덱스
        if x % 10 == 9:
            indrange_test.append(x)
        if x % 20 == 8:
            indrange_test.append(x)

        # validation: 각 20개 중 18번 인덱스
        if x % 20 == 18:
            indrange_validation.append(x)

    return indrange_train, indrange_validation, indrange_test


def globalscale_data_partition():
    """
    글로벌 스케일 데이터셋의 인덱스를 train, validation, test, out-domain test로 분할하는 함수.

    데이터셋 인덱스 할당:
      - train: 0 ~ 2374 (총 2375개)
      - validation: 2375 ~ 2713 (총 339개)
      - test (in-domain): 2714 ~ 3337 (총 624개)
      - test (out-domain): 0 ~ 129 (총 130개, 별도로 지정되어 있음)

    Returns:
        tuple: (train 인덱스 리스트, validation 인덱스 리스트, test 인덱스 리스트, out-domain test 인덱스 리스트)
    """

    indrange_train = list(range(2375))  # 학습용
    indrange_validation = list(range(2375, 2714))  # 검증용
    indrange_test = list(range(2714, 3338))  # 테스트(in-domain)

    indrange_test_out_domain = list(range(130))  # 테스트(out-domain)

    return indrange_train, indrange_validation, indrange_test, indrange_test_out_domain


def spacenet_data_partition():
    """
    SpaceNet 데이터셋의 인덱스를 train, validation, test로 분할하는 함수.

    Returns:
        tuple: (train 인덱스 리스트, validation 인덱스 리스트, test 인덱스 리스트)
    """

    with open("/data2/Aerial/RoadGraph/spacenet/data_split.json", "r") as jf:
        data_list = json.load(jf)

    # 학습, 검증, 테스트 인덱스 리스트 추출
    train_list = data_list["train"]
    val_list = data_list["validation"]
    test_list = data_list["test"]

    # 결과 반환
    return train_list, val_list, test_list


def get_patch_info_one_img(
    image_index, image_size, sample_margin, patch_size, patches_per_edge
):
    """
    한 장의 이미지에서 패치 영역들의 정보를 계산하는 함수.

    Args:
        image_index (int): 이미지 인덱스 (배치 내 혹은 전체 이미지 중 몇 번째 이미지인지).
        image_size (int): 이미지 한 변의 픽셀 크기 (정사각형 이미지를 가정).
        sample_margin (int): 이미지 테두리에서 패치 추출 시 띄울 여백(마진, 픽셀 단위).
        patch_size (int): 추출할 패치 한 변의 픽셀 크기.
        patches_per_edge (int): 한 변을 따라 몇 개의 패치를 추출할지.

    Returns:
        list: (image_index, (좌상단(x, y)), (우하단(x, y))) 형태의 튜플 리스트.
    """

    patch_info = []

    # 패치 좌상단 좌표의 최소, 최대 범위 설정
    sample_min = sample_margin
    sample_max = image_size - (patch_size + sample_margin)

    # 패치 좌상단 좌표들을 동일 간격으로 샘플링
    eval_samples = np.linspace(start=sample_min, stop=sample_max, num=patches_per_edge)
    eval_samples = [round(x) for x in eval_samples]

    # 모든 x, y 조합에 대해 패치 좌표 정보 생성
    for x in eval_samples:
        for y in eval_samples:
            # (이미지 인덱스, (좌상단 좌표), (우하단 좌표)) 형태로 저장
            patch_info.append((image_index, (x, y), (x + patch_size, y + patch_size)))

    return patch_info


class GraphLabelGenerator:
    """이미지 한 장의 GT 도로 그래프(full_graph, sat2graph 포맷)로부터 학습용 그래프 라벨을
    생성하는 클래스. 이미지당 한 번 생성되어, sample_patch가 호출될 때마다 그 패치 영역
    안에서 노드 좌표와 (source, target) 쌍의 연결 여부(topology) 샘플을 뽑아 반환한다.

    __init__에서 미리 계산해두는 것들:
        - full_graph_subdivide: 엣지를 촘촘하게 세분화한 그래프 (라벨 밀도 확보용)
        - crossover_points / exclude_indices: 실제로는 연결되지 않은 채 겹쳐 보이는 지점을
          찾아 학습 라벨에서 제외
        - nms_score_override: 교차점(차수!=2)에는 항상 NMS에서 살아남도록 높은 점수 부여
        - sample_weights: 교차점 근방은 샘플링 확률을 높여 학습이 교차로 구조에 더 집중하도록 함
    """

    def __init__(self, config, full_graph, coord_transform):
        self.config = config
        self.full_graph_origin = graph_utils.igraph_from_adj_dict(
            full_graph, coord_transform
        )

        # crossover point 탐색
        self.crossover_points = graph_utils.find_crossover_points(
            self.full_graph_origin
        )

        # 그래프 세분화
        self.subdivide_resolution = 4
        self.full_graph_subdivide = graph_utils.subdivide_graph(
            self.full_graph_origin, self.subdivide_resolution
        )

        self.subdivide_points = np.array(self.full_graph_subdivide.vs["point"])

        self.graph_rtee = rtree.index.Index()
        for i, v in enumerate(self.subdivide_points):
            x, y = v
            self.graph_rtee.insert(i, (x, y, x, y))  # point 삽입

        self.graph_kdtree = scipy.spatial.KDTree(self.subdivide_points)

        # crossover 점들 학습 제외
        crossover_exclude_radius = 4
        exclude_indices = set()
        for p in self.crossover_points:
            nearby_indices = self.graph_kdtree.query_ball_point(
                p, crossover_exclude_radius
            )
            exclude_indices.update(nearby_indices)
        self.exclude_indices = exclude_indices

        # 교차로는 항상 학습에 중요하기 때문에 가중치 부여
        itsc_indices = set()
        point_num = len(self.full_graph_subdivide.vs)
        for i in range(point_num):
            if self.full_graph_subdivide.degree(i) != 2:
                itsc_indices.add(i)
        self.nms_score_override = np.zeros((point_num,), dtype=np.float32)
        self.nms_score_override[np.array(list(itsc_indices))] = 2.0

        interesting_indices = set()
        interesting_radius = 32
        # near itsc
        for i in itsc_indices:
            p = self.subdivide_points[i]
            nearby_indices = self.graph_kdtree.query_ball_point(p, interesting_radius)
            interesting_indices.update(nearby_indices)
        for p in self.crossover_points:
            nearby_indices = self.graph_kdtree.query_ball_point(
                np.array(p), interesting_radius
            )
            interesting_indices.update(nearby_indices)
        self.sample_weights = np.full((point_num,), 0.1, dtype=np.float32)
        self.sample_weights[list(interesting_indices)] = 0.9

    def sample_patch(self, patch, rot_index=0):
        """주어진 패치 영역 안에서 그래프 노드 좌표와 topology 학습 샘플을 뽑는다.

        절차: 패치 안의 세분화 그래프 점들을 rtree로 조회 -> 랜덤 점수 기반 NMS로 밀도를
        낮춤(교차점은 점수를 강제로 높여 항상 유지) -> sample_weights에 따라 TOPO_SAMPLE_NUM개
        점을 샘플링 -> 각 샘플 점마다 KDTree로 이웃 후보를 찾고 BFS로 실제 그래프상 연결
        여부를 판정 -> 좌표를 패치 기준 상대좌표로 변환(+회전 증강 적용).

        Args:
            patch (tuple): ((x0, y0), (x1, y1)) 패치의 좌상단/우하단 좌표.
            rot_index (int): 0~3, 90도 단위 회전 증강 인덱스 (train.py의 회전 증강과 일치시킴).

        Returns:
            tuple:
                nmsed_points (np.ndarray): [N, 2] 패치 좌표계로 변환된 노드 좌표.
                samples (list): 길이 TOPO_SAMPLE_NUM. 각 원소는
                    (pairs, shall_connect, valid) - 한 소스 노드에 대한 최대
                    MAX_NEIGHBOR_QUERIES개의 (source, target) 쌍과 실제 연결 여부, 유효성.
        """
        (x0, y0), (x1, y1) = patch
        query_box = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

        # 전역 index에서 패치내에 존재하는 index들
        patch_indices_all = set(self.graph_rtee.intersection(query_box))
        patch_indices = patch_indices_all - self.exclude_indices

        # Use NMS to downsample, params shall resemble inference time
        patch_indices = np.array(list(patch_indices))
        if len(patch_indices) == 0:
            # print("==== Patch is empty ====")
            # this shall be rare, but if no points in side the patch, return null stuff
            sample_num = self.config.TOPO_SAMPLE_NUM
            max_nbr_queries = self.config.MAX_NEIGHBOR_QUERIES
            fake_points = np.array([[0.0, 0.0]], dtype=np.float32)
            fake_sample = (
                [[0, 0]] * max_nbr_queries,
                [False] * max_nbr_queries,
                [False] * max_nbr_queries,
            )
            return fake_points, [fake_sample] * sample_num

        patch_points = self.subdivide_points[patch_indices, :]

        # GT에서 얻는 vertex의 점수를 랜덤 설정
        nms_scores = np.random.uniform(low=0.9, high=1.0, size=patch_indices.shape[0])

        # 교차점에 대한 점수는 graphlabelgenerator를 초기화할때 2.0으로 이미 할당
        # 교차점은 항상 유지
        nms_score_override = self.nms_score_override[patch_indices]
        nms_scores = np.maximum(nms_scores, nms_score_override)
        nms_radius = self.config.ROAD_NMS_RADIUS  # 16

        # 랜덤 점수 기반 nms
        nmsed_points, kept_indices = graph_utils.nms_points(
            patch_points, nms_scores, radius=nms_radius, return_indices=True
        )

        # now this is into the subdivide graph
        nmsed_indices = patch_indices[kept_indices]
        nmsed_point_num = nmsed_points.shape[0]

        sample_num = self.config.TOPO_SAMPLE_NUM  # 128 or 512
        sample_weights = self.sample_weights[nmsed_indices]  # 교차점은 0.9 나머지 0.1

        # indices into the nmsed points in the patch
        sample_indices_in_nmsed = np.random.choice(
            # nms 후 남은 vertex의 index 리스트
            np.arange(start=0, stop=nmsed_points.shape[0], dtype=np.int32),
            size=sample_num,  # 각 패치별 뽑을 vertex 수, 128 or 512
            replace=True,  # 복원 추출 여부, True면 중복 허용
            p=sample_weights / np.sum(sample_weights),  # 각 점이 뽑힐 가중치
        )

        sample_indices = nmsed_indices[sample_indices_in_nmsed]

        # 각 점의 이웃 node 탐색
        radius = self.config.NEIGHBOR_RADIUS  # 64
        max_nbr_queries = self.config.MAX_NEIGHBOR_QUERIES  # 16
        nmsed_kdtree = scipy.spatial.KDTree(nmsed_points)
        sampled_points = self.subdivide_points[sample_indices, :]

        # [n_sample, n_nbr]
        # k+1 because the nearest one is always self
        knn_d, knn_idx = nmsed_kdtree.query(
            sampled_points, k=max_nbr_queries + 1, distance_upper_bound=radius
        )  # k안에 속하지만, 설정 거리보다 멀면 inf

        samples = []
        for i in range(sample_num):
            source_node = sample_indices[i]

            # inf 처리를 위해 knn_idx[i, :] < nmsed_point_num
            valid_nbr_indices = knn_idx[i, knn_idx[i, :] < nmsed_point_num]

            # 자기 자신 제외
            valid_nbr_indices = valid_nbr_indices[1:]

            # 인덱스가 nmsed_kdtree 기준이므로 nmsed_indices
            target_nodes = [nmsed_indices[ni] for ni in valid_nbr_indices]

            ### BFS to find immediate neighbors on graph
            reached_nodes = graph_utils.bfs_with_conditions(
                self.full_graph_subdivide,
                source_node,
                set(target_nodes),
                radius // self.subdivide_resolution,  # NOTE 64 // 4 == 16인데.. 왜지?
            )
            shall_connect = [t in reached_nodes for t in target_nodes]
            ###
            pairs = []
            valid = []
            source_nmsed_idx = sample_indices_in_nmsed[i]
            for target_nmsed_idx in valid_nbr_indices:
                pairs.append((source_nmsed_idx, target_nmsed_idx))
                valid.append(True)
            # zero-pad
            for i in range(len(pairs), max_nbr_queries):
                pairs.append((source_nmsed_idx, source_nmsed_idx))
                shall_connect.append(False)
                valid.append(False)
            samples.append((pairs, shall_connect, valid))
        # Transform points
        # [N, 2]
        nmsed_points -= np.array([x0, y0])[np.newaxis, :]
        # homo for rot
        # [N, 3]
        nmsed_points = np.concatenate(
            [nmsed_points, np.ones((nmsed_point_num, 1), dtype=nmsed_points.dtype)],
            axis=1,
        )
        trans = np.array(
            [
                [1, 0, -0.5 * self.config.PATCH_SIZE],
                [0, 1, -0.5 * self.config.PATCH_SIZE],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
        # ccw 90 deg in img (x, y)
        rot = np.array(
            [
                [0, 1, 0],
                [-1, 0, 0],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
        nmsed_points = (
            nmsed_points
            @ trans.T
            @ np.linalg.matrix_power(rot.T, rot_index)
            @ np.linalg.inv(trans.T)
        )
        nmsed_points = nmsed_points[:, :2]
        return nmsed_points, samples


def graph_collate_fn(batch):
    """
    배치 내 그래프 데이터를 올바르게 패딩 및 스택하여 배치 텐서로 변환하는 collate 함수
    그래프의 포인트 수가 샘플마다 다르므로, "graph_points" 항목은 패딩 처리하여 배치로 만든다.
    나머지 항목들은 일반적인 스택 처리.
    """
    keys = batch[0].keys()
    collated = {}
    for key in keys:
        if key == "graph_points":
            # 각 배치 요소에서 "graph_points" 텐서를 모음
            tensors = [item[key] for item in batch]
            # 최대 포인트 개수 계산 (동일 크기로 패딩 목적)
            max_point_num = max([x.shape[0] for x in tensors])
            padded = []
            for x in tensors:
                pad_num = max_point_num - x.shape[0]
                # 부족한 개수만큼 (0, 0)으로 패딩
                padded_x = torch.concat([x, torch.zeros(pad_num, 2)], dim=0)
                padded.append(padded_x)
            # 패딩된 텐서들을 배치 차원으로 스택
            collated[key] = torch.stack(padded, dim=0)
        else:
            # "graph_points" 이외 항목은 단순히 스택
            collated[key] = torch.stack([item[key] for item in batch], dim=0)
    return collated


class SatMapDataset(Dataset):
    """cityscale / globalscale / spacenet 위성 이미지 데이터셋 공통 클래스.

    is_train=True면 매 __getitem__마다 이미지와 패치 위치를 무작위로 골라 샘플링하고(+90도
    단위 회전 증강), is_train=False면 get_patch_info_one_img로 미리 계산해둔 격자 패치들을
    순서대로 순회한다(평가 시 재현성/커버리지 확보). 반환값에는 GraphLabelGenerator가 만든
    그래프 노드/토폴로지 라벨이 함께 포함된다.
    """

    def __init__(
        self,
        config,
        is_train,
        data_root="/home/work/data/RoadGraph",
        dev_run=False,
    ):
        self.config = config
        assert self.config.DATASET in {"cityscale", "globalscale", "spacenet"}
        if self.config.DATASET == "cityscale":
            self.IMAGE_SIZE = 2048
            # TODO: SAMPLE_MARGIN here is for training, the one in config is for inference
            self.SAMPLE_MARGIN = 64
            rgb_pattern = osp.join(data_root, "cityscale/20cities/region_{}_sat.png")
            keypoint_mask_pattern = osp.join(
                data_root, "cityscale/processed/keypoint_mask_{}.png"
            )
            road_mask_pattern = osp.join(
                data_root, "cityscale/processed/road_mask_{}.png"
            )
            gt_graph_pattern = osp.join(
                data_root, "cityscale/20cities/region_{}_refine_gt_graph.p"
            )

            train, val, test = cityscale_data_partition()

            # coord-transform = (r, c) -> (x, y)
            # takes [N, 2] points
            coord_transform = lambda v: v[:, ::-1]

        # 경로 수정 필요
        elif self.config.DATASET == "globalscale":
            self.IMAGE_SIZE = 2048
            # TODO: SAMPLE_MARGIN here is for training, the one in config is for inference
            self.SAMPLE_MARGIN = 64
            rgb_pattern = osp.join(
                data_root, "Global-Scale/Global-Scale/train/region_{}_sat.png"
            )
            keypoint_mask_pattern = osp.join(
                data_root, "Global-Scale/Global-Scale/processed/keypoint_mask_{}.png"
            )
            road_mask_pattern = osp.join(
                data_root, "Global-Scale/Global-Scale/processed/road_mask_{}.png"
            )
            gt_graph_pattern = osp.join(
                data_root, "Global-Scale/Global-Scale/train/region_{}_refine_gt_graph.p"
            )

            train, val, test, test_out = globalscale_data_partition()

            # coord-transform = (r, c) -> (x, y)
            # takes [N, 2] points
            coord_transform = lambda v: v[:, ::-1]

        elif self.config.DATASET == "spacenet":
            self.IMAGE_SIZE = 400
            self.SAMPLE_MARGIN = 0
            rgb_pattern = osp.join(data_root, "spacenet/RGB_1.0_meter/{}__rgb.png")
            keypoint_mask_pattern = osp.join(
                data_root, "spacenet/processed/keypoint_mask_{}.png"
            )
            road_mask_pattern = osp.join(
                data_root, "spacenet/processed/road_mask_{}.png"
            )
            gt_graph_pattern = osp.join(
                data_root, "spacenet/RGB_1.0_meter/{}__gt_graph.p"
            )

            train, val, test = spacenet_data_partition()

            # coord-transform, (r, c) -> (x, y) 및 y축 반전
            # takes [N, 2] points
            coord_transform = lambda v: np.stack([v[:, 1], 400 - v[:, 0]], axis=1)

        self.is_train = is_train

        train_split = train + val
        test_split = test
        tile_indices = train_split if self.is_train else test_split

        self.tile_indices = tile_indices

        # rgb 이미지 리스트, 키포인트 마스크 리스트, 도로 마스크 리스트
        self.rgbs, self.keypoint_masks, self.road_masks = [], [], []
        # For graph label generation.
        self.graph_label_generators = []

        ##### FAST DEBUG
        if dev_run:
            tile_indices = tile_indices[:4]

        self.trainnum = tile_indices

        for tile_idx in tile_indices:
            print(f"loading tile {tile_idx}")
            rgb_path = rgb_pattern.format(tile_idx)
            road_mask_path = road_mask_pattern.format(tile_idx)
            keypoint_mask_path = keypoint_mask_pattern.format(tile_idx)

            # graph label gen
            gt_graph_adj = pickle.load(open(gt_graph_pattern.format(tile_idx), "rb"))
            if len(gt_graph_adj) == 0:
                print(f"===== skipped empty tile {tile_idx} =====")
                continue
            self.rgbs.append(read_rgb_img(rgb_path))
            self.road_masks.append(cv2.imread(road_mask_path, cv2.IMREAD_GRAYSCALE))
            self.keypoint_masks.append(
                cv2.imread(keypoint_mask_path, cv2.IMREAD_GRAYSCALE)
            )
            graph_label_generator = GraphLabelGenerator(
                config, gt_graph_adj, coord_transform
            )
            self.graph_label_generators.append(graph_label_generator)

        self.sample_min = self.SAMPLE_MARGIN
        self.sample_max = self.IMAGE_SIZE - (
            self.config.PATCH_SIZE + self.SAMPLE_MARGIN
        )  # 이미지 경계에서 margin 떨어진 곳까지만 샘플링

        if not self.is_train:  # 평가시에는 랜덤 샘플링이 아니라 순서대로 수행
            eval_patches_per_edge = math.ceil(
                (self.IMAGE_SIZE - 2 * self.SAMPLE_MARGIN) / self.config.PATCH_SIZE
            )
            self.eval_patches = []
            for i in range(len(tile_indices)):
                self.eval_patches += get_patch_info_one_img(
                    i,
                    self.IMAGE_SIZE,
                    self.SAMPLE_MARGIN,
                    self.config.PATCH_SIZE,
                    eval_patches_per_edge,
                )

    def __len__(self):
        if self.is_train:
            if (
                self.config.DATASET == "cityscale"
                or self.config.DATASET == "globalscale"
            ):
                num_patches_per_image = (
                    max(1, int(self.IMAGE_SIZE / self.config.PATCH_SIZE)) ** 2
                )  # SAMRoad에서는 4 ** 2 * 2500 -> 4만번
                return len(self.trainnum) * num_patches_per_image
            elif self.config.DATASET == "spacenet":
                num_patches_per_image = (
                    max(1, int(self.IMAGE_SIZE / self.config.PATCH_SIZE)) ** 2
                )
                return (
                    len(self.trainnum) * num_patches_per_image
                )  # SAMRoad에서는 84667 하드코딩
        else:
            return len(self.eval_patches)

    def __getitem__(self, idx):
        if self.is_train:
            img_idx = np.random.randint(low=0, high=len(self.rgbs))
            begin_x = np.random.randint(low=self.sample_min, high=self.sample_max + 1)
            begin_y = np.random.randint(low=self.sample_min, high=self.sample_max + 1)
            end_x, end_y = (
                begin_x + self.config.PATCH_SIZE,
                begin_y + self.config.PATCH_SIZE,
            )
        else:
            img_idx, (begin_x, begin_y), (end_x, end_y) = self.eval_patches[idx]

        rgb_patch = self.rgbs[img_idx][begin_y:end_y, begin_x:end_x, :]
        keypoint_mask_patch = self.keypoint_masks[img_idx][begin_y:end_y, begin_x:end_x]
        road_mask_patch = self.road_masks[img_idx][begin_y:end_y, begin_x:end_x]

        rot_index = 0
        if self.is_train:
            rot_index = np.random.randint(0, 4)
            # CCW
            rgb_patch = np.rot90(rgb_patch, rot_index, [0, 1]).copy()
            keypoint_mask_patch = np.rot90(
                keypoint_mask_patch, rot_index, [0, 1]
            ).copy()
            road_mask_patch = np.rot90(road_mask_patch, rot_index, [0, 1]).copy()
        # Sample graph labels from patch
        patch = ((begin_x, begin_y), (end_x, end_y))
        # points are img (x, y) inside the patch.
        graph_points, topo_samples = self.graph_label_generators[img_idx].sample_patch(
            patch, rot_index
        )
        pairs, connected, valid = zip(*topo_samples)
        # rgb: [H, W, 3] 0-255
        # masks: [H, W] 0-1

        return {
            "rgb": torch.tensor(rgb_patch, dtype=torch.float32),  # H, W, 3
            "keypoint_mask": torch.tensor(keypoint_mask_patch, dtype=torch.float32)
            / 255.0,  # H, W
            "road_mask": torch.tensor(road_mask_patch, dtype=torch.float32)
            / 255.0,  # H, W
            "graph_points": torch.tensor(
                graph_points, dtype=torch.float32
            ),  # N_points, 2
            "pairs": torch.tensor(pairs, dtype=torch.int32),  # 128, 16, 2 or 512, 16, 2
            "connected": torch.tensor(
                connected, dtype=torch.bool
            ),  # 128, 16 or 512, 16
            "valid": torch.tensor(valid, dtype=torch.bool),  # 128, 16 or 512, 16
        }
