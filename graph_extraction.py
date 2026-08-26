"""
모델이 예측한 keypoint mask / road mask(픽셀 단위 확률맵)로부터 도로 그래프(노드+엣지)를
추출하는 후처리 로직.

흐름:
1. `extract_graph_points`: 두 마스크에서 임계값을 넘는 픽셀을 후보 점으로 뽑고 NMS로 솎아낸다.
2. `extract_graph_astar`: 후보 점들 사이를 A*(tcod)로 실제 도로 위를 지나는지 검사해 엣지를 연결한다.
   (참고: SAM-Road++ 파이프라인 실제 추론에서는 A* 대신 학습된 TopoNet으로 엣지를 예측하며,
   이 파일의 A* 방식은 대안/디버깅용 그래프 추출 방법이다.)
"""

import cv2
import tcod
import numpy as np
import networkx as nx

from skimage.draw import line
from graph_utils import nms_points
from sklearn.neighbors import KDTree

# import math
# import torch
# from torch.utils.data import Dataset

IMAGE_SIZE = 2048
SAMPLE_MARGIN = 64


def read_rgb_img(path):
    bgr = cv2.imread(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


# returns (x, y)
def get_points_and_scores_from_mask(mask, threshold):
    """마스크에서 threshold를 넘는 픽셀들의 (x, y) 좌표와 해당 값을 점수로 추출한다."""
    rcs = np.column_stack(np.where(mask > threshold))
    xys = rcs[:, ::-1]
    scores = mask[mask > threshold]
    return xys, scores


def draw_points_on_image(image, points, radius):
    """
    Draws points on a square image using OpenCV.

    Parameters:
    - size: The size of the square image (width and height) in pixels.
    - points: A list of tuples, where each tuple represents the (x, y) coordinates of a point in pixel coordinates.
    - radius: The radius of the circles to be drawn for each point, in pixels.

    Returns:
    - A square image with the given points drawn as filled circles.
    """

    # Iterate through the list of points
    for point in points:
        cv2.circle(image, point, radius, (0, 255, 0), -1)

    return image


def draw_points_on_grayscale_image(image, points, radius):
    """
    Draws points on a square image using OpenCV.

    Parameters:
    - size: The size of the square image (width and height) in pixels.
    - points: A list of tuples, where each tuple represents the (x, y) coordinates of a point in pixel coordinates.
    - radius: The radius of the circles to be drawn for each point, in pixels.

    Returns:
    - A square image with the given points drawn as filled circles.
    """

    # Iterate through the list of points
    for point in points:
        cv2.circle(image, point, radius, 255, -1)

    return image


# takes xy
def is_connected_bresenham(cost, start, end):
    """Bresenham 직선 알고리즘으로 start-end를 잇는 픽셀들을 훑어, cost 값이 모두 255 미만인지
    (즉, 도로가 아닌 막힌 픽셀을 지나지 않는지) 검사한다. is_connected_astar보다 단순/저렴한
    "직선 경로만" 검사하는 방식 (현재 코드에서는 is_connected_astar가 실제로 쓰인다)."""
    c0, r0 = start
    c1, r1 = end
    rr, cc = line(r0, c0, r1, c1)
    kp_block_radius = 4
    cv2.circle(cost, start, kp_block_radius, 0, -1)
    cv2.circle(cost, end, kp_block_radius, 0, -1)

    # mean_cost = np.mean(cost[rr, cc])
    max_cost = np.max(cost[rr, cc])

    cv2.circle(cost, start, kp_block_radius, 255, -1)
    cv2.circle(cost, end, kp_block_radius, 255, -1)

    return max_cost < 255


def is_connected_astar(pathfinder, cost, start, end, max_path_len):
    """A*(tcod)로 start에서 end까지 cost 맵 위를 우회 경로까지 포함해 탐색하고,
    경로가 존재하며 그 길이가 max_path_len보다 짧으면 두 점이 연결된 것으로 판단한다.
    시작/끝점 주변은 일시적으로 통행 가능(cost=1)하게 뚫어준 뒤 탐색이 끝나면 원복한다."""
    # we can still modify the cost matrix after creating the pathfinder with it
    # seems pathfinder uses reference
    c0, r0 = start
    c1, r1 = end
    kp_block_radius = 6
    cv2.circle(cost, start, kp_block_radius, 1, -1)
    cv2.circle(cost, end, kp_block_radius, 1, -1)

    path = pathfinder.get_path(r0, c0, r1, c1)
    connected = (len(path) != 0) and (len(path) < max_path_len)

    cv2.circle(cost, start, kp_block_radius, 0, -1)
    cv2.circle(cost, end, kp_block_radius, 0, -1)

    return connected


def create_cost_field(sample_pts, road_mask):
    """is_connected_bresenham용 cost 필드 생성. 도로가 아닌 영역(255-road_mask)과
    후보 점 주변 원을 255(막힘)로 표시한다. (road mask는 0-255 uint8이어야 함)"""
    # road mask shall be uint8 normalized to 0-255
    cost_field = np.zeros(road_mask.shape, dtype=np.uint8)
    kp_block_radius = 4
    for point in sample_pts:
        cv2.circle(cost_field, point, kp_block_radius, 255, -1)
    cost_field = np.maximum(cost_field, 255 - road_mask)
    return cost_field


def create_cost_field_astar(sample_pts, road_mask, block_threshold=200):
    """is_connected_astar(tcod)용 cost 필드 생성. tcod에서는 0이 '통행 불가'를 의미하므로
    road_mask 기반 cost를 반전시켜, 도로가 아니거나(255-road_mask) 값이 block_threshold보다
    큰 픽셀은 0(막힘)으로, 그 외는 1(통행 가능, 낮은 비용)로 만든다.
    (road mask는 0-255 uint8이어야 함)"""
    # road mask shall be uint8 normalized to 0-255
    # for tcod, 0 is blocked
    cost_field = np.zeros(road_mask.shape, dtype=np.uint8)
    kp_block_radius = 6
    for point in sample_pts:
        cv2.circle(cost_field, point, kp_block_radius, 255, -1)
    cost_field = np.maximum(cost_field, 255 - road_mask)
    cost_field[cost_field == 0] = 1
    cost_field[cost_field > block_threshold] = 0

    return cost_field


def extract_graph_points(keypoint_mask, road_mask, config):
    """keypoint_mask와 road_mask로부터 그래프 노드가 될 후보 점들을 추출한다.

    1) keypoint_mask(교차점 확률맵)에서 ITSC_THRESHOLD를 넘는 픽셀을 NMS(반경 ITSC_NMS_RADIUS)로 솎아낸다.
    2) road_mask(도로 확률맵)에서 ROAD_THRESHOLD를 넘는 픽셀을 NMS(반경 ROAD_NMS_RADIUS)로 솎아낸다.
    3) 두 후보 집합을 합치되, keypoint 유래 점에는 점수 1.0을 부여해 NMS 시 우선권을 준다
       (교차점이 도로 점보다 우선적으로 살아남도록).

    Returns:
        np.ndarray: [N, 2] 최종 채택된 (x, y) 노드 후보 좌표.
    """
    kp_candidates, kp_scores = get_points_and_scores_from_mask(
        keypoint_mask, config.ITSC_THRESHOLD * 255
    )
    kps_0 = nms_points(kp_candidates, kp_scores, config.ITSC_NMS_RADIUS)
    kp_candidates, kp_scores = get_points_and_scores_from_mask(
        road_mask, config.ROAD_THRESHOLD * 255
    )
    kps_1 = nms_points(kp_candidates, kp_scores, config.ROAD_NMS_RADIUS)
    # prioritize intersection points
    kp_candidates = np.concatenate([kps_0, kps_1], axis=0)
    kp_scores = np.concatenate(
        [np.ones((kps_0.shape[0])), np.zeros((kps_1.shape[0]))], axis=0
    )
    kps = nms_points(kp_candidates, kp_scores, config.ROAD_NMS_RADIUS)
    return kps


def extract_graph_astar(keypoint_mask, road_mask, config):
    """마스크에서 추출한 노드 후보들을 A* 경로 탐색으로 연결하여 networkx 그래프를 만든다.

    각 노드에 대해 NEIGHBOR_RADIUS 이내의 다른 노드들을 KDTree로 찾고, A*로 도로 위를
    지나는 경로가 존재하면 두 노드를 엣지로 연결한다. TopoNet 기반 추론(inferencer.py)의
    대안이 되는, 학습 없이 마스크만으로 그래프를 만드는 휴리스틱 방법이다.
    """
    kps = extract_graph_points(keypoint_mask, road_mask, config)

    # cost_field = create_cost_field(kps, road_mask)
    cost_field = create_cost_field_astar(kps, road_mask)
    viz_cost_field = np.array(cost_field)
    viz_cost_field[viz_cost_field == 0] = 255
    # cv2.imwrite('astar_cost_dbg.png', viz_cost_field)
    pathfinder = tcod.path.AStar(cost_field)

    tree = KDTree(kps)
    graph = nx.Graph()
    checked = set()
    for p in kps:
        # TODO: add radius to config
        neighbor_indices = tree.query_radius(
            p[np.newaxis, :], r=config.NEIGHBOR_RADIUS
        )[0]
        for n_idx in neighbor_indices:
            n = kps[n_idx]
            start, end = (int(p[0]), int(p[1])), (int(n[0]), int(n[1]))
            if (start, end) in checked:
                continue
            # if is_connected_bresenham(cost_field, p, n):
            if is_connected_astar(
                pathfinder, cost_field, p, n, max_path_len=config.NEIGHBOR_RADIUS
            ):
                graph.add_edge(start, end)
            checked.add((start, end))
    return graph


# takes xys
def visualize_image_and_graph(img, graph):
    # Draw nodes as green squares
    for node in graph.nodes():
        x, y = node
        cv2.rectangle(
            img, (int(x) - 2, int(y) - 2), (int(x) + 2, int(y) + 2), (0, 255, 0), -1
        )
    # Draw edges as white lines
    for start_node, end_node in graph.edges():
        cv2.line(
            img,
            (int(start_node[0]), int(start_node[1])),
            (int(end_node[0]), int(end_node[1])),
            (255, 255, 255),
            1,
        )
    return img


if __name__ == "__main__":

    # cost = np.array(
    #     [[1, 0, 1],
    #      [0, 1, 0],
    #      [0, 0, 0]],
    #      dtype=np.int32
    # )
    # pathfinder = tcod.path.AStar(cost)
    # print(pathfinder.get_path(0, 2, 0, 0))
    # cost[1, 1] = 0
    # print(pathfinder.get_path(0, 2, 0, 0))
    # cost[1, 1] = 1
    # print(pathfinder.get_path(0, 2, 0, 0))

    rgb_pattern = "./cityscale/20cities/region_{}_sat.png"
    keypoint_mask_pattern = "./cityscale/processed/keypoint_mask_{}.png"
    road_mask_pattern = "./cityscale/processed/road_mask_{}.png"

    index = 0
    rgb = read_rgb_img(rgb_pattern.format(index))
    road_mask = cv2.imread(road_mask_pattern.format(index), cv2.IMREAD_GRAYSCALE)
    keypoint_mask = cv2.imread(
        keypoint_mask_pattern.format(index), cv2.IMREAD_GRAYSCALE
    )

    graph = extract_graph_astar(keypoint_mask, road_mask)
    viz = visualize_image_and_graph(rgb, graph)
    cv2.imwrite("test_graph_astar_blk6_r40_m40_inms.png", viz)
