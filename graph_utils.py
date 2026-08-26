"""
도로 그래프(노드-엣지 구조)를 다루기 위한 유틸리티 모음.

크게 세 가지 용도로 쓰인다.
1. 그래프 정제: 고립 노드 제거, 인접 노드 병합, 너무 긴 엣지 분할 등 (`remove_isolate_nodes`,
   `merge_nodes`, `split_edges`, `merge_into_large_graph`)
2. 포맷 변환: Sat2Graph 포맷(dict) <-> (nodes, edges) 배열 <-> networkx/igraph 그래프
   상호 변환 (`convert_to_sat2graph_format`, `convert_from_sat2graph_format`,
   `convert_from_nx`, `igraph_from_adj_dict`)
3. 학습용 그래프 라벨 생성 보조: 그래프 세분화(subdivide), 교차점(crossover) 탐색,
   포인트 NMS, BFS 탐색 등 (`subdivide_graph`, `find_crossover_points`, `nms_points`,
   `bfs_with_conditions`) - dataset.py의 GraphLabelGenerator에서 사용된다.
"""

import rtree
import scipy
import unittest
import numpy as np
import igraph as ig
import networkx as nx
import matplotlib.pyplot as plt

from collections import deque
from shapely.geometry import LineString
from sklearn.cluster import DBSCAN
from shapely.geometry import Point, LineString
from shapely.strtree import STRtree

# from scipy import interpolate
# from matplotlib.colors import ListedColormap


def inspect_graph(node_array, edge_array):
    """디버깅용: 단방향 엣지 개수와 중복(거의 같은 위치) 노드 개수를 출력한다.

    Args:
        node_array (np.ndarray): [N_node, 2] 노드 좌표.
        edge_array (np.ndarray): [N_edge, 2] (src_idx, dst_idx) 쌍.
    """
    edge_set = set()
    for edge in edge_array:
        src, dst = edge[0], edge[1]
        edge_set.add((src, dst))
    one_way_edge_count = 0
    for src, dst in edge_set:
        if (dst, src) not in edge_set:
            one_way_edge_count += 1
    print(f"DEBUG: One-way-edge count {one_way_edge_count}")

    node_dist_matrix = node_array[:, np.newaxis, :] - node_array[np.newaxis, :, :]
    node_dist_matrix = np.sum(node_dist_matrix**2, axis=-1) ** 0.5
    node_num = node_array.shape[0]
    pair_is_close = node_dist_matrix < 0.1
    duplicate_node_count = (np.sum(pair_is_close.astype(int)) - node_num) / 2

    print(f"DEBUG: duplicate_node_count: {duplicate_node_count}")


def filter_nodes(node_array, edge_array, keep_node):
    """keep_node 마스크로 노드를 필터링하고, 삭제된 노드에 연결된 엣지도 함께 제거한다.

    노드 인덱스가 재배열되므로 edge_array의 인덱스도 새 인덱스로 갱신한다.

    Args:
        node_array (np.ndarray): [N_node, 2] 노드 좌표.
        edge_array (np.ndarray): [N_edge, 2] (src_idx, dst_idx) 쌍.
        keep_node (np.ndarray): [N_node,] 남길 노드를 표시하는 bool 마스크.

    Returns:
        tuple: (필터링된 노드 배열, 인덱스가 갱신된 엣지 배열)
    """
    new_nodes = node_array[keep_node, :]
    old_node_num = node_array.shape[0]
    keep_indices = np.where(keep_node)[0]
    new_node_num = keep_indices.shape[0]
    old_to_new_indices = np.full((old_node_num,), fill_value=-1, dtype=np.int32)
    old_to_new_indices[keep_indices] = np.arange(
        start=0, stop=new_node_num, step=1, dtype=np.int32
    )
    # Replaces node indices in edge_array
    edge_nodes = edge_array.flatten()
    new_edge_nodes = old_to_new_indices[edge_nodes]
    new_edges = new_edge_nodes.reshape(-1, 2)
    # Filters disconnected edge
    keep_edge = np.all(new_edges > -1, axis=-1)
    new_edges = new_edges[keep_edge, :]
    return new_nodes, new_edges


# NOTE: 아래 두 함수는 이름이 같은 edge_list_to_adj_table이며, 두 번째 정의가 첫 번째를 덮어써서
# 실제로는 (nodes, edges)를 받는 버전만 사용된다. 첫 번째 버전은 nodes 인자 없이 edge에 등장한
# 인덱스로부터 노드 개수를 추정하던 예전 버전으로 보인다 (죽은 코드지만 참고용으로 남아있음).
def edge_list_to_adj_table(edges):
    # edges: [[src_idx, dst_idx], ...] node indices must start from 0 and
    # be continuous.
    # Returns:
    # adj_table: list of sets. len(adj_table) = num_nodes, adj_table[i]
    # = neighbor node indices of node i. Empty if no neighbors.
    nodes = set()
    for edge in edges:
        start_idx, end_idx = edge[0], edge[1]
        nodes.add(start_idx)
        nodes.add(end_idx)
    node_num = len(nodes)
    adj_table = [set() for i in range(node_num)]
    for edge in edges:
        start_idx, end_idx = edge[0], edge[1]
        adj_table[start_idx].add(end_idx)
    return adj_table


def edge_list_to_adj_table(nodes, edges):
    """엣지 리스트를 인접 리스트(adjacency list)로 변환한다.

    Args:
        nodes: 노드 목록 (개수만 사용됨).
        edges: [[src_idx, dst_idx], ...] 노드 인덱스는 0부터 시작하는 연속된 정수여야 한다.

    Returns:
        list[set]: adj_table[i] = 노드 i의 이웃 노드 인덱스 집합 (단방향, src -> dst만 기록됨).
    """
    node_num = len(nodes)
    adj_table = [set() for i in range(node_num)]
    for edge in edges:
        start_idx, end_idx = edge[0], edge[1]
        adj_table[start_idx].add(end_idx)
    return adj_table


def trace_segment(start_edge, adj_table):
    """시작 엣지에서 출발하여, 차수(degree)가 2인 노드를 계속 따라가며 하나의 폴리라인(segment)을 추적한다.

    분기점(차수!= 1인 미방문 이웃이 여러 개이거나 0개)을 만나면 추적을 멈춘다.

    Args:
        start_edge (tuple): 추적을 시작할 (start_node, next_node) 쌍.
        adj_table (list[set]): edge_list_to_adj_table로 만든 인접 리스트.

    Returns:
        list[int]: 추적된 segment를 구성하는 노드 인덱스 순서 리스트.
    """
    segment_nodes = [start_edge[0], start_edge[1]]
    visited_nodes = set(segment_nodes)
    while True:
        curr_node = segment_nodes[-1]
        unvisited_neighbor_num = 0
        next_node = -1
        for neighbor in adj_table[curr_node]:
            if neighbor not in visited_nodes:
                unvisited_neighbor_num += 1
                next_node = neighbor
        if unvisited_neighbor_num != 1:
            break
        segment_nodes.append(next_node)
        visited_nodes.add(next_node)
    return segment_nodes


def unique_edge(src, dst):
    """방향에 상관없이 동일한 엣지를 같은 키로 취급하기 위해 (작은 인덱스, 큰 인덱스) 순으로 정규화."""
    return (min(src, dst), max(src, dst))


def find_segments_in_road_graph(adj_table):
    """도로 그래프를 교차점(차수 != 2인 노드) 사이의 폴리라인 segment 목록으로 분해한다.

    각 교차점/끝점 노드에서 시작해 아직 방문하지 않은 엣지를 하나씩 따라가며
    trace_segment로 segment를 추적한다. 고립된 루프(모든 노드의 차수가 2인 폐곡선)는
    어느 교차점에서도 시작되지 않으므로 별도 경고를 출력한다.

    Args:
        adj_table (list[set]): edge_list_to_adj_table로 만든 도로 그래프의 인접 리스트.

    Returns:
        list[list[int]]: segments[i] = i번째 segment를 구성하는 노드 인덱스 리스트.
    """
    segments = list()
    visited_edges = set()
    # Goes over each edge in the graph.
    node_num = len(adj_table)
    for node in range(node_num):
        # See if node is a segment end point.
        if len(adj_table[node]) == 2:
            continue
        # Trace down an unvisited edge.
        for neighbor in adj_table[node]:
            edge = unique_edge(node, neighbor)
            if edge in visited_edges:
                continue
            # Needs edge direction for correct tracing.
            segment = trace_segment((node, neighbor), adj_table)
            for i in range(len(segment) - 1):
                visited_edge = unique_edge(segment[i], segment[i + 1])
                visited_edges.add(visited_edge)
            segments.append(segment)

    all_unique_edges = set()
    for node in range(node_num):
        for neighbor in adj_table[node]:
            all_unique_edges.add(unique_edge(node, neighbor))
    total_edge_num = len(all_unique_edges)
    if len(visited_edges) < total_edge_num:
        diff = total_edge_num - len(visited_edges)
        print(f"!!! Warning: Isolated loop detected. {diff} edges are missing.")

    return segments


def normalize_segments(coords, segments):
    """각 segment의 방향을 정규화한다. x가 더 작은 끝점이 먼저 오도록(같으면 y가 작은 쪽) 뒤집는다.

    같은 segment를 항상 동일한 방향으로 표현하기 위한 정규화이며, 이후 폴리라인 리샘플링이나
    비교 연산 시 방향 모호성을 없애기 위해 사용된다.

    Args:
        coords (np.ndarray): [N_node, 2] 노드 좌표.
        segments (list[list[int]]): find_segments_in_road_graph의 결과.

    Returns:
        list[list[int]]: 방향이 정규화된 segment 리스트.
    """
    normalized_segments = []
    for i in range(len(segments)):
        segment = segments[i]
        first = coords[segment[0], :]
        last = coords[segment[-1], :]

        if first[0] > last[0] or (first[0] == last[0] and first[1] > last[1]):
            segment = segment[::-1]

        normalized_segments.append(segment)

    return normalized_segments


def get_resampled_polylines(coords, segments, num_points):
    """각 segment(폴리라인)를 shapely LineString으로 만든 뒤, 호(arc length) 기준으로
    num_points개 지점으로 균등 리샘플링한다.

    Args:
        coords (np.ndarray): [N_node, 2] 노드 좌표.
        segments (list[list[int]]): segment를 구성하는 노드 인덱스 리스트들.
        num_points (int): 리샘플링할 포인트 개수.

    Returns:
        list[np.ndarray]: 각 원소가 [num_points, 2] 형태인 리샘플링된 폴리라인 리스트.
    """

    resampled_polylines = []

    for segment in segments:
        polyline_coords = coords[segment]
        polyline = LineString(polyline_coords)

        # Uniform parameter values
        dists = np.linspace(0, polyline.length, num_points)

        # Resample polyline
        resampled_polyline = np.array(
            [list(polyline.interpolate(d).coords)[0] for d in dists]
        )

        resampled_polylines.append(resampled_polyline)

    return resampled_polylines


def get_polylines_from_road_graph(coords, edges, num_points_per_segment):
    """도로 그래프(노드+엣지)를 교차점 단위로 끊은 폴리라인 리스트로 변환하는 파이프라인 함수.

    edge_list_to_adj_table -> find_segments_in_road_graph -> normalize_segments ->
    get_resampled_polylines 순서로 적용한다.
    """
    adj_table = edge_list_to_adj_table(edges)
    segments = find_segments_in_road_graph(adj_table)
    segments = normalize_segments(coords, segments)
    polylines = get_resampled_polylines(coords, segments, num_points_per_segment)
    return polylines


def get_polyline_connectivity(polylines, dist_threhsold):
    """폴리라인들의 끝점끼리 거리가 가까우면 서로 연결된 것으로 간주하여 연결 관계를 찾는다.

    Args:
        polylines (list[np.ndarray]): 각 원소가 [N_points, 2]인 폴리라인 리스트.
        dist_threhsold (float): 이 거리보다 가까운 끝점 쌍은 연결된 것으로 판단.

    Returns:
        tuple:
            connected_pairs (list[tuple]): (src_idx, dst_idx) 폴리라인 인덱스 쌍. 역방향 쌍도 포함됨.
            connected_point_indices (list[tuple]): 각 폴리라인에서 겹치는 끝점의 인덱스(0 또는 마지막).
    """
    connected_pairs = []
    connected_point_indices = []
    polyline_num = len(polylines)
    for i in range(polyline_num):
        for j in range(i + 1, polyline_num):
            a, b = polylines[i], polylines[j]
            endpoint_indices = [
                (0, 0),
                (0, b.shape[0] - 1),
                (a.shape[0] - 1, 0),
                (a.shape[0] - 1, b.shape[0] - 1),
            ]
            for a_idx, b_idx in endpoint_indices:
                if np.linalg.norm(a[a_idx] - b[b_idx]) < dist_threhsold:
                    connected_pairs.append((i, j))
                    connected_pairs.append((j, i))
                    connected_point_indices.append((a_idx, b_idx))
                    connected_point_indices.append((b_idx, a_idx))
    return connected_pairs, connected_point_indices


def visualize_polylines(image, polylines):
    # 디버깅/시각화용: 각 폴리라인을 서로 다른 색으로 이미지 위에 그려서 matplotlib으로 표시.
    # image: [H, W, C]
    # polylines: list of [length, 2] float arrays, each entry a (row, col)
    # tuple in pixel coordinates.

    # Generate a color map with as many colors as there are polylines
    cmap = plt.cm.get_cmap("hsv", len(polylines))

    # Display the image
    plt.imshow(image)

    # Draw each polyline with a different color
    for idx, polyline in enumerate(polylines):
        plt.plot(polyline[:, 1], polyline[:, 0], color=cmap(idx), linewidth=2)

    plt.show()


def visualize_polyline_graph(
    image, polylines, connected_pairs, connected_point_indices
):
    # 디버깅용: get_polyline_connectivity로 찾은 연결 쌍을 하나씩 순서대로(빨강->초록) 시각화.
    for pair_idx, (pair, endpoints) in enumerate(
        zip(connected_pairs, connected_point_indices)
    ):
        print(f"pair {pair_idx+1}/{len(connected_pairs)}")
        plt.imshow(image)
        idx_a, idx_b = pair
        line_a, line_b = polylines[idx_a], polylines[idx_b]
        plt.plot(line_a[:, 1], line_a[:, 0], color="red", linewidth=2)
        plt.plot(line_b[:, 1], line_b[:, 0], color="green", linewidth=2)
        end_a, end_b = line_a[endpoints[0], :], line_b[endpoints[1], :]
        plt.plot(end_a[1], end_a[0], marker="o", markersize=8, color="blue")
        plt.plot(end_b[1], end_b[0], marker="o", markersize=8, color="blue")
        plt.show()


## Utils for aggregating the large map.
# 아래 4개 함수는 추론 시 패치 단위로 예측한 그래프 조각들을 이어붙여
# 하나의 큰 지도로 합칠 때(merge_into_large_graph) 사용된다.
def remove_isolate_nodes(nodes, edges):
    """어떤 엣지에도 연결되지 않은 고립 노드를 제거하고, 남은 노드로 인덱스를 재정렬한다."""
    node_indices = np.arange(nodes.shape[0])
    graph = nx.Graph()
    graph.add_nodes_from(node_indices)
    graph.add_edges_from(edges)

    isolated_nodes = list(nx.isolates(graph))
    graph.remove_nodes_from(isolated_nodes)

    remaining_node_indices = list(graph.nodes())
    remaining_node_indices.sort()
    remaining_nodes = nodes[remaining_node_indices, :]

    new_graph = nx.convert_node_labels_to_integers(graph)
    new_edges = list(new_graph.edges())

    return remaining_nodes, new_edges


def merge_nodes(nodes, edges, distance_threshold):
    """DBSCAN으로 distance_threshold 이내에 몰려있는 노드들을 하나의 클러스터로 묶고,
    클러스터 중심 좌표로 병합한다. 패치 경계에서 겹치는 노드를 하나로 합칠 때 사용.

    같은 클러스터로 합쳐진 두 노드를 잇는 엣지(자기 자신으로의 엣지)는 제거된다.

    Args:
        nodes (np.ndarray): [N_node, 2] 노드 좌표.
        edges (list[tuple]): (src_idx, dst_idx) 엣지 리스트.
        distance_threshold (float): 같은 클러스터로 묶을 거리 기준.

    Returns:
        tuple: (클러스터 중심 좌표 배열, 병합된 유일한 엣지 리스트)
    """
    clustering = DBSCAN(eps=distance_threshold, min_samples=1).fit(nodes)
    node_cluster_indices = clustering.labels_
    num_clusters = len(np.unique(node_cluster_indices))
    cluster_centers = np.zeros((num_clusters, 2), dtype=np.float32)
    cluster_size = np.zeros((num_clusters,), dtype=np.float32)
    for node_index, node in enumerate(nodes):
        cluster_index = node_cluster_indices[node_index]
        cluster_centers[cluster_index, :] += node
        cluster_size[cluster_index] += 1
    cluster_centers = cluster_centers / cluster_size[:, np.newaxis]
    unique_edges = set()
    for start, end in edges:
        new_start = node_cluster_indices[start]
        new_end = node_cluster_indices[end]

        # Removes self-loops
        if new_start == new_end:
            continue

        new_edge = (min(new_start, new_end), max(new_start, new_end))
        unique_edges.add(new_edge)
    return cluster_centers, list(unique_edges)


def split_edges(nodes, edges, distance_threshold):
    """엣지 근처(distance_threshold 이내)에 다른 노드가 있으면, 그 노드를 경유하도록
    엣지를 둘로 쪼갠다. 인접 패치에서 온 노드가 엣지 중간을 가로막고 있을 때
    두 그래프 조각을 자연스럽게 이어 붙이기 위해 사용한다.

    STRtree로 각 엣지 주변(buffer)의 후보 노드를 찾고, 가장 가까운 노드를 경유점으로 삼아
    큐 기반으로 재귀적으로(더 쪼갤 게 없을 때까지) 분할한다.

    Args:
        nodes (np.ndarray): [N_node, 2] 노드 좌표.
        edges (list[tuple]): (src_idx, dst_idx) 엣지 리스트.
        distance_threshold (float): 엣지와 노드 사이, 분할을 트리거하는 거리 기준.

    Returns:
        tuple: (노드 배열 그대로, 분할되어 갱신된 유일한 엣지 리스트)
    """
    points = [Point(x, y) for x, y in nodes]
    point_tree = STRtree(points)

    edge_queue = deque()
    for edge in edges:
        edge_queue.appendleft(edge)

    new_edges = list()

    while len(edge_queue) > 0:
        start, end = edge_queue.pop()
        start_pt, end_pt = nodes[start, :], nodes[end, :]
        line_segment = LineString([start_pt, end_pt])
        nearby_region = line_segment.buffer(
            distance=distance_threshold, cap_style="flat"
        )
        nearby_point_indices = point_tree.query(nearby_region).tolist()
        min_dist = distance_threshold + 88.8
        nearest_point_index = None
        for index in nearby_point_indices:
            if index == start or index == end:
                continue
            point = points[index]
            dist = line_segment.distance(point)
            if dist < min_dist:
                min_dist, nearest_point_index = dist, index

        if nearest_point_index is None or min_dist >= distance_threshold:
            new_edges.append((start, end))
            continue
        else:
            e1, e2 = (start, nearest_point_index), (nearest_point_index, end)
            edge_queue.appendleft(e1)
            edge_queue.appendleft(e2)

    # TODO(congrui): share the edge dedup logic
    unique_edges = set()
    for start, end in new_edges:
        new_edge = (min(start, end), max(start, end))
        unique_edges.add(new_edge)

    return nodes, list(unique_edges)


def combine_graphs(graphs):
    """여러 (nodes, edges) 그래프 조각을 노드 인덱스 오프셋을 적용해 하나로 이어붙인다(병합 없이 단순 합집합).

    Args:
        graphs (list[tuple]): [(nodes, edges), ...] 형태의 그래프 조각 리스트.

    Returns:
        tuple: (합쳐진 노드 배열, 인덱스가 오프셋된 엣지 배열)
    """
    # graphs: list of (nodes, edges)
    offset = 0
    combined_nodes, combined_edges = [], []
    for nodes, edges in graphs:
        combined_nodes.append(nodes)
        edges_np = np.array(edges)
        edges_np += offset
        combined_edges.append(edges_np)
        offset += nodes.shape[0]
    combined_nodes = np.concatenate(combined_nodes, axis=0)
    combined_edges = np.concatenate(combined_edges, axis=0)
    return combined_nodes, combined_edges


def merge_into_large_graph(
    nodes, edges, merge_node_dist_thresh, split_edge_dist_thresh
):
    """여러 패치에서 예측된 그래프 조각들을 하나의 정합된 큰 그래프로 정리하는 전체 파이프라인.

    순서: 고립 노드 제거 -> 가까운 노드 병합 -> 엣지가 다른 노드를 가로지르면 분할
    -> 다시 고립 노드 제거. combine_graphs로 합친 직후에 호출하는 것을 전제로 한다.
    """
    nodes1, edges1 = remove_isolate_nodes(nodes, edges)
    nodes2, edges2 = merge_nodes(
        nodes1, edges1, distance_threshold=merge_node_dist_thresh
    )
    nodes3, edges3 = split_edges(
        nodes2, edges2, distance_threshold=split_edge_dist_thresh
    )
    nodes4, edges4 = remove_isolate_nodes(nodes3, edges3)
    return nodes4, edges4


def convert_to_sat2graph_format(nodes, edges):
    """(nodes, edges) 배열 표현을 Sat2Graph 라벨 포맷(dict)으로 변환한다.

    Args:
        nodes (np.ndarray): [N_node, 2] (row, col) 이미지 좌표.
        edges (np.ndarray): [N_edge, 2] (start, end) 노드 인덱스 쌍.

    Returns:
        dict: 키는 각 노드의 (row, col) 좌표(실수는 반올림), 값은 이웃 노드들의 (row, col) 리스트.
        무방향 그래프이므로 입력 엣지에 역방향 엣지를 추가하여 양쪽 모두 기록한다.
    """
    reverse_edges = edges[:, ::-1]
    all_edges = np.concatenate((edges, reverse_edges), axis=0)

    adj_table = edge_list_to_adj_table(nodes, all_edges)

    int_nodes = [(round(x), round(y)) for x, y in nodes]

    result = dict()
    for node_idx, neighbor_indices in enumerate(adj_table):
        # Notice, we expect the input graph has gone through node-merging so
        # there shouldn't be two nodes at the same pixel location.
        key = int_nodes[node_idx]
        value = [int_nodes[neighbor_idx] for neighbor_idx in neighbor_indices]
        result[key] = value
    return result


def convert_from_sat2graph_format(graph):
    """Sat2Graph 라벨 포맷(dict)을 (nodes, edges) 배열 표현으로 변환한다. convert_to_sat2graph_format의 역변환.

    Args:
        graph (dict): 키는 각 노드의 (row, col) 좌표, 값은 이웃 노드들의 (row, col) 리스트.

    Returns:
        tuple:
            nodes (np.ndarray): [N_node, 2] (row, col) 이미지 좌표.
            edges (list[tuple]): (start, end) 노드 인덱스 쌍. 무방향이며 중복 제거는 하지 않는다.
    """
    node_to_idx = dict()
    for node, neighbors in graph.items():
        if node not in node_to_idx.keys():
            node_to_idx[node] = len(node_to_idx)
        for neighbor in neighbors:
            if neighbor not in node_to_idx.keys():
                node_to_idx[neighbor] = len(node_to_idx)

    edges = list()
    for node, neighbors in graph.items():
        for neighbor in neighbors:
            src_idx, dst_idx = node_to_idx[node], node_to_idx[neighbor]
            edges.append((src_idx, dst_idx))

    num_nodes = len(node_to_idx)
    nodes = [None] * num_nodes
    for node, idx in node_to_idx.items():
        nodes[idx] = node
    return np.array(nodes), edges


def convert_from_nx(graph):
    """networkx 그래프(노드가 (x, y) 튜플)를 (nodes, edges) 배열 표현으로 변환한다.
    좌표는 (x, y) -> (row, col)로 뒤바뀐다.

    Args:
        graph (nx.Graph): 노드가 (x, y) 좌표 튜플인 networkx 그래프.

    Returns:
        tuple:
            nodes (np.ndarray): [N_node, 2] (row, col) 이미지 좌표.
            edges (np.ndarray): [N_edge, 2] (start, end) 노드 인덱스 쌍.
    """
    node_to_idx = dict()
    nodes = list()
    edges = list()
    for node in graph.nodes():
        if node not in node_to_idx.keys():
            node_to_idx[node] = len(node_to_idx)
        x, y = node
        nodes.append((y, x))  # to rc
    for node_0, node_1 in graph.edges():
        edges.append((node_to_idx[node_0], node_to_idx[node_1]))

    return np.array(nodes), np.array(edges)


### igraph utils for performance
# 아래 함수들은 dataset.py의 GraphLabelGenerator에서 학습용 그래프 라벨을 만들 때 쓰인다.
# python 순정 구조 대신 igraph를 쓰는 이유는 대규모 그래프 연산 속도 때문.


def igraph_from_adj_dict(graph, coord_transform):
    """Sat2Graph 포맷의 adjacency dict를 igraph 그래프 객체로 변환한다 (엣지는 중복 제거됨).

    Args:
        graph (dict): Sat2Graph 포맷의 adjacency dict ((row, col) -> [(row, col), ...]).
        coord_transform (callable): [N, 2] 좌표 배열을 받아 (예: (r,c) -> (x,y)) 변환하는 함수.
            변환된 좌표가 각 정점의 "point" 속성으로 저장된다.

    Returns:
        ig.Graph: 정점 속성 "point"에 변환된 좌표가 담긴 igraph 그래프.
    """
    nodes, edges = convert_from_sat2graph_format(graph)
    n_vertices = nodes.shape[0]
    if n_vertices == 0:
        nodes = np.zeros((0, 2), dtype=nodes.dtype)
    edges = set([(min(src, tgt), max(src, tgt)) for src, tgt in edges])
    g = ig.Graph(n_vertices, list(edges))
    try:
        g.vs["point"] = coord_transform(nodes)  # to xy
    except Exception:
        print("==================")
        print(nodes.shape)
        print(nodes)
        import pdb

        pdb.set_trace()
    return g


def get_line_bbox(line):
    """선분 (x0,y0)-(x1,y1)을 감싸는 바운딩 박스를 1픽셀 여유를 두고 계산한다 (rtree 질의용)."""
    (x0, y0), (x1, y1) = line
    l = min(x0, x1) - 1
    b = min(y0, y1) - 1
    r = max(x0, x1) + 1
    t = max(y0, y1) + 1
    return (l, b, r, t)


def find_intersection(segment1, segment2):
    """
    Finds the intersection point of two line segments, if it exists.

    Parameters:
        segment1 (tuple): A tuple representing the first line segment ((x1, y1), (x2, y2)).
        segment2 (tuple): A tuple representing the second line segment ((x3, y3), (x4, y4)).

    Returns:
        A tuple (x, y) representing the intersection point, or None if there is no intersection.
    """
    (x1, y1), (x2, y2) = segment1
    (x3, y3), (x4, y4) = segment2
    line1 = LineString([segment1[0], segment1[1]])
    line2 = LineString([segment2[0], segment2[1]])

    # Check for intersection
    intersection = line1.intersection(line2)

    if not intersection.is_empty and intersection.geom_type == "Point":
        if not (
            intersection.equals(Point(x1, y1))
            or intersection.equals(Point(x2, y2))
            or intersection.equals(Point(x3, y3))
            or intersection.equals(Point(x4, y4))
        ):
            return (intersection.x, intersection.y)
    # geom_type could be line if two parallel lines overlap
    # or just no intersection
    # or intersection is at endpoints
    return None


def find_crossover_points(graph):
    """도로 그래프에서 서로 다른 두 엣지가 (교차점 노드 없이) 시각적으로 겹쳐 지나가는
    교차 지점(예: 입체교차로처럼 실제로는 연결되지 않은 두 도로가 겹쳐 보이는 곳)을 찾는다.

    각 엣지의 바운딩 박스를 rtree에 등록해 후보 쌍만 빠르게 찾아 교차 여부를 검사한다.
    NOTE: A가 B와 교차하는 경우와 B가 A와 교차하는 경우를 각각 세므로 같은 교차점이
    두 번 카운트될 수 있다 (현재는 문제없이 사용 중이라 이대로 둠).

    Args:
        graph (ig.Graph): igraph_from_adj_dict로 만든, "point" 속성을 가진 igraph 그래프.

    Returns:
        list[tuple]: 교차점 (x, y) 좌표 리스트.
    """
    # takes igraph
    # y axis shall point upwards for rtree to work properly
    # crossover points are counted twice: A cross B, B cross A
    # - which is fine for now just be aware
    points = graph.vs["point"]
    edges = graph.es
    lines = [(points[edge.source], points[edge.target]) for edge in edges]
    line_bboxes = [get_line_bbox(line) for line in lines]
    line_index = rtree.index.Index()
    for idx, bbox in enumerate(line_bboxes):
        line_index.insert(idx, bbox) # MBR (window 생성)

    crossover_points = []
    tested_pairs = set()
    for i, line_0 in enumerate(lines):
        bbox = line_bboxes[i]
        nearby_indices = list(line_index.intersection(bbox))
        for ni in nearby_indices:
            pair = (min(i, ni), max(i, ni))
            if pair in tested_pairs:
                continue
            line_1 = lines[ni]
            itsc = find_intersection(line_0, line_1)
            if itsc is not None:
                crossover_points.append(itsc)
            tested_pairs.add(pair)

    return crossover_points


def subdivide_graph(graph, resolution):
    """각 엣지를 resolution 간격으로 세분화하여 원본 엣지 위에 새로운 정점들을 추가한 그래프를 만든다.

    긴 도로 엣지를 짧은 구간들로 잘게 쪼개어, 이후 point-wise한 라벨 샘플링/NMS/최근접
    탐색(KD-tree) 등이 도로 전체에 걸쳐 촘촘하게 이뤄지도록 하기 위함이다
    (더 많은 점을 찍어 촘촘한 그래프를 만들기 위함).

    Args:
        graph (ig.Graph): "point" 속성을 가진 igraph 그래프.
        resolution (float): 세분화 간격(픽셀 단위). 엣지 길이를 이 값으로 나눈 몫만큼 조각낸다.

    Returns:
        ig.Graph: 원본 정점 + 세분화로 추가된 정점을 모두 포함하는 새 그래프.
    """
    new_points = [p for p in graph.vs["point"]]
    new_edges = []
    for edge in graph.es:
        p0, p1 = graph.vs["point"][edge.source], graph.vs["point"][edge.target]
        length = np.linalg.norm(p1 - p0)
        sample_pieces = max(1, int(length / resolution))
        # [N, ]
        samples = np.linspace(0.0, 1.0, sample_pieces + 1, endpoint=True)
        # [N, 2] = [1, 2] + [N, 1] @ [1, 2]
        sampled_pts = np.expand_dims(np.array(p0), axis=0) + np.expand_dims(
            samples, axis=1
        ) @ np.expand_dims(p1 - p0, axis=0)
        # [N-2, 2]
        sampled_pts = sampled_pts[1:-1, :]  # 원래 그래프에 있던 부분은 제외
        new_point_indices = []
        for new_pt in sampled_pts:
            new_point_indices.append(len(new_points))
            new_points.append(new_pt)
        new_edges_sources = [edge.source] + new_point_indices
        new_edges_targets = new_point_indices + [edge.target]
        new_edges += list(zip(new_edges_sources, new_edges_targets))

    new_graph = ig.Graph(len(new_points), new_edges)
    new_graph.vs["point"] = np.array(new_points)
    return new_graph


def nms_points(points, scores, radius, return_indices=False):
    """점 집합에 대한 Non-Maximum Suppression. score가 높은 점부터 순서대로 채택하고,
    이미 채택된 점 반경(radius) 이내의 다른 점들은 억제(제거)한다.

    score가 1.0을 초과하는 점은 (교차점처럼 항상 유지해야 하는 점) 다른 점에 의해
    억제되더라도 강제로 유지된다.

    Args:
        points (np.ndarray): [N, 2] 후보 점 좌표.
        scores (np.ndarray): [N,] 각 점의 점수. 높을수록 우선 채택.
        radius (float): 억제 반경. 이 거리 이내의 낮은 점수 점들은 제거됨.
        return_indices (bool): True면 원본 points 배열 기준 인덱스도 함께 반환.

    Returns:
        np.ndarray 또는 (np.ndarray, np.ndarray): 채택된 점들의 좌표 (그리고 선택적으로 원본 인덱스).
    """
    # if score > 1.0, the point is forced to be kept regardless
    sorted_indices = np.argsort(scores)[::-1]
    sorted_points = points[sorted_indices, :]
    sorted_scores = scores[sorted_indices]
    kept = np.ones(sorted_indices.shape[0], dtype=bool)
    tree = scipy.spatial.KDTree(sorted_points)
    for idx, p in enumerate(sorted_points):
        if not kept[idx]:
            continue
        # neighbor_indices = tree.query_radius(p[np.newaxis, :], r=radius)[0]
        neighbor_indices = tree.query_ball_point(p, r=radius)  # 인접 node indice 반환
        neighbor_scores = sorted_scores[neighbor_indices]
        keep_nbr = np.greater(neighbor_scores, 1.0)  # neighbor_scores > 1.0
        kept[neighbor_indices] = keep_nbr
        kept[idx] = True
    if return_indices:
        return sorted_points[kept], sorted_indices[kept]
    else:
        return sorted_points[kept]


def bfs_with_conditions(graph, start_node, stop_nodes, max_depth):
    """
    igraph 그래프에서 start_node로부터 BFS(너비 우선 탐색)를 수행한다.
    stop_nodes에 포함된 노드를 방문하거나 깊이가 max_depth에 도달하면 그 지점에서 더 이상
    확장하지 않는다. dataset.py에서 두 점이 그래프상으로 max_depth(=이웃 탐색 반경) 이내에
    실제로 연결되어 있는지(topology label) 판정할 때 사용된다.

    Args:
    - graph (ig.Graph): 탐색할 그래프.
    - start_node (int): BFS 시작 노드 인덱스.
    - stop_nodes (set): 방문 시 탐색을 멈출 노드 인덱스 집합.
    - max_depth (int): 최대 탐색 깊이.

    Returns:
    - set: 방문한 노드 인덱스 집합 (멈춘 stop_nodes 포함).
    """
    visited = set()  # To keep track of visited nodes
    queue = deque()
    queue.append((start_node, 0))  # Queue of (node, depth)

    while queue:
        current_node, current_depth = (
            queue.popleft()
        )  # Dequeue the next node and its depth

        # Mark node as visited
        visited.add(current_node)

        # Check if the current node is a stop node or if the current depth exceeds max_depth
        if current_node in stop_nodes or current_depth >= max_depth:
            # Stop condition met, do not extend
            continue

        # Get neighbors and enqueue them with incremented depth, considering all edges
        neighbors = graph.neighbors(current_node, mode="all")
        for neighbor in neighbors:
            if neighbor not in visited:
                queue.append((neighbor, current_depth + 1))

    return visited


##### Unit tests #####
class TestGraphUtils(unittest.TestCase):
    def test_remove_isolated_nodes(self):
        nodes = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        edges = [[0, 2]]
        new_nodes, new_edges = remove_isolate_nodes(nodes, edges)
        gt_new_nodes = np.array([[0.0, 0.0], [2.0, 2.0]])
        gt_new_edges = np.array([[0, 1]])
        np.testing.assert_array_equal(new_nodes, gt_new_nodes)
        np.testing.assert_array_equal(np.array(new_edges), gt_new_edges)

    def test_merge_nodes(self):
        nodes = np.array([[0.0, 0.0], [1.0, 1.0], [1.1, 1.1], [2.0, 2.0], [0.1, 0.1]])
        edges = [[0, 1], [1, 2], [1, 3], [2, 3], [2, 4]]
        new_nodes, new_edges = merge_nodes(nodes, edges, 0.2)
        gt_new_nodes = np.array([[0.05, 0.05], [1.05, 1.05], [2.0, 2.0]])
        gt_new_edges = np.array([[0, 1], [1, 2]])
        np.testing.assert_almost_equal(new_nodes, gt_new_nodes)
        np.testing.assert_array_equal(np.array(new_edges), gt_new_edges)

    def test_split_edges(self):
        nodes = np.array([[0.0, 0.0], [1.01, 1.01], [2.0, 2.0], [2.0, 0.0]])
        edges = [[0, 1], [1, 2], [0, 2], [2, 3]]
        new_nodes, new_edges = split_edges(nodes, edges, 0.2)
        gt_new_nodes = nodes
        gt_new_edges = np.array([[0, 1], [1, 2], [2, 3]])
        np.testing.assert_almost_equal(new_nodes, gt_new_nodes)
        np.testing.assert_array_equal(np.array(new_edges), gt_new_edges)

    def test_combine_graphs(self):
        nodes0 = np.array([[0.0, 0.0], [1.0, 0.0]])
        edges0 = [[0, 1]]
        nodes1 = np.array([[2.0, 2.0], [3.0, 3.0]])
        edges1 = [[0, 1]]
        new_nodes, new_edges = combine_graphs([(nodes0, edges0), (nodes1, edges1)])
        gt_new_nodes = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 2.0], [3.0, 3.0]])
        gt_new_edges = np.array([[0, 1], [2, 3]])
        np.testing.assert_almost_equal(new_nodes, gt_new_nodes)
        np.testing.assert_array_equal(np.array(new_edges), gt_new_edges)

    def test_buffer_region(self):
        start_pt, end_pt = np.array([0.0, 0.0]), np.array([10.0, 0.0])
        line_segment = LineString([start_pt, end_pt])
        nearby_region = line_segment.buffer(distance=2.0, cap_style="flat")

        # Get the vertices of the polygon as a list of tuples
        vertices_list = list(nearby_region.exterior.coords)
        # Convert the list of tuples to a NumPy array
        vertices_array = np.array(vertices_list)

        gt_vertices = np.array(
            [[10.0, 2.0], [10.0, -2.0], [0.0, -2.0], [0.0, 2.0], [10.0, 2.0]]
        )
        np.testing.assert_almost_equal(vertices_array, gt_vertices)

    def test_convert_to_sat2graph_format(self):
        nodes = np.array([[0.0, 0.0], [1.1, 1.1], [1.6, 1.6]])
        edges = np.array([[0, 1], [1, 2]])
        result = convert_to_sat2graph_format(nodes, edges)
        gt_result = {(0, 0): [(1, 1)], (1, 1): [(0, 0), (2, 2)], (2, 2): [(1, 1)]}
        for k, v in result.items():
            self.assertTrue(k in gt_result.keys())
            self.assertSetEqual(set(v), set(gt_result[k]))

    def test_convert_from_sat2graph_format(self):
        graph = {(0, 0): [(1, 1)], (1, 1): [(0, 0), (2, 2)], (2, 2): [(1, 1)]}
        nodes, edges = convert_from_sat2graph_format(graph)
        gt_nodes = np.array([[0, 0], [1, 1], [2, 2]])
        gt_edges = np.array([[0, 1], [1, 0], [1, 2], [2, 1]])
        np.testing.assert_almost_equal(nodes, gt_nodes)
        np.testing.assert_almost_equal(np.array(edges), gt_edges)

    def test_convert_from_nx(self):
        graph = nx.Graph()
        graph.add_edge((1, 2), (3, 4))
        graph.add_edge((3, 4), (5, 6))
        nodes, edges = convert_from_nx(graph)
        gt_nodes = np.array([[2, 1], [4, 3], [6, 5]])
        gt_edges = np.array([[0, 1], [1, 2]])
        np.testing.assert_almost_equal(nodes, gt_nodes)
        np.testing.assert_almost_equal(edges, gt_edges)

    def test_igraph_from_sat2graph_format(self):
        adj = {
            (1, 2): [(3, 4), (5, 6)],
            (3, 4): [(1, 2), (5, 6)],
        }
        rc2xy = lambda x: x[:, ::-1]
        g = igraph_from_adj_dict(adj, rc2xy)
        self.assertEqual(len(g.es), 3)
        self.assertEqual(len(g.vs), 3)
        self.assertEqual(g.vs[0]["point"][0], 2)
        self.assertEqual(g.vs[0]["point"][1], 1)

    def test_find_crossover_points(self):
        adj = {
            (0, 1): [
                (10, 1),
            ],
            (2, -2): [
                (2, 10),
            ],
            (10, 1): [
                (20, 1),
            ],
        }
        rc2xy = lambda x: x[:, ::-1]
        g = igraph_from_adj_dict(adj, rc2xy)
        pts = find_crossover_points(g)
        self.assertEqual(len(pts), 1)
        gt = np.array([1.0, 2.0])
        pd = np.array(pts[0])
        np.testing.assert_almost_equal(gt, pd)

    def test_subdivide_graph(self):
        adj = {
            (0, 0): [
                (10, 0),
            ],
            (10, 0): [
                (20, 0),
            ],
        }
        rc2xy = lambda x: x[:, ::-1]
        g = igraph_from_adj_dict(adj, rc2xy)
        g1 = subdivide_graph(g, resolution=2.0)
        self.assertEqual(len(g1.vs["point"]), 11)
        self.assertEqual(len(g1.es), 10)


if __name__ == "__main__":
    unittest.main()
