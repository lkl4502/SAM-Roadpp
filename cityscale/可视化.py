import json
import matplotlib.pyplot as plt

# 读取JSON文件
with open('/home/yinpan/sam_road/cityscale/20cities/region_0_gt_graph.json', 'r') as file:
    data = json.load(file)

# 假设图像尺寸为2000x2000像素
image_size = (2000, 2000)

# 提取各类别的坐标
overpass = data.get('overpass', [])
complicated_intersections = data.get('complicated_intersections', [])
parallel_road = data.get('parallel_road', [])

# 创建一个空白图像
fig, ax = plt.subplots(figsize=(10, 10))

# 绘制各类别的点
if overpass:
    overpass_x, overpass_y = zip(*overpass)
    ax.scatter(overpass_x, overpass_y, c='r', label='Overpass')

if complicated_intersections:
    ci_x, ci_y = zip(*complicated_intersections)
    ax.scatter(ci_x, ci_y, c='g', label='Complicated Intersections')

if parallel_road:
    pr_x, pr_y = zip(*parallel_road)
    ax.scatter(pr_x, pr_y, c='b', label='Parallel Road')

# 设置图像尺寸和标题
ax.set_xlim(0, image_size[0])
ax.set_ylim(0, image_size[1])
ax.set_title('Road Annotations Visualization')
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.legend()

# 显示图像
plt.gca().invert_yaxis()  # 反转y轴以匹配图像坐标系
plt.show()
