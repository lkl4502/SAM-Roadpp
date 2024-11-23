import pickle
import matplotlib.pyplot as plt

# 读取 .p 文件
with open('/home/yinpan/sam_road/spacenet/data/AOI_2_Vegas_30__gt_graph_dense.p', 'rb') as file:
    data = pickle.load(file)

# 打印数据以检查其结构
print(data)

# 创建一个新的图形窗口
plt.figure(figsize=(10, 10))
# 绘制点
for point, connected_points in data.items():
    x, y = point
    # 交换x和y的值进行翻转
    x, y = y, x
    plt.scatter(x, y, c='b', marker='o')  # 绘制当前点
    for connected_point in connected_points:
        x_c, y_c = connected_point
        # 交换连接点的x和y的值进行翻转
        x_c, y_c = y_c, x_c
        plt.plot([x, x_c], [y, y_c], c='r')  # 绘制当前点与连接点之间的线段


# 显示图形
plt.grid(False)  # 显示网格
plt.show()
