import matplotlib.pyplot as plt
import networkx as nx

# 创建一个有向图
G = nx.DiGraph()

# 添加节点
G.add_node("x1", pos=(0, 2))
G.add_node("x2", pos=(0, 1))
G.add_node("x3", pos=(0, 0))

G.add_node("h1", pos=(1, 1.5))
G.add_node("h2", pos=(1, 0.5))

G.add_node("y", pos=(2, 1))

# 添加边
G.add_edges_from([("x1", "h1"), ("x2", "h1"), ("x3", "h1"),
                  ("x1", "h2"), ("x2", "h2"), ("x3", "h2"),
                  ("h1", "y"), ("h2", "y")])

# 获取节点的位置
pos = nx.get_node_attributes(G, 'pos')

# 绘制节点
nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='lightblue')

# 绘制边
nx.draw_networkx_edges(G, pos, edgelist=G.edges(), arrows=True)

# 绘制节点标签
labels = {
    "x1": "$x_1$",
    "x2": "$x_2$",
    "x3": "$x_3$",
    "h1": r"$h_1 = \sigma(w_{11}x_1 + w_{12}x_2 + w_{13}x_3 + b_1)$",
    "h2": r"$h_2 = \sigma(w_{21}x_1 + w_{22}x_2 + w_{23}x_3 + b_2)$",
    "y": r"$y = w_{31}h_1 + w_{32}h_2 + b_3$"
}
nx.draw_networkx_labels(G, pos, labels, font_size=18)

# 添加层的标识
plt.text(0, 2.5, "Input Layer\n(3 neurons)", horizontalalignment='center', fontsize=12)
plt.text(1, 2.5, "Hidden Layer\n(2 neurons)", horizontalalignment='center', fontsize=12)
plt.text(2, 2.5, "Output Layer\n(1 neuron)", horizontalalignment='center', fontsize=12)

# 添加层的分隔线
plt.plot([0, 0], [-0.5, 2.5], color='gray', linestyle='')
plt.plot([1, 1], [-0.5, 2.5], color='gray', linestyle='')
plt.plot([2, 2], [-0.5, 2.5], color='gray', linestyle='')

plt.title("Shallow Neural Network")
plt.axis('off')
plt.show()

