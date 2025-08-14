import torch
from torch_geometric.data import Data
# 只有张量，没有业务含义的原始字符串，也没有 NetworkX 图对象。
graph_data  = torch.load('/Users/qinchao/work/AI/GNN/benchmark/TeleGraph/Data/TELECOM/telecom_graph.pt')
print(graph_data)

# 检查对象类型
if isinstance(graph_data, Data):
    print("PyG Data 对象检测成功!")
    print(f"节点特征形状: {graph_data.x.shape}")
    print(f"边索引形状: {graph_data.edge_index.shape}")
    print(f"边特征: {graph_data.edge_attr.shape if graph_data.edge_attr is not None else '无'}")
    print(f"可用属性: {graph_data.keys}")
'''
Data(
  x=[num_nodes, node_feat_dim],          # 节点特征矩阵
  edge_index=[2, num_edges],              # 边连接关系 (COO格式)
  edge_attr=[num_edges, edge_feat_dim],   # 边特征矩阵
  y=[num_nodes, 1] or [num_edges, 1],     # 预测目标 (流量/延迟等)
  pos=[num_nodes, 2],                     # 空间坐标 (经/纬度)
  time=[num_nodes, time_steps]            # 时序流量数据
)
'''



# 这是“原始全量信息”，包括节点/边的原始字符串 id、业务标签、时间戳等，方便做可视化、传统图算法或调试。
# gpickle文件是NetworkX 的序列化图文件，使用 pickle 保存。
import pickle
with open('/Users/qinchao/work/AI/GNN/benchmark/TeleGraph/Data/TeleGraph.gpickle','rb') as f:
    G = pickle.load(f)
print(type(G)) # <class 'networkx.classes.graph.Graph'>
print(G.number_of_nodes()) # 41143
print(G.number_of_edges()) # 41683

# 节点
num_nodes = G.number_of_nodes()
nodes_iter = G.nodes(data=True)        # 迭代器：(node, attr_dict)

# 边
num_edges = G.number_of_edges()
edges_iter = G.edges(data=True)   # 迭代器：(u, v, attr_dict)

# 举例打印前 3 个
for i, (n, d) in enumerate(nodes_iter):
    if i >= 3: break
    print("node", n, d)

for i, (u, v, d) in enumerate(edges_iter):
    if i >= 3: break
    print("edge", u, v, d)

# 过滤掉双下划线的方法，直观一点
members = [m for m in dir(G) if not m.startswith('_')]
print(members)

import pandas as pd
rows = []
for n, d in G.nodes(data=True):
    rows.append(d)

df = pd.DataFrame(rows)
print(df.head())     # 直接看前 5 行
df.to_csv('/Users/qinchao/work/AI/GNN/benchmark/TeleGraph/Data/nodes_all_attrs.csv', index=False)