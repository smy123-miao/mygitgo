import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv  # 引用GCN网络作为模型
from torch_geometric.datasets import facebook  # 数据集使用facebook数据集

# 数据集加载
dataset = facebook.FacebookPagePage(root='D:/AStudy/phython/tgsocial/data_facebook')


# 网络定义
class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 32)
        self.conv2 = GCNConv(32, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


# 实例化网络
net = Net()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = net.to(device)
data = dataset[0]
print(data)

# 选取训练集以及测试集
data.train_mask = data.x[0:100, 4:8].long()
data.test_mask = data.x[100:200, 4:8].long()

# 选取优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# 网络训练
model.train()
for epoch in range(60):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

# 模型测试
model.eval()
_, pred = model(data).max(dim=1)
correct = ((pred[data.test_mask] == data.y[data.test_mask]).sum()) / dataset.num_classes
index = data.test_mask.long()
acc = float(correct / index.sum())
print(f'Accuracy: {acc:.4f}')
