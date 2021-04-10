import numpy as np 
import torch 

# x = np.arange(6).reshape(2,3,1)

# # l0norm = np.linalg.norm(x, ord=0) # norm 0: so luong cac phan tu khac 0
# # l1norm = np.linalg.norm(x, ord=1) # norm 1: mahattan distance
# # l2norm = np.linalg.norm(x, ord=2) # norm 2: euclit distance
# # print(l0norm)

# # print(l1norm)

# # print(l2norm)

# x = torch.from_numpy(x).float()
# # x = x**2
# # x = x.square()
# norm = x.pow(2).sum(dim=1, keepdims=True).sqrt() # keepdims = True: (batch, channels, height, width) --> (batch, 1, heights, width)
# x = x/norm

# weight = torch.nn.Parameter(torch.Tensor(3)).unsqueeze(0).unsqueeze(2).expand_as(x)

# print(x)

# print(weight)

# a = weight*x

# print(a, a.shape)

# boxes = torch.randn(5,4)
# x1 = boxes[:,0]
# x2 = boxes[:,1]
# x3 = boxes[:,2]
# x4 = boxes[:,3]

# y = x2.new()
# print(y)
# torch.index_select(x1, 0, torch.tensor([0,1]), out=y)

# print(y)

# index = torch.Tensor([0,1,2,3,4])
# num = torch.Tensor([3,4,5,-1,3])

# index = index[num>3] # 1,2

# print(index)

# x1 = torch.randn(2,4)
# x2 = torch.randn(2,4)

# x = torch.cat([x1,x2],1)

# x = x.view(2,-1,4)
# print(x.shape)

a = np.arange(10).reshape((2,5))
# print(a)
# print(np.max(a, axis=1))
# print(np.max(a, axis=0))

b = torch.from_numpy(a)
print(b)
print(b.dim())
# _, idxs = b.sort(1, descending=True)
# _, ranks = idxs.sort(1)

# print(idxs)
# print(ranks)