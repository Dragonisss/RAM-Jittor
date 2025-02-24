import jittor as jt

class Net(jt.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = jt.nn.Conv2d(3, 3, 3,1,1)
        self.conv2 = jt.nn.Conv2d(3, 3, 3,1,1)
        self.conv3 = jt.nn.Conv2d(3, 3, 3,1,1)
    def execute(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

net = Net()
# net.eval()
class Hook_back_loop:
    """钩子类,用于捕获前向和后向信息"""
    def __init__(self, module, module_name):
        self.name = module_name
        self.hook = module.register_forward_hook(self.forward_hook)
    
    def forward_hook(self, module, inp, out):    
        self.input = inp
        self.output = out
        # jittor中需要手动设置grad_fn来捕获梯度
        self.output.register_hook(self.backward_hook)

    def backward_hook(self,grad):
        print(grad.shape,"in a")
        self.grad = grad
        return grad

hook_list = []
hook_list.append(Hook_back_loop(net.conv1, "conv1"))
hook_list.append(Hook_back_loop(net.conv2, "conv2"))
hook_list.append(Hook_back_loop(net.conv3, "conv3"))

x = jt.randn(1, 3, 10, 10)
y = net(x)

gt = jt.randn(1, 3, 10, 10)
l1 = jt.nn.L1Loss()
loss = l1(y, gt)

# 只反向传播计算梯度,不更新参数
optimizer = jt.optim.Adam(net.parameters(), lr=0.001)
optimizer.zero_grad()
optimizer.backward(loss)

# # 打印每一层的梯度
# for name, param in net.named_parameters():
#     print(f"{name} grad:", param.opt_grad(optimizer))


