import numpy as np
from numpy.linalg import solve
import torch


# 定义一个简单的二分类网络，单元测试用
def my_net(x: torch.Tensor) -> int:
    if torch.sum(x) >= x.numel():
        return 1
    else:
        return 0


# 对原样本添加噪声，获取一个随机的对抗样本
def get_x_adv(x_o: torch.Tensor, net: torch.nn.Module) -> torch.Tensor:
    std = 1.0
    x_adv = x_o + torch.normal(0.0, std, x_o.shape)
    original_label = net(x_o)
    while net(x_adv) == original_label:
        std *= 2    # 若仍然不是对抗样本，则标准差不对增大(即噪声不断增大）
        x_adv = x_o + torch.normal(0.0, std, x_o.shape)
    return x_adv


# 根据原样本和对抗样本，获取u，不考虑数据类型的情况下x_o+u=x_adv
def get_u(x_o: torch.Tensor, x_adv: torch.Tensor) -> np.array:
    x_o_array = x_o.numpy().reshape(1, -1)
    x_adv_array = x_adv.numpy().reshape(1, -1)
    u = x_adv_array - x_o_array
    return u / np.linalg.norm(u)


# 将v添加到v_array中，构成一组正交的单位向量组，便于下一次找到新的v
def add_v(v_array: np.array, v: np.array) -> np.array:
    if v_array.shape[0] == 0:
        v_array = v.reshape((1, -1))
    else:
        v_array = np.append(v_array, v.reshape((1, -1)), axis=0)
    return v_array


# 获取一个垂直于vector_array中每一个向量的v
def get_orthogonal_vector(vector_array: np.array) -> np.array:
    shape = vector_array.shape

    # 如果vector_array不是满秩矩阵，则方程是欠定方程，解不唯一，为了获得一个确定的随机解，我们将待定系数全部置为1.0，这样是不失一般性的
    if shape[0] < shape[1]:
        for i in range(shape[1] - shape[0]):
            temp_array = np.zeros((1, shape[1]), dtype=np.float32)
            temp_array[0][shape[1] - i - 1] = 1.0
            vector_array = np.append(vector_array, temp_array, axis=0)
    elif shape[0] > shape[1]:
        raise Exception('Overdetermined problem! Cannot be solved')
    else:
        pass

    coefficient_matrix = np.mat(vector_array)
    constant_column = np.mat(np.zeros((shape[1], 1), dtype=np.float32))
    for i in range(shape[1] - shape[0]):
        constant_column[shape[1] - i - 1][0] = 1.0
    orthogonal_matrix = solve(coefficient_matrix, constant_column)
    orthogonal_vector = np.array(orthogonal_matrix)
    orthogonal_vector = orthogonal_vector.reshape((1, -1))
    return orthogonal_vector / np.linalg.norm(orthogonal_vector)


# 在一个维度下，获取最佳的对抗样本x_hat
def get_x_hat_in_2d(x_o: torch.Tensor, x_adv: torch.Tensor, u: np.array, v: np.array,
                    net: torch.nn.Module) -> torch.Tensor:
    x_o_numpy = x_o.numpy().reshape(1, -1)
    x_adv_numpy = x_adv.numpy().reshape(1, -1)
    original_label = net(x_o)
    d = np.linalg.norm(x_adv_numpy - x_o_numpy)

    # 二分法找到最佳的theta，进而找到最佳的x_hat
    left_theta = 0.0
    right_theta = np.pi / 2
    theta = (left_theta + right_theta) / 2
    x_hat_1_numpy = x_adv_numpy
    while right_theta - left_theta >= 1e-3:
        x_numpy = x_o_numpy + d * (u * np.cos(theta) + v * np.sin(theta)) * np.cos(theta)
        if original_label == net(torch.Tensor(x_numpy).view(x_o.shape)):
            right_theta = theta
            theta = (left_theta + right_theta) / 2
        else:
            left_theta = theta
            theta = (left_theta + right_theta) / 2
            x_hat_1_numpy = x_numpy

    left_theta = 0.0
    right_theta = np.pi / 2
    theta = (left_theta + right_theta) / 2
    x_hat_2_numpy = x_adv_numpy
    while right_theta - left_theta >= 1e-3:
        x_numpy = x_o_numpy + d * (u * np.cos(theta) - v * np.sin(theta)) * np.cos(theta)
        if original_label == net(torch.Tensor(x_numpy).view(x_o.shape)):
            right_theta = theta
            theta = (left_theta + right_theta) / 2
        else:
            left_theta = theta
            theta = (left_theta + right_theta) / 2
            x_hat_2_numpy = x_numpy

    # 对称的两个待定点，选取更近的那一个
    if np.linalg.norm(x_hat_1_numpy - x_o_numpy) < np.linalg.norm(x_hat_2_numpy - x_o_numpy):
        x_hat_numpy = x_hat_1_numpy
    else:
        x_hat_numpy = x_hat_2_numpy
    return torch.Tensor(x_hat_numpy).view(x_o.shape)


# 根据原样本x_o和预先训练好的判别器net，获取最佳的对抗样本x_hat
def get_x_hat(x_o: torch.Tensor, net: torch.nn.Module) -> torch.Tensor:
    x_adv = get_x_adv(x_o, net)
    x_hat = x_adv
    v_array = np.array([])

    while v_array.shape[0] < x_o.view(1, -1).shape[1]:  # v_array有非0解
        # 两次优化，可以得到一个2D平面上的最优解
        u = get_u(x_o, x_adv)
        if v_array.shape[0] == 0:
            v_array = add_v(v_array, u)
        else:
            v_array[0] = u
        v = get_orthogonal_vector(v_array)
        x_hat = get_x_hat_in_2d(x_o, x_adv, u, v, my_net)

        u_new = get_u(x_o, x_hat)
        v_new = u - np.dot(u, u_new.T) / np.dot(v, u_new.T) * v     # 见说明文档
        v_new = v_new / np.linalg.norm(v_new)

        x_hat = get_x_hat_in_2d(x_o, x_hat, u_new, v_new, my_net)
        x_adv = x_hat
        v_array = add_v(v_array, v)

    return x_hat
