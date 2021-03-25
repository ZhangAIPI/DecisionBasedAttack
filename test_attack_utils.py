from unittest import TestCase
import numpy as np
from attack_utils import *
import torch


class Test(TestCase):
    def test_get_orthogonal_vector(self):
        vector_array = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
        orthogonal_vector = get_orthogonal_vector(vector_array)
        self.assertTrue(self, 1)
        for i in vector_array:
            temp = np.inner(i, orthogonal_vector)[0]
            self.assertAlmostEqual(temp, 0.0)

    def test_get_u(self):
        x_o = torch.Tensor([[1, 2], [3, 4]], dtype=torch.float32)
        x_adv = torch.Tensor([[5, 6], [7, 8]], dtype=torch.float32)
        u = get_u(x_o, x_adv)
        self.assertTrue((u == np.array([[4, 4, 4, 4]])).all())

    def test_add_v(self):
        v_array = np.array([[1, 2, 3, 4]])
        v = np.array([1, 2, 3, 4], dtype=np.float32)
        v_array = add_v(v_array, v)
        self.assertTrue(v_array.shape == (2, 4) and ((v_array == [[1, 2, 3, 4]])).all())

    def test_get_x_adv(self):
        x_o = torch.tensor([[2, 2]])
        x_adv = get_x_adv(x_o, my_net())
        self.assertTrue(1)

    def test_get_x_hat(self):
        x_o = torch.Tensor([[0.0], [0.0], [0.0]])
        x_adv = torch.Tensor([[1.0], [0.0], [3.0]])
        v_array = np.array([])

        while v_array.shape[0] < x_o.view(1, -1).shape[1]:
            u = get_u(x_o, x_adv)
            if v_array.shape[0] == 0:
                v_array = add_v(v_array, u)
            else:
                v_array[0] = u
            v = get_orthogonal_vector(v_array)
            x_hat = get_x_hat_in_2d(x_o, x_adv, u, v, my_net)

            u_new = get_u(x_o, x_hat)
            v_new = u - np.dot(u, u_new.T) / np.dot(v, u_new.T) * v
            v_new = v_new / np.linalg.norm(v_new)

            x_hat = get_x_hat_in_2d(x_o, x_hat, u_new, v_new, my_net)
            x_adv = x_hat
            v_array = add_v(v_array, v)

        self.assertTrue(1)

    def test_get_x_hat(self):
        x_o = torch.Tensor([[0.0], [0.0], [0.0]])
        x_hat = get_x_hat(x_o, my_net)
        print(x_hat)
        self.assertTrue(1)
