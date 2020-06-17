import unittest

import torch
import numpy as np

from falkon.tests.helpers import gen_random
from falkon.utils import decide_cuda

if decide_cuda({}):
    from falkon.utils import cuda_helpers
    from falkon.cuda import initialization


@unittest.skipIf(not decide_cuda({}), "No GPU found.")
class TestCopyToDeviceNoOrder(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        initialization.init({'compute_arch_speed': False, 'use_cpu': False})

    def test_all_F_contig(self):
        h_A = torch.from_numpy(gen_random(20, 20, 'float32', F=True))
        d_A = torch.from_numpy(gen_random(20, 20, 'float32', F=True)).cuda()

        copy_out = cuda_helpers.copy_to_device_noorder(10, 10, h_A, 5, 5, d_A, 5, 5)

        self.assertEqual(copy_out.data_ptr(), d_A[5,5].data_ptr())
        np.testing.assert_allclose(h_A[5:15,5:15].numpy(), d_A[5:15,5:15].cpu().numpy())

    def test_all_C_contig(self):
        h_A = torch.from_numpy(gen_random(20, 20, 'float32', F=False))
        d_A = torch.from_numpy(gen_random(20, 20, 'float32', F=False)).cuda()

        copy_out = cuda_helpers.copy_to_device_noorder(10, 10, h_A, 5, 5, d_A, 5, 5)

        self.assertEqual(copy_out.data_ptr(), d_A[5,5].data_ptr())
        np.testing.assert_allclose(h_A[5:15,5:15].numpy(), d_A[5:15,5:15].cpu().numpy())

    def test_mixed_contig(self):
        h_A = torch.from_numpy(gen_random(20, 20, 'float32', F=True))
        d_A = torch.from_numpy(gen_random(20, 20, 'float32', F=False)).cuda()

        copy_out = cuda_helpers.copy_to_device_noorder(10, 10, h_A, 5, 5, d_A, 5, 5)

        self.assertEqual(copy_out.data_ptr(), d_A[5,5].data_ptr())
        np.testing.assert_allclose(h_A[5:15,5:15].numpy(), d_A[5:15,5:15].cpu().numpy())
