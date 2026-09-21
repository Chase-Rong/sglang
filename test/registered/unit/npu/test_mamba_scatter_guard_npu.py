"""Exercise scalar index loads next to genuinely unmapped NPU guard pages."""
import ctypes as C
import os

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get('SGLANG_FSA_TEST_ATTENTION') != '1',
                                reason='requires an explicitly selected idle NPU')


class Location(C.Structure):
    _fields_ = [('id', C.c_uint32), ('type', C.c_int)]


class Property(C.Structure):
    _fields_ = [('handleType', C.c_int), ('allocationType', C.c_int),
                ('memAttr', C.c_int), ('location', Location), ('reserve', C.c_uint64)]


class Guarded:
    def __init__(self, data, device, at_end=False):
        self.dtype = data.dtype
        self.device = torch.device(device)
        self.acl = C.CDLL('libascendcl.so')
        # CANN 9.1 rejects the legacy ACL_HBM_MEM attribute (value 1) for
        # virtual-memory physical allocations.  Use the supported huge-page
        # HBM attribute defined by acl_rt.h instead.
        self.prop = Property(0, 0, 4, Location(self.device.index, 1), 0)
        self.base, self.handle, gran = C.c_void_p(), C.c_void_p(), C.c_size_t()
        self.acl.aclrtMemGetAllocationGranularity.argtypes = [C.POINTER(Property), C.c_int, C.POINTER(C.c_size_t)]
        self.acl.aclrtReserveMemAddress.argtypes = [C.POINTER(C.c_void_p), C.c_size_t, C.c_size_t, C.c_void_p, C.c_uint64]
        self.acl.aclrtMallocPhysical.argtypes = [C.POINTER(C.c_void_p), C.c_size_t, C.POINTER(Property), C.c_uint64]
        self.acl.aclrtMapMem.argtypes = [C.c_void_p, C.c_size_t, C.c_size_t, C.c_void_p, C.c_uint64]
        self.acl.aclrtMemcpy.argtypes = [C.c_void_p, C.c_size_t, C.c_void_p, C.c_size_t, C.c_int]
        self.acl.aclrtUnmapMem.argtypes = [C.c_void_p]
        self.acl.aclrtFreePhysical.argtypes = [C.c_void_p]
        self.acl.aclrtReleaseMemAddress.argtypes = [C.c_void_p]
        assert self.acl.aclrtMemGetAllocationGranularity(C.byref(self.prop), 0, C.byref(gran)) == 0
        self.gran = gran.value
        assert self.acl.aclrtReserveMemAddress(C.byref(self.base), 3*self.gran, 0, None, 0) == 0
        assert self.acl.aclrtMallocPhysical(C.byref(self.handle), self.gran, C.byref(self.prop), 0) == 0
        self.mapped = self.base.value + self.gran
        assert self.acl.aclrtMapMem(self.mapped, self.gran, 0, self.handle, 0) == 0
        size = data.numel()*data.element_size()
        self.ptr = self.mapped + (self.gran-size if at_end else 0)
        assert self.acl.aclrtMemcpy(self.ptr, size, data.data_ptr(), size, 1) == 0

    def data_ptr(self):
        return self.ptr

    def close(self):
        assert self.acl.aclrtUnmapMem(self.mapped) == 0
        assert self.acl.aclrtFreePhysical(self.handle) == 0
        assert self.acl.aclrtReleaseMemAddress(self.base) == 0


@pytest.mark.parametrize('boundary', ['before_second_set', 'after_first_set', 'after_metadata'])
def test_conv_multi_scalar_guard(boundary):
    import torch_npu
    from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
        _conv_multi_build_meta, _fused_conv_window_scatter_multi_kernel,
    )
    device = f"npu:{int(os.environ['SGLANG_FSA_TEST_DEVICE_ID'])}"
    torch.npu.set_device(device)
    src = torch.arange(2*2*6*3*64, dtype=torch.int32).reshape(2,2,6,3,64).remainder(97).to(torch.bfloat16).to(device)
    dst = torch.zeros((2,4,3,64), dtype=torch.bfloat16, device=device)
    idx1 = torch.tensor([1],dtype=torch.int32,device=device)
    idx2 = torch.tensor([2],dtype=torch.int32,device=device)
    step1 = torch.tensor([3],dtype=torch.int32,device=device)
    step2 = torch.tensor([5],dtype=torch.int32,device=device)
    meta, blocks = _conv_multi_build_meta([(dst,src)],1024)
    original = {'before_second_set':step2, 'after_first_set':step1, 'after_metadata':meta}[boundary]
    guarded = Guarded(original.cpu(),device,at_end=boundary!='before_second_set')
    if boundary == 'before_second_set':step2 = guarded
    elif boundary == 'after_first_set':step1 = guarded
    else:meta = guarded
    print('guard',boundary,hex(guarded.data_ptr()),'page_bytes',guarded.gran,flush=True)
    _fused_conv_window_scatter_multi_kernel[(2,2,blocks)](
        meta,idx1,step1,idx2,step2,1,2,6,4,
        NUM_TYPES=1,META_COLS=12,BLOCK_SIZE=1024,
    )
    torch.npu.synchronize()
    expected = torch.zeros_like(dst,device='cpu')
    expected[:,1] = src.cpu()[:,0,3]
    expected[:,2] = src.cpu()[:,0,5]
    torch.testing.assert_close(dst.cpu(),expected,rtol=0,atol=0)
    guarded.close()
