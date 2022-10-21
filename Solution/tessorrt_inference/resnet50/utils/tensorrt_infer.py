import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import time

# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem
 
    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)
 
    def __repr__(self):
        return self.__str__()
 
def allocate_buffers(engine,max_batch_size=16):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        dims = engine.get_binding_shape(binding) 
        # print(dims) 
        if dims[0] == -1:
            assert(max_batch_size is not None)
            dims[0] = max_batch_size #动态batch_size适应
        
        #size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        size = trt.volume(dims) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        #print(dtype,size)
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype) #开辟出一片显存
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def do_inference_v2(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


def infer(engine, data):
    batch_size = data.shape[0]
    context = engine.create_execution_context()
    context.set_binding_shape(0, (batch_size, 3, 224, 224)) #这句非常重要！！！定义batch为动态维度
    inputs, outputs, bindings, stream = allocate_buffers(engine, max_batch_size=batch_size) #构建输入，输出，流指针

    np.copyto(inputs[0].host, data.ravel())
    t1 = time.time()
    result = do_inference_v2(context, bindings, inputs, outputs, stream)[0]
    t2 = time.time()
    result = np.reshape(result, [batch_size, 1000])
    
    return result, t2-t1