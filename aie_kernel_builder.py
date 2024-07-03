import npu
from npu.build.kernel import Kernel
from npu.build.appbuilder import AppBuilder
from npu.runtime import AppRunner
from npu.build.itkernel import ITWrite
import numpy as np
import ipulog

class AIEKernelBuilder_1in_1out(AppBuilder):
    def __init__(self, kernelfx, inarg:np.ndarray, outarg:np.ndarray):
        self.k = kernelfx
        self.inarg = inarg
        self.outarg = outarg
        self.stdout = np.zeros(shape=(64), dtype=np.uint32)
        self.k.stdout.array = np.zeros(shape=(64), dtype=np.uint32)
        self.app = None
        self.log = None
        super().__init__()

    def callgraph(self, inarg:np.ndarray, outarg:np.ndarray, stdout:np.ndarray,)->None:
        x, y = self.k(inarg)
        _ = ITWrite(x, bufref=outarg)
        _ = ITWrite(y, bufref=stdout)

    def build(self):
        super().build(self.inarg, self.outarg, self.stdout)
        self.log = ipulog.CreateIpuLogDecoder(self)
    
    def run(self, cache=True):
        if (not cache) or (self.log is None):
            self.build()
    
        if self.app is None:
            try:
                self.app = AppRunner('AIEKernelBuilder_1in_1out.xclbin') 
            except:
                self.build()
        
        inbuf = self.app.allocate(shape=(4096,), dtype=np.uint8)
        outbuf = self.app.allocate(shape=(4096,), dtype=np.uint8)
        stdout = self.app.allocate(shape=(64,), dtype=np.uint32)

        inbuf[:] = self.inarg
        
        inbuf.sync_to_npu()
        self.app.call(inbuf, outbuf, stdout)
        outbuf.sync_from_npu()
        stdout.sync_from_npu()
        
        print(f"Build completed, running...\n\n")
        msgs = self.log.decode(stdout)
        for m in msgs:
            print(m)
        return outbuf

class AIEKernelBuilder_2in_1out(AppBuilder):
    def __init__(self, kernelfx, inarg1:np.ndarray, inarg2:np.ndarray, outarg:np.ndarray):
        self.k = kernelfx
        self.inarg1 = inarg1
        self.inarg2 = inarg2
        self.outarg = outarg
        self.stdout = np.zeros(shape=(64), dtype=np.uint32)
        self.k.stdout.array = np.zeros(shape=(64), dtype=np.uint32)
        self.app = None
        self.log = None
        super().__init__()

    def callgraph(self, inarg1:np.ndarray, inarg2:np.ndarray, outarg:np.ndarray, stdout:np.ndarray,)->None:
        x, y = self.k(inarg1, inarg2)
        _ = ITWrite(x, bufref=outarg)
        _ = ITWrite(y, bufref=stdout)

    def build(self):
        super().build(self.inarg1, self.inarg2, self.outarg, self.stdout)
        self.log = ipulog.CreateIpuLogDecoder(self)
    
    def run(self, cache=True):
        if (not cache) or (self.log is None):
            self.build()
    
        if self.app is None:
            try:
                self.app = AppRunner('AIEKernelBuilder_2in_1out.xclbin') 
            except:
                self.build()
        
        inbuf1 = self.app.allocate(shape=(4096,), dtype=np.uint8)
        inbuf2 = self.app.allocate(shape=(4096,), dtype=np.uint8)
        outbuf = self.app.allocate(shape=(4096,), dtype=np.uint8)
        stdout = self.app.allocate(shape=(64,), dtype=np.uint32)

        inbuf1[:] = self.inarg1
        inbuf2[:] = self.inarg2
        
        inbuf1.sync_to_npu()
        inbuf2.sync_to_npu()
        self.app.call(inbuf1, inbuf2, outbuf, stdout)
        outbuf.sync_from_npu()
        stdout.sync_from_npu()
        
        print(f"Build completed, running...\n\n")
        msgs = self.log.decode(stdout)
        for m in msgs:
            print(m)
        return outbuf
