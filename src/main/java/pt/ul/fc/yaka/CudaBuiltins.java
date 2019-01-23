package pt.ul.fc.yaka;

import pt.ul.fc.bytec.CImplementation;

public class CudaBuiltins {
    @CImplementation("__syncthreads();")
    public static void __syncthreads() {
    }

    @CImplementation("return threadIdx.x;")
    public static int getThreadIdX_X() {
        return 1;
    }

    @CImplementation("return threadIdx.y;")
    public static int getThreadIdX_Y() {
        return 1;
    }

    @CImplementation("return threadIdx.z;")
    public static int getThreadIdX_Z() {
        return 1;
    }

    @CImplementation("return blockIdx.x;")
    public static int getBlockIdx_X() {
        return 1;
    }

    @CImplementation("return blockIdx.y;")
    public static int getBlockIdx_Y() {
        return 1;
    }

    @CImplementation("return blockIdx.z;")
    public static int getBlockIdx_Z() {
        return 1;
    }

    @CImplementation("return blockDim.x;")
    public static int getBlockDim_X() {
        return 1;
    }

    @CImplementation("return blockDim.y;")
    public static int getBlockDim_Y() {
        return 1;
    }

    @CImplementation("return blockDim.z;")
    public static int getBlockDim_Z() {
        return 1;
    }

    @CImplementation("return gridDim.x;")
    public static int getGridDim_X() {
        return 1;
    }

    @CImplementation("return gridDim.y;")
    public static int getGridDim_Y() {
        return 1;
    }

    @CImplementation("return gridDim.z;")
    public static int getGridDim_Z() {
        return 1;
    }
}
