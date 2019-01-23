package pt.ul.fc.yaka.tests;

import pt.ul.fc.bytec.CImplementation;
import pt.ul.fc.yaka.CudaBuiltins;


public class YakaTestKernel {

    public static void addOne(int[] input, int[] output) {
        int length = input[0];
        for (int i = 0; i < length; i++) {
            output[i] = input[i + 1] + 2;
        }
        CudaBuiltins.__syncthreads();
    }

    static void floyd_warshall_gpu_kern_entry(int[] input, int[] output) {
        floyd_warshall_gpu_kern(input);
    }

    static void floyd_warshall_gpu_kern(int[] output) {
        for (int k = 0; k < 3000; k++) {
            for (int i = CudaBuiltins.getBlockIdx_X() * CudaBuiltins.getBlockDim_X() + CudaBuiltins.getThreadIdX_X();
                 i < 3000;
                 i += CudaBuiltins.getBlockDim_X() * CudaBuiltins.getGridDim_X()) {
                for (int j = CudaBuiltins.getBlockIdx_Y() * CudaBuiltins.getBlockDim_Y() + CudaBuiltins.getThreadIdX_Y();
                     j < 3000;
                     j += CudaBuiltins.getBlockDim_Y() * CudaBuiltins.getGridDim_Y()) {
                    if (output[i * 3000 + k] + output[k * 3000 + j] < output[i * 3000 + j]) {
                        output[i * 3000 + j] = output[i * 3000 + k] + output[k * 3000 + j];
                    }
                }
            }
            sync_k(k);
        }
    }

    static void sync_k(int k) {
        if (CudaBuiltins.getThreadIdX_X() == 0) {
            atomicAddArrived(1);
            while (arrived != (k + 1) * CudaBuiltins.getBlockDim_X() * CudaBuiltins.getBlockDim_Y()) {
            }
        }
        CudaBuiltins.__syncthreads();
    }

    public static int arrived;

    @CImplementation("atomicAdd(&pt_DOT_ul_DOT_fc_DOT_yaka_DOT_tests_DOT_YakaTestKernel_AT_arrived, aI0);")
    public static void atomicAddArrived(int nr) {
        arrived += nr;
    }

    static void prescan_kern(int[] input, int[] output) {
        /* taken from https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch39.html */
        /* Input[0] contains the length */
        int n = input[0];
        int[] temp = output;  //todo should be an extern allocated on invokation
        int thid = CudaBuiltins.getThreadIdX_X();
        int offset = -1;
        /* Increase index by one on input since the first element of input contains n */
        temp[thid * 2] = input[2 * thid + 1];
        temp[thid * 2 + 1] = input[2 * thid + 2];
        for (int d = n >> 1; d > 0; d >>= 1) {
            CudaBuiltins.__syncthreads();
            if (thid < d) {
                int ai = offset * (2 * thid + 1) - 1;
                int bi = offset * (2 * thid + 2) - 1;
                temp[bi] += temp[ai];
            }
            offset *= 2;
        }
        if (thid == 0) {
            temp[n - 1] = 0; //clear last element
        }
        for (int d = 1; d < n; d *= 2) { //transverse down the tree and build scan
            offset >>= 1;
            CudaBuiltins.__syncthreads();
            if (thid < d) {
                int ai = offset * (2 * thid + 1) - 1;
                int bi = offset * (2 * thid + 2) - 1;
                int t = temp[ai];
                temp[ai] = temp[bi];
                temp[bi] += t;
            }
        }
        CudaBuiltins.__syncthreads();
        output[2 * thid] = temp[2 * thid];
        output[2 * thid + 1] = temp[2 * thid + 1];
    }
}
