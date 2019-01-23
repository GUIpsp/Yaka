package pt.ul.fc.yaka;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;
import org.objectweb.asm.tree.ClassNode;
import org.objectweb.asm.tree.FieldNode;
import org.objectweb.asm.tree.MethodNode;
import pt.ul.fc.bytec.ByteC;
import pt.ul.fc.bytec.ConversionConfiguration;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Method;

import static jcuda.driver.JCudaDriver.*;
import static pt.ul.fc.bytec.ByteC.mangle;

public class Yaka {

    public static void runGPU(int[] input, int[] output, Method func, DeviceConfiguration conf) {
        try {
            String name = func.getName();
            Class<?> declaringClass = func.getDeclaringClass();
            File cudaFile = new ByteC(new ConversionConfiguration() {
                @Override
                public String getModifiers(ClassNode clazz, MethodNode method) {
                    if (method.name.equals(name)) {
                        return "extern \"C\" __global__";
                    }
                    return "__device__";
                }

                @Override
                public String getExtension() {
                    return ".cu";
                }

                @Override
                public String getExtraHeaders() {
                    return "#include <cuda.h>\n";
                }

                @Override
                public String getFieldModifiers(ClassNode clazz, FieldNode method) {
                    return "__device__";
                }
            }).convertClass(declaringClass);
            File ptx = preparePtxFile(cudaFile);
            JCudaDriver.setExceptionsEnabled(true);
            // Initialize the driver and create a context for the first device.
            cuInit(0);
            CUdevice device = new CUdevice();
            cuDeviceGet(device, 0);
            CUcontext context = new CUcontext();
            cuCtxCreate(context, 0, device);

            // Load the ptx file.
            CUmodule module = new CUmodule();
            cuModuleLoad(module, ptx.getAbsolutePath());

            // Obtain a function pointer to the function.
            CUfunction function = new CUfunction();
            cuModuleGetFunction(function, module, mangle(declaringClass.getCanonicalName(), name));


            try (H2DPointer inPointer = new H2DPointer(input);
                 D2HPointer outPointer = new D2HPointer(output)) {
                Pointer params = Pointer.to(Pointer.to(inPointer.device), Pointer.to(outPointer.device));
                cuLaunchKernel(function, conf.xGrid, conf.yGrid, conf.zGrid, conf.xBlock, conf.yBlock, conf.zBlock, 0, null, params, null);
                System.out.println("after sync");
                System.out.flush();
                cuCtxSynchronize();
            }

        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private static class H2DPointer extends AutomaticPointer {

        protected H2DPointer(int[] host) {
            super(host);
            JCudaDriver.cuMemcpyHtoD(this.device, this.host, size);
        }
    }

    private static class D2HPointer extends AutomaticPointer {
        protected D2HPointer(int[] host) {
            super(host);
        }

        @Override
        protected void preClose() {
            JCudaDriver.cuMemcpyDtoH(this.host, this.device, size);
        }
    }

    private static abstract class AutomaticPointer implements AutoCloseable {
        boolean closed = false;
        public final CUdeviceptr device;
        public final Pointer host;
        public final int size;

        protected AutomaticPointer(int[] host) {
            this.size = host.length * Sizeof.INT;
            this.device = new CUdeviceptr();
            JCudaDriver.cuMemAlloc(device, size * 300);
            this.host = Pointer.to(host);
        }

        protected void preClose() {

        }

        @Override
        public final void close() {
            if (!closed) {
                preClose();
                JCudaDriver.cuMemFree(device);
                closed = true;
            }

        }
    }

    /**
     * The extension of the given file name is replaced with "ptx".
     * If the file with the resulting name does not exist, it is
     * compiled from the given file using NVCC. The name of the
     * PTX file is returned.
     *
     * @param cuFile The .CU file
     * @return The name of the PTX file
     * @throws IOException If an I/O error occurs
     */
    private static File preparePtxFile(File cuFile) throws IOException {
        String cuFileName = cuFile.getCanonicalFile().getName();
        int endIndex = cuFileName.lastIndexOf('.');
        if (endIndex == -1) {
            endIndex = cuFileName.length() - 1;
        }
        String ptxFileName = cuFileName.substring(0, endIndex + 1) + "ptx";
        File ptxFile = new File(ptxFileName);
        System.out.println("Path: " + ptxFile.getAbsolutePath());
        if (ptxFile.exists()) {
            return ptxFile;
        }

        if (!cuFile.exists()) {
            throw new IOException("Input file not found: " + cuFileName);
        }
        String modelString = "-m" + System.getProperty("sun.arch.data.model");
        String command = String.format("nvcc %s -ptx %s -o %s", modelString, cuFile.getPath(), ptxFileName);

        System.out.println("Executing\n" + command);
        Process process = Runtime.getRuntime().exec(command);

        String errorMessage =
                new String(toByteArray(process.getErrorStream()));
        String outputMessage =
                new String(toByteArray(process.getInputStream()));
        int exitValue = 0;
        try {
            exitValue = process.waitFor();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException(
                    "Interrupted while waiting for nvcc output", e);
        }

        if (exitValue != 0) {
            System.out.println("nvcc process exitValue " + exitValue);
            System.out.println("errorMessage:\n" + errorMessage);
            System.out.println("outputMessage:\n" + outputMessage);
            throw new IOException(
                    "Could not create .ptx file: " + errorMessage);
        }

        System.out.println("Finished creating PTX file");
        return ptxFile;
    }

    /**
     * Fully reads the given InputStream and returns it as a byte array
     *
     * @param inputStream The input stream to read
     * @return The byte array containing the data from the input stream
     * @throws IOException If an I/O error occurs
     */
    private static byte[] toByteArray(InputStream inputStream)
            throws IOException {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        byte[] buffer = new byte[8192];
        while (true) {
            int read = inputStream.read(buffer);
            if (read == -1) {
                break;
            }
            baos.write(buffer, 0, read);
        }
        return baos.toByteArray();
    }

}
