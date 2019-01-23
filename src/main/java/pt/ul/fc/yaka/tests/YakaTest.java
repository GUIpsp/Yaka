package pt.ul.fc.yaka.tests;

import pt.ul.fc.yaka.DeviceConfiguration;
import pt.ul.fc.yaka.Yaka;

import static pt.ul.fc.yaka.tests.GraphGen.GRAPH_SIZE;

public class YakaTest {
    public static void main(String[] args) throws NoSuchMethodException {
        int[] input = new int[]{5, 10, 20, 30, 40, 50};
        int[] output = new int[]{0, 0, 0, 0, 0};
        int[] graph = new int[GRAPH_SIZE * GRAPH_SIZE];
        int[] graphOutput = new int[GRAPH_SIZE * GRAPH_SIZE];
        GraphGen.generate_random_graph(graph);
        Yaka.runGPU(graph, graphOutput, YakaTestKernel.class.getDeclaredMethod("prescan_kern", int[].class, int[].class),
                new DeviceConfiguration(1, 1, 1, 1, 1, 1));
        for (int i : output) {
            System.out.println(i);
        }
    }
}
