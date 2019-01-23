package pt.ul.fc.yaka.tests;

import java.util.Random;

public class GraphGen {

    public static int GRAPH_SIZE = 3000;
    public static int INF = 0x1fffffff;

    static void generate_random_graph(int[] output) {
        int i, j;
        Random r = new Random(0xdadadada);
        for (i = 0; i < GRAPH_SIZE; i++) {
            for (j = 0; j < GRAPH_SIZE; j++) {
                if (i == j) {
                    output[i * GRAPH_SIZE + j] = 0;
                } else {
                    int rand = r.nextInt() % 40;
                    if (rand > 20) {
                        rand = INF;
                    }

                    output[i * GRAPH_SIZE + j] = rand;
                }
            }
        }
    }
}
