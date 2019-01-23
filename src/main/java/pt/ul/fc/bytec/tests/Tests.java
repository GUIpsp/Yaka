package pt.ul.fc.bytec.tests;

import pt.ul.fc.bytec.CImplementation;

public class Tests {
    public static int fun(int a, int b) {
        return (a - b) / (b + a) + (a << b) + (b >> a) + (b >>> a);
    }

    public static int sub(int a) {
        return a - 2;
    }

    public static int ior(int a, int b) {
        return a | b;
    }

    public static int iand(int a, int b) {
        return a & b;
    }

    public static int ixor(int a, int b) {
        return a ^ b;
    }

    public static int max(int a, int b) {
        return a > b ? a : b;
    }

    public static int max3(int a, int b, int c) {
        int max = a > b ? a : b;
        return max > c ? max : c;
    }

    public static int roundToMultiple(int a, int b) {
        while (a % b != 0) {
            a++;
        }
        return a;
    }

    public static int popCount(int v) {
        int c; // c accumulates the total bits set in v
        for (c = 0; v != 0; c++) {
            v &= v - 1; // clear the least significant bit set
        }
        return c;
    }

//    public static double maxDouble(double a, double b) {
//        return a > b ? a : b;
//    }

//    public static long maxLong(long a, long b) {
//        return a > b ? a : b;
//    }

    public static int gcd2(int p, int q) {
        while (q != 0) {
            int temp = q;
            q = p % q;
            p = temp;
        }
        return p;
    }

    public static int gcd3(int p, int q, int r) {
        while (q != 0) {
            int temp = q;
            q = p % q;
            p = temp;
        }
        while (r != 0) {
            int temp = r;
            r = p % r;
            p = temp;
        }
        return p;
    }

    public static void tripleSum(int[] p, int[] q, int[] r, int size) {
        for (int i = 0; i < size; i++) {
            int p1 = p[i];
            int q1 = q[i];
            int r1 = r[i];
            p[i] = p1 + q1 + r1;
        }
    }

    public static void shouldUnroll(int[] p, int[] q, int[] r, int size) {
        final int N = 10;
        for (int i = 0; i < size - N; i++) {
            int acc = 0;
            for (int i1 = 0; i1 < N; i1++) {
                acc += q[i + i1] + r[i + i1];
            }
            p[i] += acc;
        }
    }

    public static int plusOne(int i) {
        return i + 1;
    }

    public static void addArray(int[] arr, int size) {
        for (int i = 0; i < size; i++) {
            arr[i] = plusOne(arr[i]);
        }
    }

    public static int rec1(int a) {
        return TestsOther.rec2(a + 1);
    }

    @CImplementation("printf(\"%d\", aI0);")
    public static void printNumber(int a) {
        System.out.printf("%d", a);
    }

    public static void increaseField(int a) {
        TestsOther.val++;
    }

    public static int getField() {
        return TestsOther.val;
    }

}
