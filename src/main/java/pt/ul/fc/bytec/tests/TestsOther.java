package pt.ul.fc.bytec.tests;

public class TestsOther {
    public static int rec2(int a) {
        if (a > 1000)
            return Tests.rec1(a / 2);
        else return a;
    }

    public static int val;
}
