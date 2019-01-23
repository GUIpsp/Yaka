package pt.ul.fc.yaka;

public class DeviceConfiguration {
    public final int xGrid;
    public final int yGrid;
    public final int zGrid;
    public final int xBlock;
    public final int yBlock;
    public final int zBlock;

    public DeviceConfiguration(int xGrid, int yGrid, int zGrid, int xBlock, int yBlock, int zBlock) {
        this.xGrid = xGrid;
        this.yGrid = yGrid;
        this.zGrid = zGrid;
        this.xBlock = xBlock;
        this.yBlock = yBlock;
        this.zBlock = zBlock;
    }
}
