package clustering;

/**
 * Created by rusland on 20.12.18.
 */
public enum Dataset {
    GLASS("data/glass.csv", 0, false, false), DERMATOLOGY("data/dermatology.csv",  0, false, true),
    FLAME("data/flame.csv", 0, false, true), COMPOUND("data/compound.csv", 0, false, true),
    SPIRAL("data/spiral", 0, false, true), PATHBASED("data/pathbased", 0, false, true),
    YEAST("data/yeast", 0, true, true);

    private String path;

    private int header;
    private boolean removeFirst, normalize;

    Dataset(String aPath, int aHeader, boolean aRemoveFirst, boolean aNormalize) {
        this.path = aPath;
        this.header = aHeader;
        this.removeFirst = aRemoveFirst;
        this.normalize = aNormalize;
    }

    public String getPath() {
        return path;
    }

    public boolean isRemoveFirst() {
        return removeFirst;
    }

    public boolean isNormalize() {
        return normalize;
    }

    public int getHeader() {
        return header;
    }

    public void setHeader(int header) {
        this.header = header;
    }

    public void setRemoveFirst(boolean removeFirst) {
        this.removeFirst = removeFirst;
    }

    public void setNormalize(boolean normalize) {
        this.normalize = normalize;
    }

    public void setPath(String path) {
        this.path = path;
    }
}