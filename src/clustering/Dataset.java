package clustering;

/**
 * Created by rusland on 20.12.18.
 */
public enum Dataset {
    GLASS("data/glass.csv", 0, false, false, 214, 10, 6), DERMATOLOGY("data/dermatology.csv", 0, false, true, 358, 32, 6),
    FLAME("data/flame.csv", 0, false, true, 240, 2, 2), COMPOUND("data/compound.csv", 0, false, true, 399, 2, 6),
    SPIRAL("data/spiral.csv", 0, false, true, 312, 2, 3), PATHBASED("data/pathbased.csv", 0, false, true, 300, 2, 3),
    YEAST("data/yeast.csv", 0, true, true, 1484, 8, 10), WINE_QUALITY("data/wq.csv", 0, false, true, 4898, 11, 7),
    A3("data/a3.csv", 0, false, true, 7500, 2, 50), DIM064("data/dim064.csv", 0, false, true, 1024, 64, 16),
    DIM256("data/dim256.csv", 0, false, true, 1024, 256, 16);

    private String path;

    private int header;
    private boolean removeFirst, normalize;

    private int N;
    private int D;
    private int K;

    Dataset(String aPath, int aHeader, boolean aRemoveFirst, boolean aNormalize, int aN, int aD, int aK) {
        this.path = aPath;
        this.header = aHeader;
        this.removeFirst = aRemoveFirst;
        this.normalize = aNormalize;
        this.N = aN;
        this.D = aD;
        this.K = aK;
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

    public int getN() {
        return N;
    }

    public int getD() {
        return D;
    }

    public int getK() {
        return K;
    }

    public void setN(int n) {
        N = n;
    }

    public void setD(int d) {
        D = d;
    }

    public void setK(int k) {
        K = k;
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