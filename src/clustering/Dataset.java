package clustering;

/** Represent a dataset
 */
public enum Dataset {
    GLASS("data/glass.csv", 0, false, false, 214, 10, 6),
    WDBC("data/wdbc.csv", 0, true, true, 569, 30, 2),
    FLAME("data/flame.csv", 0, false, true, 240, 2, 2), COMPOUND("data/compound.csv", 0, false, true, 399, 2, 6),
    PATHBASED("data/pathbased.csv", 0, false, true, 300, 2, 3), JAIN("data/jain.csv", 0, false, true, 373, 2, 2),
    S1("data/s1r5.csv", 0, false, true, 5000, 2, 15), S3("data/s3r5.csv", 0, false, true, 1000, 2, 15),
    DIM064("data/dim064.csv", 0, false, true, 1024, 64, 16), DIM256("data/dim256.csv", 0, false, true, 1024, 256, 16);

    // indicates path to dataset file
    private String path;
    // indicates line number of the dataset header;
    private int header;
    // indicates whether to remove first attribute or not (true in case of WDBC dataset);
    private boolean removeFirst;
    // indicates whether to normalize the dataset (false for GLASS)
    private boolean normalize;

    private int N;
    private int D;
    private int K;

    public int[] getLabels() {
        return labels;
    }

    public void setLabels(int[] labels) {
        this.labels = labels.clone();
    }

    private int[] labels;

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

    public static void main(String[] args) {
        for (Dataset dataset: Dataset.values()) {
            System.out.println(dataset.name());
        }
    }
}