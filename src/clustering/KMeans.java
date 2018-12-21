package clustering;

import utils.Utils;

import java.io.IOException;
import java.util.*;

/**
 * Created by rusland on 10.11.18.
 * code is taken from http://cecs.wright.edu/~keke.chen/cloud/labs/mapreduce/KMeans.java
 * https://github.com/JasonAltschuler/KMeansPlusPlus/blob/master/src/KMeans.java
 * http://commons.apache.org/proper/commons-math/apidocs/org/apache/commons/math4/ml/clustering/KMeansPlusPlusClusterer.html#KMeansPlusPlusClusterer-int-
 */
public class KMeans {
    private boolean supplied;

    private int niter = 500;

    public KMeans() {
    }

    public enum Initialization {
        RANDOM, KMEANS_PLUS_PLUS, HILL_CLIMBER
    }

    private static int default_seed = 10;
    private static Initialization default_init = Initialization.KMEANS_PLUS_PLUS;

    private int[] label;
    private double[][] centroids;
    private int k;
    private int seed;
    private double pow = 2.0;
    private Random rnd;
    private Initialization initialization;
    private double[][] initialStartPoint;

    public KMeans(KMeans kMeans) {
        this.centroids = Utils.deepCopy(kMeans.getCentroids());
        this.setSeed(kMeans.getSeed());
        this.label = kMeans.getLabels().clone();
        this.k = kMeans.numberOfClusters();
        this.pow = kMeans.pow;
        this.initialization = kMeans.initialization;
    }

    public KMeans(int aK, double aPow) {
        this.k = aK;
        this.pow = aPow;
        this.seed = default_seed;
        this.initialization = default_init;
    }

    public void setInitializationMethod(Initialization aInitialization) {
        this.initialization = aInitialization;
    }

    public void setMaxIterations(int aIter) {
        this.niter = aIter;
    }

    public void setInitial(double[][] initial) {
        this.supplied = true;
        this.k = initial.length;
        this.initialStartPoint = Utils.deepCopy(initial);
    }

    /**
     * performs complete buildClusterer
     *  */
    public void buildClusterer(double[][] data) {
        double[][] copiedData = Utils.deepCopy(data);
        initialize(copiedData);
        int N = copiedData.length;
        assert (niter >= 1 || niter <= 500);

        double [][] c1 = centroids;
        double threshold = 0.000000001;
        int round=0;

        while (true) {
            // update _centroids with the last round results
            centroids = c1;

            //assign record to the clusterInstance centroid
            label = new int[N];
            for (int i = 0; i < N; i++) {
                label[i] = clusterInstance(copiedData[i]);
            }

            // recompute centroids based on the assignments
            c1 = updateCentroids(copiedData);
            round++;
            if (niter > 0 && round >= niter)
                break;
            if (converge(centroids, c1, threshold)) {
                //System.out.println("converged at: " + round);
                break;
            }
        }
        getRidOfEmptyCentroids();
    }

    private void getRidOfEmptyCentroids() {
        double[][] copy = Utils.deepCopy(this.centroids);
        Set<Integer> distLabels = Utils.distinctItems(this.label);
        int i = 0;
        this.centroids = new double[distLabels.size()][];
        HashMap<Integer, Integer> oldToNewIndex = new HashMap<>();
        for (int distLabel: distLabels) {
            this.centroids[i] = copy[distLabel];
            oldToNewIndex.put(distLabel, i);
            ++i;
        }
        for (i = 0; i < label.length; ++i) {
            label[i] = oldToNewIndex.get(label[i]);
        }
    }

    public int numberOfClusters() {
        return centroids.length;
    }

    public int[] getLabels() {
        return label;
    }

    public double[][] getCentroids() {
        return Utils.deepCopy(centroids);
    }

    /**
     * performs one iteration of k-means buildClusterer
     * */
    /*void oneIter() {
        centroids = updateCentroids();
        //assign record to the clusterInstance centroid
        for (int i=0; i < N; i++){
            label[i] = clusterInstance(data[i]);
        }
    }*/

    public String toString() {
        return "number of clusters: " + this.k;
    }

    private void initialize(double[][] data) {
        int N = data.length;
        int D = data[0].length;
        label = new int[N];
        assert (this.k < N);
        // choose existing data points as initial data points
        centroids = new double[k][D];

        if (this.initialization == Initialization.KMEANS_PLUS_PLUS) {
            int firstIdx = rnd.nextInt(N);
            List<Integer> centroidList = new ArrayList<>();
            boolean[] taken = new boolean[N];
            centroidList.add(firstIdx);
            taken[firstIdx] = true;

            final double[] minDistSquared = new double[N];
            // Initialize the elements.  Since the only point in resultSet is firstPoint,
            // this is very easy.
            for (int i = 0; i < N; i++) {
                if (i != firstIdx) { // That point isn't considered
                    double d = Utils.dist(data[firstIdx], data[i], pow);
                    minDistSquared[i] = d*d;
                }
            }

            while (centroidList.size() < k) {
                double distSqSum = 0.0;

                for (int i = 0; i < N; i++) {
                    if (!taken[i]) {
                        distSqSum += minDistSquared[i];
                    }
                }

                // Add one new data point as a center. Each point x is chosen with
                // probability proportional to D(x)2
                final double r = rnd.nextDouble() * distSqSum;
                // The index of the next point to be added to the resultSet.
                int nextPointIndex = -1;
                // Sum through the squared min distances again, stopping when
                // sum >= r.
                double sum = 0.0;
                for (int i = 0; i < N; i++) {
                    if (!taken[i]) {
                        sum += minDistSquared[i];
                        if (sum >= r) {
                            nextPointIndex = i;
                            break;
                        }
                    }
                }

                // If it's not set to >= 0, the point wasn't found in the previous
                // for loop, probably because distances are extremely small.  Just pick
                // the last available point.
                if (nextPointIndex == -1) {
                    for (int i = N - 1; i >= 0; i--) {
                        if (!taken[i]) {
                            nextPointIndex = i;
                            break;
                        }
                    }
                }
                // We found one.
                if (nextPointIndex >= 0) {
                    centroidList.add(nextPointIndex);
                    taken[nextPointIndex] = true;
                    if (centroidList.size() < k) {
                        for (int j = 0; j < N; j++) {
                            if (!taken[j]) {
                                double d = Utils.dist(data[nextPointIndex], data[j], 2.0);
                                double d2 = d * d;
                                if (d2 < minDistSquared[j]) {
                                    minDistSquared[j] = d2;
                                }
                            }
                        }
                    }
                } else {
                    break;
                }
            }

            assert (centroids.length == centroidList.size());
            for (int i = 0; i < centroidList.size(); ++i) {
                centroids[i] = data[centroidList.get(i)];
            }
        } else if (this.initialization == Initialization.RANDOM) {
            double[][] copy = Utils.deepCopy(data);
            for (int i = 0; i < k; i++) {
                int rand = rnd.nextInt(N - i);
                for (int j = 0; j < D; j++) {
                    centroids[i][j] = copy[rand][j];       // store chosen centroid
                    copy[rand][j] = copy[N - 1 - i][j];    // ensure sampling without replacement
                }
            }
            // assign data points to clusterInstance centroids
            for (int i = 0; i < N; i++) {
                label[i] = clusterInstance(data[i]);
            }
        } else if (this.initialization == Initialization.HILL_CLIMBER) {
            assert (supplied == true);
            this.centroids = this.initialStartPoint;
            this.k = this.initialStartPoint.length;
        }

        Utils.checkClusterLabels(getLabels(), k);
    }

    private double[][] updateCentroids(double[][] data) {
        int N = data.length;
        int D = data[0].length;
        // initialize centroids and set to 0
        double [][] newc = new double [k][]; //new centroids
        int [] counts = new int[k]; // sizes of the clusters

        // intialize
        for (int i=0; i<k; i++) {
            counts[i] =0;
            newc[i] = new double[D];
            for (int j=0; j<D; j++)
                newc[i][j] =0;
        }

        for (int i=0; i<N; i++){
            for (int j=0; j<D; j++){
                newc[label[i]][j] += data[i][j]; // update that centroid by adding the member data record
            }
            counts[label[i]]++;
        }

        // finally get the average
        for (int i=0; i< k; i++){
            for (int j=0; j<D; j++){
                newc[i][j] /= counts[i];
            }
        }

        return newc;
    }

    /**
     *  find the clusterInstance centroid for the record v
     *  */
    public int clusterInstance(double[] v){
        double mindist = Utils.dist(v, centroids[0], this.pow);
        int label = 0;
        for (int i = 1; i < k; i++) {
            double t = Utils.dist(v, centroids[i], this.pow);
            if (mindist > t) {
                mindist = t;
                label = i;
            }
        }
        return label;
    }

    /**
     * check convergence condition
     * max{dist(c1[i], c2[i]), i=1..numClusters < threshold
     * */
    private boolean converge(double [][] c1, double [][] c2, double threshold){
        // c1 and c2 are two sets of centroids
        double maxv = 0;
        for (int i = 0; i < k; i++){
            double d = Utils.dist(c1[i], c2[i], this.pow);
            if (maxv < d)
                maxv = d;
        }

        if (maxv < threshold)
            return true;
        else
            return false;
    }

    public static void main(String[] args) throws IOException {
        /*double[][] data;
        int k;
        int[] labels;
        int skipLines = 0;
        //data = new double[][]{{2,2}, {3,3}, {3,1}, {4,2},
        //        {1.6,-0.5}, {3.01, -1.5}, {-4, 2}, {-2, 2}, {-3, 3},{7,7}};
        //k = 4;
        //labels = {0,0,0,0,0,0,1,1,1,2};


        List<String[]> dataStr = Utils.readFile("data/glass.csv", ',');
        assert (dataStr.size()>0);

        labels = Utils.extractLabels(dataStr, dataStr.get(0).length-1);

        // exclude columns from csv file
        int[] excludedColumns = {0,dataStr.get(0).length-1};
        data = Utils.extractAttributes(dataStr, excludedColumns);
        k = data.length/20;
        clustering.KMeans kMeans;
        kMeans = new clustering.KMeans(data, data.length, data[0].length, k);
        kMeans.oneIter();
        //kMeans.buildClusterer(100);
        int[] labelsPred = kMeans.getLabels();

        //smile.buildClusterer.clustering.KMeans kMeans = new smile.buildClusterer.clustering.KMeans(data,k,100);
        //int[] labelsPred = kMeans.getClusterLabel();
        System.out.println(Arrays.toString(labelsPred));
        System.out.println(new AdjustedRandIndex().measure(labels, labelsPred));*/
    }

    public int getSeed() {
        return seed;
    }

    public void setSeed(int aSeed) {
        rnd = new Random(aSeed);
        this.seed = aSeed;
    }
}
