package clustering;

import utils.Utils;

import java.io.IOException;
import java.util.*;

/**
 * Implementation of K-Means clustering algorithm with different centroid initialization methods: Random, K-Means++, Hill-climb
 *  */
public class KMeans {
    public enum Initialization {
        RANDOM, KMEANS_PLUS_PLUS, HILL_CLIMBER
    }

    private static int maxIters = 500;
    private static double threshold = 0.005;
    private static int default_seed = 10;
    private static double default_distMeasure = 2.0;
    private static Initialization default_init = Initialization.KMEANS_PLUS_PLUS;

    // in case of hill-climbing - whether centroids are supplied or not
    private boolean supplied;
    // number of iterations
    private int numIters = 50;
    private int[] labels;
    private double[][] centroids;
    private int k;
    private int seed;

    // distance measure, e.g. Manhattan, Euclidean
    private double distMeasure;
    private Random rnd;
    private Initialization initialization;
    private double[][] initialStartPoint;

    public KMeans() {
        this.distMeasure = default_distMeasure;
        this.seed = default_seed;
        this.rnd = new Random(seed);
        this.initialization = default_init;
    }


    public KMeans(KMeans kMeans) {
        this.centroids = Utils.deepCopy(kMeans.getCentroids());
        this.labels = kMeans.getLabels().clone();
        this.k = kMeans.numberOfClusters();
        this.distMeasure = kMeans.distMeasure;
        this.initialization = kMeans.initialization;
    }

    public KMeans(int aK, double aPow) {
        this.k = aK;
        this.distMeasure = aPow;
        this.seed = default_seed;
        this.initialization = default_init;
    }

    public void buildClusterer(double[][] data) throws Exception {
        double[][] copiedData = Utils.deepCopy(data);
        initialize(copiedData);
        int N = copiedData.length;
        assert (numIters >= 1 && numIters <= maxIters);

        double [][] prevCentroids = Utils.deepCopy(centroids);
        int[] prevLabels = this.labels.clone();
        int round = 0;

        while (round < numIters) {
            // recompute centroids based on the assignments
            centroids = updateCentroids(copiedData);
            //assign record to the clusterInstance centroid
            labels = new int[N];
            for (int i = 0; i < N; i++) {
                labels[i] = clusterInstance(copiedData[i]);
            }

            double curSSE = utils.Utils.sse(centroids, labels, data);
            double prevSSE = utils.Utils.sse(prevCentroids, prevLabels, data);
            double diff = Math.abs(curSSE - prevSSE);
            if (initialization == Initialization.HILL_CLIMBER && diff < threshold) {
                //System.out.println("converged at: " + round);
                break;
            }
            round++;
            prevCentroids = Utils.deepCopy(centroids);
            prevLabels = labels.clone();
        }
        getRidOfEmptyCentroids();
    }

    /**
     *  find the clusterInstance centroid for the record v
     *  */
    public int clusterInstance(double[] v){
        double mindist = Utils.dist(v, centroids[0], this.distMeasure);
        int label = 0;
        for (int i = 1; i < k; i++) {
            double t = Utils.dist(v, centroids[i], this.distMeasure);
            if (mindist > t) {
                mindist = t;
                label = i;
            }
        }
        return label;
    }

    /**
     * Remove empty centroids in clustering solution
     */
    private void getRidOfEmptyCentroids() {
        double[][] copy = Utils.deepCopy(this.centroids);
        Set<Integer> distLabels = Utils.distinctItems(this.labels);
        this.k = distLabels.size();
        int i = 0;
        this.centroids = new double[distLabels.size()][];
        HashMap<Integer, Integer> oldToNewIndex = new HashMap<>();
        for (int distLabel: distLabels) {
            this.centroids[i] = copy[distLabel];
            oldToNewIndex.put(distLabel, i);
            ++i;
        }
        for (i = 0; i < labels.length; ++i) {
            labels[i] = oldToNewIndex.get(labels[i]);
        }
    }

    private void initialize(double[][] data) {
        int N = data.length;
        int D = data[0].length;
        labels = new int[N];
        if (this.k > Math.sqrt(N)) {
            System.out.println("too many clusters");
        }
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
                    double d = Utils.dist(data[firstIdx], data[i], distMeasure);
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
        } else if (this.initialization == Initialization.HILL_CLIMBER) {
            assert (supplied);
            this.centroids = this.initialStartPoint;
            this.k = this.initialStartPoint.length;
        }

        // assign data points to clusterInstance centroids
        for (int i = 0; i < N; i++) {
            labels[i] = clusterInstance(data[i]);
        }

        Utils.checkClusterLabels(getLabels(), k);
    }

    private double[][] updateCentroids(double[][] data) {
        int N = data.length;
        int D = data[0].length;
        // initialize centroids and set to 0
        double[][] newc = new double [k][]; //new centroids
        int[] counts = new int[k]; // sizes of the clusters

        // intialize
        for (int i=0; i<k; i++) {
            counts[i] =0;
            newc[i] = new double[D];
            for (int j=0; j<D; j++) {
                newc[i][j] = 0;
            }
        }

        for (int i=0; i<N; i++){
            for (int j=0; j<D; j++){
                newc[labels[i]][j] += data[i][j]; // update that centroid by adding the member data record
            }
            counts[labels[i]]++;
        }

        // finally get the average
        for (int i=0; i< k; i++){
            for (int j=0; j<D; j++){
                newc[i][j] /= counts[i];
            }
        }

        return newc;
    }

    public void setDistMeasure(double distMeasure) {
        this.distMeasure = distMeasure;
    }

    public void setInitializationMethod(Initialization aInitialization) {
        this.initialization = aInitialization;
    }

    public void setMaxIterations(int aIter) {
        this.numIters = aIter;
    }

    public void setInitial(double[][] initial) {
        this.supplied = true;
        this.k = initial.length;
        this.initialStartPoint = Utils.deepCopy(initial);
    }

    public int numberOfClusters() {
        return centroids.length;
    }

    public int[] getLabels() {
        return labels;
    }

    public double[][] getCentroids() {
        return Utils.deepCopy(centroids);
    }

    public String toString() {
        return "number of clusters: " + this.k;
    }

    public void setSeed(int aSeed) {
        rnd = new Random(aSeed);
        this.seed = aSeed;
    }

    /**
     * check convergence condition
     * max{dist(c1[i], c2[i]), i=1..numClusters < threshold
     * */
    /**
     * check convergence condition
     * max{dist(c1[i], c2[i]), i=1..numClusters < threshold
     * */
    /*private boolean converge(double [][] c1, double [][] c2, double threshold){
        // c1 and c2 are two sets of centroids
        double maxv = 0;
        for (int i = 0; i < k; i++){
            double d = Utils.dist(c1[i], c2[i], this.distMeasure);
            if (maxv < d)
                maxv = d;
        }

        if (maxv < threshold)
            return true;
        else
            return false;
    }*/

}
