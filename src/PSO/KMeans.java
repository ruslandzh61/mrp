package PSO;

import smile.validation.AdjustedRandIndex;
import utils.Utils;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Created by rusland on 10.11.18.
 * code is taken from http://cecs.wright.edu/~keke.chen/cloud/labs/mapreduce/KMeans.java
 * https://github.com/JasonAltschuler/KMeansPlusPlus/blob/master/src/KMeans.java
 * http://commons.apache.org/proper/commons-math/apidocs/org/apache/commons/math4/ml/clustering/KMeansPlusPlusClusterer.html#KMeansPlusPlusClusterer-int-
 */
public class KMeans {
    private int[] label;
    private double[][] data;
    private int N, D;
    private double[][] centroids;
    private int k;
    private int seed;
    private static int default_seed = 10;
    private double pow = 2;
    Random rnd;
    private boolean plus = true;

    KMeans(double[][] aData, int aN, int aD, int aK, double aPow) {
        data = aData;
        N = aN;
        D = aD;
        label = new int[N];
        this.pow = aPow;
        if (aK > N) {
            this.k = N;
        } else {
            this.k = aK;
        }

        this.seed = default_seed;
    }

    /**
     * performs complete buildClusterer
     *  */
    public void buildClusterer(int niter) {
        initialize();

        if (niter <= 0 || niter > 500)
            niter = 500;
        double [][] c1 = centroids;
        double threshold = 0.000000001;
        int round=0;

        while (true) {
            // update _centroids with the last round results
            centroids = c1;

            //assign record to the closest centroid
            label = new int[N];
            for (int i = 0; i < N; i++) {
                label[i] = closest(data[i]);
            }

            // recompute centroids based on the assignments
            c1 = updateCentroids();
            round++;
            if (niter > 0 && round >= niter)
                break;
            if (converge(centroids, c1, threshold)) {
                //System.out.println("converged at: " + round);
                break;
            }
        }
    }

    public int[] getLabels() {
        return label;
    }

    public double[][] getCentroids() {
        return Utils.deepCopy(centroids);
    }

    public void setPlus(boolean plus) {
        this.plus = plus;
    }

    /**
     * performs one iteration of k-means buildClusterer
     * */
    void oneIter() {
        centroids = updateCentroids();
        //assign record to the closest centroid
        for (int i=0; i < N; i++){
            label[i] = closest(data[i]);
        }
    }

    private void initialize() {
        // choose existing data points as initial data points
        centroids = new double[k][D];

        if (plus) {
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
        } else {
            double[][] copy = Utils.deepCopy(data);
            for (int i = 0; i < k; i++) {
                int rand = rnd.nextInt(N - i);
                for (int j = 0; j < D; j++) {
                    centroids[i][j] = copy[rand][j];       // store chosen centroid
                    copy[rand][j] = copy[N - 1 - i][j];    // ensure sampling without replacement
                }
            }
            // assign data points to closest centroids
            for (int i = 0; i < N; i++) {
                label[i] = closest(data[i]);
            }
        }
        Utils.checkClusterLabels(getLabels(), k);
    }

    private double[][] updateCentroids() {
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
     *  find the closest centroid for the record v
     *  */
    private int closest(double[] v){
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
        for (int i=0; i< k; i++){
            double d= Utils.dist(c1[i], c2[i], this.pow);
            if (maxv<d)
                maxv = d;
        }

        if (maxv <threshold)
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


        List<String[]> dataStr = Utils.readDataFromCustomSeperator("data/glass.csv", ',');
        assert (dataStr.size()>0);

        labels = Utils.extractLabels(dataStr, dataStr.get(0).length-1);

        // exclude columns from csv file
        int[] excludedColumns = {0,dataStr.get(0).length-1};
        data = Utils.extractAttributes(dataStr, excludedColumns);
        k = data.length/20;
        KMeans kMeans;
        kMeans = new KMeans(data, data.length, data[0].length, k);
        kMeans.oneIter();
        //kMeans.buildClusterer(100);
        int[] labelsPred = kMeans.getLabels();

        //smile.buildClusterer.KMeans kMeans = new smile.buildClusterer.KMeans(data,k,100);
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
