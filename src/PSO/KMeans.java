package PSO;

import smile.validation.AdjustedRandIndex;
import utils.Utils;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Created by rusland on 10.11.18.
 * code is taken from http://cecs.wright.edu/~keke.chen/cloud/labs/mapreduce/KMeans.java
 * https://github.com/JasonAltschuler/KMeansPlusPlus/blob/master/src/KMeans.java
 */
public class KMeans {
    private int[] label;
    private double[][] data;
    private int N, D;
    private double[][] centroids;
    private int k;
    private int seed;
    private static int default_seed = 10;
    Random rnd;

    KMeans(double[][] aData, int aN, int aD, int aK, int aSeed) {
        data = aData;
        N = aN;
        D = aD;
        label = new int[N];
        if (aK > N) {
            this.k = N;
        } else {
            this.k = aK;
        }

        setSeed(aSeed);

        // choose existing data points as initial data points
        centroids = new double[k][D];
        double[][] copy = Utils.deepCopy(data);
        for (int i = 0; i < k; i++) {
            int rand = rnd.nextInt(N - i);
            for (int j = 0; j < D; j++) {
                centroids[i][j] = copy[rand][j];       // store chosen centroid
                copy[rand][j] = copy[N - 1 - i][j];    // ensure sampling without replacement
            }
        }
        // assign data points to closest centroids
        for (int i=0; i < N; i++){
            label[i] = closest(data[i]);
        }
        Utils.checkClusterLabels(getLabels(), k);
    }

    /**
     * performs complete clustering
     *  */
    public void clustering(int niter) {
        if (niter <= 0 || niter > 1000)
            niter = 100;
        double [][] c1 = centroids;
        double threshold = 0.00001;
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
            if ((niter > 0 && round >= niter))
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

    /**
     * performs one iteration of k-means clustering
     * */
    void oneIter() {
        centroids = updateCentroids();
        //assign record to the closest centroid
        for (int i=0; i < N; i++){
            label[i] = closest(data[i]);
        }
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
        double mindist = dist(v, centroids[0]);
        int label = 0;
        for (int i = 1; i < k; i++) {
            double t = dist(v, centroids[i]);
            if (mindist > t) {
                mindist = t;
                label = i;
            }
        }
        return label;
    }

    /**
     * compute Euclidean distance between two vectors v1 and v2
     * */
    private double dist(double [] v1, double [] v2){
        double sum=0;
        for (int i=0; i < D; i++){
            double d = v1[i]-v2[i];
            sum += d*d;
        }
        return Math.sqrt(sum);
    }

    /**
     * check convergence condition
     * max{dist(c1[i], c2[i]), i=1..numClusters < threshold
     * */
    private boolean converge(double [][] c1, double [][] c2, double threshold){
        // c1 and c2 are two sets of centroids
        double maxv = 0;
        for (int i=0; i< k; i++){
            double d= dist(c1[i], c2[i]);
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
        //kMeans.clustering(100);
        int[] labelsPred = kMeans.getLabels();

        //smile.clustering.KMeans kMeans = new smile.clustering.KMeans(data,k,100);
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
