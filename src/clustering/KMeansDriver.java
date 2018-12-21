package clustering;

import smile.validation.AdjustedRandIndex;
import utils.Utils;
import weka.clusterers.SimpleKMeans;
import weka.core.Instances;
import weka.core.SelectedTag;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

/**
 * Created by rusland on 20.12.18.
 */
public class KMeansDriver {
    private double[][] dataAttrs;
    private int[] labelsTrue;

    public void runMyKmeans(KMeans.Initialization init, int runs, String path,
                            boolean removeFirst, boolean normalize) throws Exception {
        char sep = ',';
        double[][] data;
        int[] labelsTrue, labelsPred;
        double meanARI = 0, meanDB = 0, meanNumClusters = 0;
        AdjustedRandIndex adjustedRandIndex = new AdjustedRandIndex();
        Random rnd = new Random(1);
        double[] sdofARI = new double[runs];
        double[] sdofDB = new double[runs];
        double[] sdOfNumClusters = new double[runs];

        /* process data */
        // step 1 - read data from file
        List<String[]> dataStr = Utils.readFile(path, sep);
        assert (dataStr.size()>0);
        assert (dataStr.get(0).length>0);

        // extract labels
        labelsTrue = Utils.extractLabels(dataStr,dataStr.get(0).length-1);
        System.out.println(Arrays.toString(labelsTrue));
        //System.out.println(Arrays.toString(labelsTrue));

        // exclude columns from csv file
        int[] excludedColumns;
        if (removeFirst) {
            excludedColumns = new int[]{0, dataStr.get(0).length - 1};
        } else {
            excludedColumns = new int[]{dataStr.get(0).length - 1};
        }
        data = Utils.extractAttributes(dataStr, excludedColumns);
        if (normalize) {
            Utils.normalize(data);
        }

        for (int run = 1; run <= runs; ++run) {
            int bestK = -1;
            double bestARI = -1;
            double bestDB = -1;
            int minK = 2;//(int)(0.02 * data.length);
            int maxK = (int) Math.sqrt(data.length); //(int)(0.1 * data.length);
            for (int k = minK; k <= maxK; ++k) {
                /*if (useSmile) {
                    labelsPred = new int[data.length];
                    smile.clustering.KMeans kMeans = new smile.clustering.KMeans(data, k, 500);
                    for (int i = 0; i < data.length; ++i) {
                        labelsPred[i] = kMeans.predict(data[i]);
                    }
                } else {*/
                KMeans kMeans = new KMeans(k, 2.0);
                kMeans.setSeed(rnd.nextInt());
                kMeans.setInitializationMethod(init);
                kMeans.buildClusterer(data);
                labelsPred = kMeans.getLabels();

                //Utils.removeNoise(labelsPred, data, 2, 2.0);
                //Utils.adjustAssignments(labelsPred);
                HashMap<Integer, double[]> centroids = Utils.centroids(data, labelsPred);
                double tmpARI = adjustedRandIndex.measure(labelsTrue, labelsPred);
                double tmpDB = Utils.dbIndexScore(centroids, labelsPred, data);
                double silhScore = Utils.silhoutte(centroids, labelsPred, data);

                if (tmpARI > bestARI) {
                    bestARI = tmpARI;
                    bestDB = tmpDB;
                    bestK = Utils.distinctNumberOfItems(labelsPred); //k;
                }

                /*System.out.println("solution: " + Arrays.toString(labelsPred));
                System.out.println("temp ARI score of kMeans:     " + Utils.doublePrecision(tmpARI, 4));
                System.out.println("temp DB score of kMeans:      " + Utils.doublePrecision(tmpDB, 4));
                System.out.println("temp Silh score of kMeans:      " + Utils.doublePrecision(silhScore, 4));
                System.out.println("temp # of clusters of kMeans: " + k + " : " + Utils.distinctNumberOfItems(labelsPred) + " : " + kMeans.numberOfClusters());
                System.out.println(Utils.distinctNumberOfItems(labelsPred) == kMeans.numberOfClusters());*/
            }
            meanARI += bestARI;
            meanDB += bestDB;
            meanNumClusters += bestK;

            sdofARI[run-1] = bestARI;
            sdofDB[run-1] = bestDB;
            sdOfNumClusters[run-1] = bestK;

            System.out.println("final ARI score of kMeans:     " + Utils.doublePrecision(bestARI, 4));
            System.out.println("final DB score of kMeans:      " + Utils.doublePrecision(bestDB, 4));
            System.out.println("final # of clusters of kMeans: " + bestK);
        }
        System.out.println("mean and std dev of ARI score:          " + Utils.doublePrecision(meanARI/runs, 4) +
                " +- " + Utils.doublePrecision(Utils.standardDeviation(sdofARI), 4));
        System.out.println("mean and std dev of DB Index score:     " + Utils.doublePrecision(meanDB/runs, 4) +
                " +- " + Utils.doublePrecision(Utils.standardDeviation(sdofDB), 4));
        System.out.println("mean and std dev of number of clusters: " + Utils.doublePrecision(meanNumClusters/runs, 4) +
                " +- " + Utils.doublePrecision(Utils.standardDeviation(sdOfNumClusters), 4));
    }

    public void runKmeans(int KMEANS, int runs, Dataset dataset) throws Exception {
        char sep = ',';
        double[][] data;
        int[] labelsTrue, labelsPred;
        double meanARI = 0, meanDB = 0, meanNumClusters = 0;
        double[] sdofARI = new double[runs];
        double[] sdofDB = new double[runs];
        double[] sdOfNumClusters = new double[runs];
        Random rnd = new Random(1);
        AdjustedRandIndex adjustedRandIndex = new AdjustedRandIndex();

        /* process data */
        // step 1 - read data from file
        List<String[]> dataStr = Utils.readFile(dataset.getPath(), sep);
        assert (dataStr.size()>0);
        assert (dataStr.get(0).length>0);

        // extract labels
        labelsTrue = Utils.extractLabels(dataStr,dataStr.get(0).length-1);
        //System.out.println(Arrays.toString(labelsTrue));

        // exclude columns from csv file
        int[] excludedColumns;
        if (dataset.isRemoveFirst()) {
            excludedColumns = new int[]{0, dataStr.get(0).length - 1};
        } else {
            excludedColumns = new int[]{dataStr.get(0).length - 1};
        }
        data = Utils.extractAttributes(dataStr, excludedColumns);
        if (dataset.isNormalize()) {
            Utils.normalize(data);
        }

        Instances instances = Utils.getData(dataset);

        for (int run = 1; run <= runs; ++run) {
            int bestK = -1;
            double bestARI = -1;
            double bestDB = -1;
            int minK = 2;//(int)(0.02 * data.length);
            int maxK = (int) Math.sqrt(data.length); //(int)(0.1 * data.length);
            for (int k = minK; k <= maxK; ++k) {
                SimpleKMeans kMeans = new SimpleKMeans();
                kMeans.setPreserveInstancesOrder(true);
                SelectedTag selectedTag = new SelectedTag(KMEANS, SimpleKMeans.TAGS_SELECTION);
                kMeans.setInitializationMethod(selectedTag);
                kMeans.setSeed(rnd.nextInt());
                kMeans.setNumClusters(k);
                kMeans.buildClusterer(instances);
                labelsPred = kMeans.getAssignments();
                Utils.removeNoise(labelsPred, data, 2, 2.0);
                Utils.adjustAssignments(labelsPred);
                HashMap<Integer, double[]> centroids = Utils.centroids(data, labelsPred);
                double tmpDB = Utils.dbIndexScore(centroids, kMeans.getAssignments(), data);
                double tmpARI = adjustedRandIndex.measure(labelsTrue, kMeans.getAssignments());
                if (tmpARI > bestARI) {
                    bestARI = tmpARI;
                    bestDB = tmpDB;
                    bestK = Utils.distinctNumberOfItems(labelsPred);
                }
            }
            meanARI += bestARI;
            meanDB += bestDB;
            meanNumClusters += bestK;


            sdofARI[run-1] = bestARI;
            sdofDB[run-1] = bestDB;
            sdOfNumClusters[run-1] = bestK;

            /*System.out.println("DB score of kMeans:      " + Utils.doublePrecision(bestDB, 4));
            System.out.println("ARI score of kMeans:     " + Utils.doublePrecision(bestARI, 4));
            System.out.println("# of clusters of kMeans: " + bestK);*/
        }
        System.out.println("mean and std dev of ARI score:          " + Utils.doublePrecision(meanARI/runs, 4) +
                " +- " + Utils.doublePrecision(Utils.standardDeviation(sdofARI), 4));
        System.out.println("mean and std dev of DB Index score:     " + Utils.doublePrecision(meanDB/runs, 4) +
                " +- " + Utils.doublePrecision(Utils.standardDeviation(sdofDB), 4));
        System.out.println("mean and std dev of number of clusters: " + Utils.doublePrecision(meanNumClusters/runs, 4) +
                " +- " + Utils.doublePrecision(Utils.standardDeviation(sdOfNumClusters), 4));
    }

    public static void main(String[] args) throws Exception {
        Dataset dataset = Dataset.GLASS;
        boolean removeFirst = false;
        boolean normalize = true;
        System.out.println("remove id: " + removeFirst);
        System.out.println("normalize: " + normalize);
        System.out.println("my k-means++: ");
        new KMeansDriver().runMyKmeans(clustering.KMeans.Initialization.KMEANS_PLUS_PLUS,
                40, dataset.getPath(), removeFirst, normalize);
        System.out.println("-------");
        /*System.out.println("my k-means: ");
        PSODriver.runMyKmeans(clustering.KMeans.Initialization.RANDOM, 10, filePath, removeFirst, normalize);
        System.out.println("-------");
        System.out.println("WEKA random k-means");
        PSODriver.runKmeans(SimpleKMeans.RANDOM, 10, filePath, filePathForWeka, removeFirst, normalize);
        System.out.println("-------");
        System.out.println("WEKA kmeans++ k-means");
        PSODriver.runKmeans(SimpleKMeans.KMEANS_PLUS_PLUS, 10, filePath, filePathForWeka, removeFirst, normalize);*/
    }
}
