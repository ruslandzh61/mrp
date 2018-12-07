package PSO;

import GA.MyGenClustPlusPlus;
import utils.NCConstruct;
import utils.Utils;
import java.io.IOException;
import java.util.*;
import smile.validation.AdjustedRandIndex;
import weka.clusterers.ClusterEvaluation;
import weka.clusterers.SimpleKMeans;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.Remove;

/**
 * Created by rusland on 27.10.18.
 * PSODriver is the main class for running PSO-based clustering algorithm (Armani,2016):
 *      'Multiobjective clustering analysis using particle swarm optimization'
 *
 * NOT COMPLETE: Pareto-optimal front is retrieved using simple algorithm,
 *      which should be substituted later with MaxiMinPSO algorithm to to determine Pareto-domination (Li, 2007):
 *          'Better Spread and Convergence: Particle Swarm Multiobjective Optimization Using the Maximin Fitness Function'
 *
 * PROBLEM: Solutions don't improve after initialization step in the next iterations of PSO
 */
public class PSODriver {

    /** simple test using artificial two-dimensional data sets
     *  similar to the one used in (Inkaya, 2014) depicted on Fig. 3:
     *      'Adaptive neighbourhood construction algorithm based on density and connectivity'
     * */
    /*public void runDummy() {
        double[][] data;
        int[] labels;
        labels = new int[]{0,0,0,0,0,0,1,1,1,2};
        data = new double[][]{
                {2,2}, {3,3}, {3,1}, {4,2}, {1.6,-0.5},
                {3.01, -1.5}, {-4, 2}, {-2, 2}, {-3, 3},{7,7}
        };
        NCConstruct ncConstruct = new NCConstruct(data);
        int maxK = 6;

        Evaluator.Evaluation[] evaluation = {Evaluator.Evaluation.CONNECTIVITY, Evaluator.Evaluation.COHESION};
        Evaluator evaluator = new Evaluator();
        Problem problem = new Problem(data, evaluator);

        PSOConfiguration configuration = new PSOConfiguration();
        configuration.maxK = maxK;
        PSO pso = new PSO(problem, ncConstruct, evaluation, configuration);
        int[] labelsPred = pso.execute();

        System.out.println(new AdjustedRandIndex().measure(labels, labelsPred));
    }*/

    /**
     * main method to run PSO-based clustering
     * */

    public static void runMyKmeans(int runs, String path, char sep, boolean removeFirst, boolean normalize) throws Exception {
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
        List<String[]> dataStr = Utils.readDataFromCustomSeperator(path, sep);
        assert (dataStr.size()>0);
        assert (dataStr.get(0).length>0);

        // extract labels
        labelsTrue = Utils.extractLabels(dataStr,dataStr.get(0).length-1);
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
                KMeans kMeans = new KMeans(data, data.length, data[0].length, k, rnd.nextInt());
                kMeans.clustering(500);
                labelsPred = kMeans.getLabels();
                HashMap<Integer, double[]> centroids = Utils.centroids(data, labelsPred);
                double tmpDB = Utils.dbIndexScore(centroids, kMeans.getLabels(), data);
                double tmpARI = adjustedRandIndex.measure(labelsTrue, kMeans.getLabels());
                if (tmpARI > bestARI) {
                    bestARI = tmpARI;
                    bestDB = tmpDB;
                    bestK = k;
                }
            }
            meanARI += bestARI;
            meanDB += bestDB;
            meanNumClusters += bestK;

            sdofARI[run-1] = bestARI;
            sdofDB[run-1] = bestDB;
            sdOfNumClusters[run-1] = bestK;

            System.out.println("DB score of kMeans:      " + bestDB);
            System.out.println("ARI score of kMeans:     " + bestARI);
            System.out.println("# of clusters of kMeans: " + bestK);
        }
        System.out.println("mean and std dev of ARI score:          " + meanARI/runs +
                " +- " + Utils.standardDeviation(sdofARI));
        System.out.println("mean and std dev of DB Index score:     " + meanDB/runs +
                " +- " + Utils.standardDeviation(sdofDB));
        System.out.println("mean and std dev of number of clusters: " + meanNumClusters/runs +
                " +- " + Utils.standardDeviation(sdOfNumClusters));
    }
    public static void runKmeans(int KMEANS, int runs, String path, String filePathForWeka,
                     char sep, boolean removeFirst, boolean normalize) throws Exception {
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
        List<String[]> dataStr = Utils.readDataFromCustomSeperator(path, sep);
        assert (dataStr.size()>0);
        assert (dataStr.get(0).length>0);

        // extract labels
        labelsTrue = Utils.extractLabels(dataStr,dataStr.get(0).length-1);
        //System.out.println(Arrays.toString(labelsTrue));

        // exclude columns from csv file
        int[] excludedColumns;
        if (removeFirst) {
            excludedColumns = new int[]{0, dataStr.get(0).length - 1};
        } else {
            excludedColumns = new int[]{dataStr.get(0).length - 1};
        }
        data = Utils.extractAttributes(dataStr, excludedColumns);

        Instances instances = Utils.getData(filePathForWeka, removeFirst, normalize);

        for (int run = 1; run <= runs; ++run) {
            int bestK = -1;
            double bestARI = -1;
            double bestDB = -1;
            int minK = (int)(0.02 * data.length);
            int maxK = (int)(0.1 * data.length);
            for (int k = minK; k <= maxK; ++k) {
                SimpleKMeans kMeans = new SimpleKMeans();
                kMeans.setPreserveInstancesOrder(true);
                SelectedTag selectedTag = new SelectedTag(KMEANS, SimpleKMeans.TAGS_SELECTION);
                kMeans.setInitializationMethod(selectedTag);
                kMeans.setSeed(rnd.nextInt());
                kMeans.setNumClusters(k);
                kMeans.setMaxIterations(50);
                kMeans.buildClusterer(instances);
                labelsPred = kMeans.getAssignments();
                HashMap<Integer, double[]> centroids = Utils.centroids(data, labelsPred);
                double tmpDB = Utils.dbIndexScore(centroids, kMeans.getAssignments(), data);
                double tmpARI = adjustedRandIndex.measure(labelsTrue, kMeans.getAssignments());
                if (tmpARI > bestARI) {
                    bestARI = tmpARI;
                    bestDB = tmpDB;
                    bestK = k;
                }
            }
            meanARI += bestARI;
            meanDB += bestDB;
            meanNumClusters += bestK;


            sdofARI[run-1] = bestARI;
            sdofDB[run-1] = bestDB;
            sdOfNumClusters[run-1] = bestK;

            System.out.println("DB score of kMeans:      " + bestDB);
            System.out.println("ARI score of kMeans:     " + bestARI);
            System.out.println("# of clusters of kMeans: " + bestK);
        }
        System.out.println("mean and std dev of ARI score:          " + meanARI/runs +
                " +- " + Utils.standardDeviation(sdofARI));
        System.out.println("mean and std dev of DB Index score:     " + meanDB/runs +
                " +- " + Utils.standardDeviation(sdofDB));
        System.out.println("mean and std dev of number of clusters: " + meanNumClusters/runs +
                " +- " + Utils.standardDeviation(sdOfNumClusters));
    }
    public void run(int runs, String path, String filePathForWeka, PSOConfiguration configuration,
                    char sep, boolean removeFirst, boolean normalize) throws Exception {
        double[][] data;
        int[] labelsTrue, labelsPred;
        double meanARI = 0;
        double meanDB = 0;
        double meanNumClusters = 0;
        AdjustedRandIndex adjustedRandIndex = new AdjustedRandIndex();

        /* process data */
        // step 1 - read data from file
        List<String[]> dataStr = Utils.readDataFromCustomSeperator(path, sep);
        assert (dataStr.size()>0);
        assert (dataStr.get(0).length>0);

        // extract labels
        labelsTrue = Utils.extractLabels(dataStr,dataStr.get(0).length-1);
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
        Instances instances = Utils.getData(filePathForWeka, removeFirst, normalize);

        // step 2 - pick objectives
        NCConstruct ncConstruct = new NCConstruct(data);
        Evaluator.Evaluation[] evaluation = {Evaluator.Evaluation.CONNECTIVITY, Evaluator.Evaluation.COHESION};
        Evaluator evaluator = new Evaluator();
        Problem problem = new Problem(data, evaluator);
        configuration.maxK = (int)(Math.sqrt(problem.getData().length));

        Random rnd = new Random(1);
        double[] sdofARI = new double[runs];
        double[] sdofDB = new double[runs];
        double[] sdOfNumClusters = new double[runs];

        for (int run = 1; run <= runs; ++run) {
            System.out.println("run: " + run);
            // step 3 - run PSO algorithm
            //maxK = (int)Math.sqrt(data.length);
            //configuration.maxK = maxK;
            PSO pso = new PSO(problem, ncConstruct, evaluation, configuration, instances, labelsTrue);
            pso.setSeed(rnd.nextInt());
            // constructed clusters
            labelsPred = Utils.adjustLabels(pso.execute());
            Utils.adjustLabels(labelsPred);


            // step 4 - measure comparing to true labels
            HashMap<Integer, double[]> centroids = Utils.centroids(data, labelsPred);
            double aRIScore = adjustedRandIndex.measure(labelsTrue, labelsPred);
            double dbScore = Utils.dbIndexScore(centroids, labelsPred, data);
            int numClusters = Utils.distinctNumberOfItems(labelsPred);

            meanARI += aRIScore;
            meanDB += dbScore;
            meanNumClusters += numClusters;

            sdofARI[run-1] = aRIScore;
            sdofDB[run-1] = dbScore;
            sdOfNumClusters[run-1] = numClusters;

            System.out.println("ARI score of PSO:   " + aRIScore);
            System.out.println("DB score of PSO:    " + dbScore);
            System.out.println("number of clusters: " + numClusters);

            // optional step - write true and constructed labels into a file
        /*Utils.whenWriteStringUsingBufferedWritter_thenCorrect(Arrays.toString(labelsTrue) +
                System.getProperty("line.separator") + "," + Arrays.toString(labelsPred), "data/output.txt");*/

            // optional step - objectives of true clusters
        /*System.out.println("objectives of true clusters: " + Arrays.toString(problem.evaluate(
                new Solution(labelsTrue, Utils.distinctNumberOfItems(labelsTrue)), evaluation, new NCConstruct(data))));*/
        }

        System.out.println("mean and std dev of ARI score:          " + meanARI/runs +
                " +- " + Utils.standardDeviation(sdofARI));
        System.out.println("mean and std dev of DB Index score:     " + meanDB/runs +
                " +- " + Utils.standardDeviation(sdofDB));
        System.out.println("mean and std dev of number of clusters: " + meanNumClusters/runs +
                " +- " + Utils.standardDeviation(sdOfNumClusters));
        System.out.println("--------------------------");
    }

    /*private static void removeNoise(int[] labels) {
        if ()
    }*/

    public static void main(String[] args) throws Exception {
        //new PSODriver().runDummy();
        try {
            // test using UCI 'glass' public data set - https://archive.ics.uci.edu/ml/datasets/glass+identification
            // pick file manually or pass a path string
            boolean pickManually = false;
            String filePath, filePathForWeka;
            filePath = pickManually ? Utils.pickAFile(): "data/flame.csv";
            filePathForWeka = pickManually ? Utils.pickAFile(): "data/p-flame.csv";
            PSOConfiguration configuration = new PSOConfiguration();
            // default configuration
            /*configuration.c1 = 1.42;
            configuration.c2 = 1.63;
            configuration.maxW = 0.9;
            configuration.minW = 0.4;
            configuration.maxIteration = 1000;
            configuration.maxIterWithoutImprovement = 50;
            configuration.pMax = 150;
            configuration.pickLeaderRandomly = false;*/
            new PSODriver().run(10, filePath, filePathForWeka, configuration, ',', false, true);
            //Utils.nominalForm("data/glass.csv");
            /*System.out.println("my k-means: ");
            PSODriver.runMyKmeans(40, filePath, ',', false, false);
            System.out.println("-------");
            System.out.println("WEKA random k-means");
            PSODriver.runKmeans(SimpleKMeans.RANDOM, 40, filePath, filePathForWeka, ',', false, true);
            System.out.println("-------");
            System.out.println("WEKA canopy k-means");
            PSODriver.runKmeans(SimpleKMeans.CANOPY, 40, filePath, filePathForWeka, ',', false, true);
            System.out.println("-------");
            System.out.println("WEKA kmeans++ k-means");
            PSODriver.runKmeans(SimpleKMeans.KMEANS_PLUS_PLUS, 40, filePath, filePathForWeka, ',', false, true);
            System.out.println("-------");
            System.out.println("WEKA farthest-first k-means");
            PSODriver.runKmeans(SimpleKMeans.FARTHEST_FIRST, 40, filePath, filePathForWeka, ',', false, true);*/
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
