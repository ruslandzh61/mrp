package PSO;

import clustering.Evaluator;
import clustering.KMeans;
import utils.NCConstruct;
import utils.Utils;
import java.io.IOException;
import java.util.*;
import smile.validation.AdjustedRandIndex;
import weka.clusterers.SimpleKMeans;
import weka.core.Instances;
import weka.core.SelectedTag;

/**
 * Created by rusland on 27.10.18.
 * PSODriver is the main class for running PSO-based buildClusterer algorithm (Armani,2016):
 *      'Multiobjective buildClusterer analysis using particle swarm optimization'
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
     * main method to run PSO-based buildClusterer
     * */

    public static void runMyKmeans(boolean useSmile, KMeans.Initialization init, int runs, String path, char sep,
                                   boolean removeFirst, boolean normalize) throws Exception {
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
                if (useSmile) {
                    labelsPred = new int[data.length];
                    smile.clustering.KMeans kMeans = new smile.clustering.KMeans(data, k, 500);
                    for (int i = 0; i < data.length; ++i) {
                        labelsPred[i] = kMeans.predict(data[i]);
                    }
                } else {
                    KMeans kMeans = new KMeans(k, 2.0);
                    kMeans.setSeed(rnd.nextInt());
                    kMeans.setInitializationMethod(init);
                    kMeans.buildClusterer(data);
                    labelsPred = kMeans.getLabels();
                }
                Utils.removeNoise(labelsPred, data, 2, 2.0);
                Utils.adjustAssignments(labelsPred);
                HashMap<Integer, double[]> centroids = Utils.centroids(data, labelsPred);
                double tmpDB = Utils.dbIndexScore(centroids, labelsPred, data);
                double tmpARI = adjustedRandIndex.measure(labelsTrue, labelsPred);
                if (tmpARI > bestARI) {
                    bestARI = tmpARI;
                    bestDB = tmpDB;
                    bestK = Utils.distinctNumberOfItems(labelsPred); //k;
                }

                /*System.out.println("temp ARI score of kMeans:     " + Utils.doublePrecision(tmpARI, 4));
                System.out.println("temp DB score of kMeans:      " + Utils.doublePrecision(tmpDB, 4));
                System.out.println("temp # of clusters of kMeans: " + Utils.distinctNumberOfItems(labelsPred));*/
            }
            meanARI += bestARI;
            meanDB += bestDB;
            meanNumClusters += bestK;

            sdofARI[run-1] = bestARI;
            sdofDB[run-1] = bestDB;
            sdOfNumClusters[run-1] = bestK;

            /*System.out.println("final ARI score of kMeans:     " + Utils.doublePrecision(bestARI, 4));
            System.out.println("final DB score of kMeans:      " + Utils.doublePrecision(bestDB, 4));
            System.out.println("final # of clusters of kMeans: " + bestK);*/
        }
        System.out.println("mean and std dev of ARI score:          " + Utils.doublePrecision(meanARI/runs, 4) +
                " +- " + Utils.doublePrecision(Utils.standardDeviation(sdofARI), 4));
        System.out.println("mean and std dev of DB Index score:     " + Utils.doublePrecision(meanDB/runs, 4) +
                " +- " + Utils.doublePrecision(Utils.standardDeviation(sdofDB), 4));
        System.out.println("mean and std dev of number of clusters: " + Utils.doublePrecision(meanNumClusters/runs, 4) +
                " +- " + Utils.doublePrecision(Utils.standardDeviation(sdOfNumClusters), 4));
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
    public void run(int runs, String path, String filePathForWeka, PSOConfiguration configuration,
                    boolean removeFirst, boolean normalize) throws Exception {
        char sep = ',';
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
        configuration.maxK = (int) (Math.sqrt(problem.getData().length));
        boolean normObjectives = true;

        Random rnd = new Random(1);
        double[] sdofARI = new double[runs];
        double[] sdofDB = new double[runs];
        double[] sdOfNumClusters = new double[runs];

        for (int run = 1; run <= runs; ++run) {
            System.out.println("run: " + run);
            // step 3 - run PSO algorithm
            //maxK = (int)Math.sqrt(data.length);
            //configuration.maxK = maxK;
            PSO pso = new PSO(problem, ncConstruct, evaluation, configuration, instances, labelsTrue, normObjectives);
            pso.setSeed(rnd.nextInt());
            // constructed clusters
            labelsPred = Utils.adjustLabels(pso.execute());
            //int[] labelsPredCloned = labelsPred.clone();
            Utils.removeNoise(labelsPred, data, 2, 2.0);
            Utils.adjustAssignments(labelsPred);

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

            System.out.println("ARI score of PSO for run:   " + Utils.doublePrecision(aRIScore, 4));
            System.out.println("DB score of PSO for run:    " + Utils.doublePrecision(dbScore, 4));
            System.out.println("number of clusters for run: " + numClusters);

            /* test hill-climber */
            /*centroids = Utils.centroids(data, labelsPredCloned);
            double[][] initialCentroids = new double[centroids.size()][];
            initialCentroids = centroids.values().toArray(initialCentroids);
            clustering.KMeans kMeans = new clustering.KMeans(data, initialCentroids.length, 2.0);
            kMeans.setInitializationMethod(clustering.KMeans.Initialization.HILL_CLIMBER);
            kMeans.setInitial(initialCentroids);
            kMeans.buildClusterer(500);
            labelsPred = Utils.adjustLabels(kMeans.getLabels());
            Utils.removeNoise(labelsPred, data, 2, 2.0);
            Utils.adjustAssignments(labelsPred);

            centroids = Utils.centroids(data, labelsPred);
            aRIScore = adjustedRandIndex.measure(labelsTrue, labelsPred);
            dbScore = Utils.dbIndexScore(centroids, labelsPred, data);
            numClusters = Utils.distinctNumberOfItems(labelsPred);

            System.out.println("ARI score of PSO with hill-climber for run:   " + Utils.doublePrecision(aRIScore, 4));
            System.out.println("DB score of PSO with hill-climber for run:    " + Utils.doublePrecision(dbScore, 4));
            System.out.println("number of clusters with hill-climber for run: " + numClusters);*/

            // optional step - write true and constructed labels into a file
        /*Utils.whenWriteStringUsingBufferedWritter_thenCorrect(Arrays.toString(labelsTrue) +
                System.getProperty("line.separator") + "," + Arrays.toString(labelsPred), "data/output.txt");*/

            // optional step - objectives of true clusters
        /*System.out.println("objectives of true clusters: " + Arrays.toString(problem.evaluate(
                new Solution(labelsTrue, Utils.distinctNumberOfItems(labelsTrue)), evaluation, new NCConstruct(data))));*/
        }

        System.out.println("mean and std dev of ARI score:          " + Utils.doublePrecision(meanARI/runs, 4) +
                " +- " + Utils.doublePrecision(Utils.standardDeviation(sdofARI), 4));
        System.out.println("mean and std dev of DB Index score:     " + Utils.doublePrecision(meanDB/runs, 4) +
                " +- " + Utils.doublePrecision(Utils.standardDeviation(sdofDB), 4));
        System.out.println("mean and std dev of number of clusters: " + Utils.doublePrecision(meanNumClusters/runs, 4) +
                " +- " + Utils.doublePrecision(Utils.standardDeviation(sdOfNumClusters), 4));
        System.out.println("--------------------------");
    }

    public static void main(String[] args) throws Exception {
        //new PSODriver().runDummy();
        try {
            // test using UCI 'glass' public data set - https://archive.ics.uci.edu/ml/datasets/glass+identification
            // pick file manually or pass a path string
            boolean pickManually = false;
            String filePath, filePathForWeka;
            filePath = pickManually ? Utils.pickAFile(): "data/glass.csv";
            filePathForWeka = pickManually ? Utils.pickAFile(): "data/p-glass.csv";
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
            new PSODriver().run(10, filePath, filePathForWeka, configuration, false, false);
            //Utils.nominalForm("data/glass.csv");
            /*boolean removeFirst = false;
            boolean normalize = false;
            System.out.println("remove id: " + removeFirst);
            System.out.println("normalize: " + normalize);
            System.out.println("my k-means: ");
            PSODriver.runMyKmeans(false, clustering.KMeans.Initialization.RANDOM, 10, filePath, ',', removeFirst, normalize);
            System.out.println("-------");
            System.out.println("my k-means++: ");
            PSODriver.runMyKmeans(false, clustering.KMeans.Initialization.KMEANS_PLUS_PLUS, 10, filePath, ',', removeFirst, normalize);
            System.out.println("-------");
            System.out.println("WEKA random k-means");
            PSODriver.runKmeans(SimpleKMeans.RANDOM, 10, filePath, filePathForWeka, ',', removeFirst, normalize);
            System.out.println("-------");
            System.out.println("WEKA kmeans++ k-means");
            PSODriver.runKmeans(SimpleKMeans.KMEANS_PLUS_PLUS, 10, filePath, filePathForWeka, ',', removeFirst, normalize);*/
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
