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
    public void run(int runs, String path, String filePathForWeka, PSOConfiguration configuration,
                    char sep, boolean removeFirst) throws Exception {
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

        Instances instances = getData(filePathForWeka, removeFirst, false);
        Random rnd = new Random(1);

        /*double meanDbKmeans = 0.0;
        double meanARIKmeans = 0.0;
        double meanK = 0.0;
        for (int seed = 1; seed <= runs; ++seed) {
            int bestK = -1;
            double bestARI = -1;
            double bestDB = -1;
            for (double i = 0.02; i <= 0.1; i += 0.01) {
                int k = (int)(i*data.length)+1;
                SimpleKMeans kMeans = new SimpleKMeans();
                kMeans.setPreserveInstancesOrder(true);
                SelectedTag selectedTag = new SelectedTag(SimpleKMeans.FARTHEST_FIRST, SimpleKMeans.TAGS_SELECTION);
                kMeans.setInitializationMethod(selectedTag);
                kMeans.setSeed(rnd.nextInt());
                kMeans.setNumClusters(k);
                kMeans.buildClusterer(instances);
                HashMap<Integer, double[]> centroids = Utils.centroids(data, kMeans.getAssignments());
                double tmpDB = Utils.dbIndexScore(centroids, kMeans.getAssignments(), data);
                double tmpARI = adjustedRandIndex.measure(labelsTrue, kMeans.getAssignments());
                if (tmpARI > bestARI) {
                    bestARI = tmpARI;
                    bestDB = tmpDB;
                    bestK = k;
                }
            }
            meanDbKmeans += bestDB;
            meanARIKmeans += bestARI;
            meanK += bestK;
            System.out.println("DB score of kMeans:      " + bestDB);
            System.out.println("ARI score of kMeans:     " + bestARI);
            System.out.println("# of clusters of kMeans: " + bestK);
        }
        System.out.println("mean DB score of kMeans:  " + meanDbKmeans/runs);
        System.out.println("mean ARI score of kMeans: " + meanARIKmeans/runs);
        System.out.println("# of clusters of kMeans: " + meanK/runs);*/

        // step 2 - pick objectives
        rnd = new Random(1);

        NCConstruct ncConstruct = new NCConstruct(data);
        Evaluator.Evaluation[] evaluation = {Evaluator.Evaluation.CONNECTIVITY, Evaluator.Evaluation.COHESION};
        Evaluator evaluator = new Evaluator();
        Problem problem = new Problem(data, evaluator);

        double meanDbKmeans = 0.0;
        double meanARIKmeans = 0.0;
        double meanK = 0.0;
        for (int seed = 1; seed <= runs; ++seed) {
            int bestK = -1;
            double bestARI = -1;
            double bestDB = -1;
            for (double i = 0.02; i <= 0.1; i += 0.01) {
                int k = (int)(i*data.length)+1;
                KMeans kMeans = new KMeans(problem.getData(), problem.getN(), problem.getD(), k, rnd.nextInt());
                kMeans.clustering(100);
                HashMap<Integer, double[]> centroids = Utils.centroids(data, kMeans.getLabels());
                double tmpDB = Utils.dbIndexScore(centroids, kMeans.getLabels(), data);
                double tmpARI = adjustedRandIndex.measure(labelsTrue, kMeans.getLabels());
                if (tmpARI > bestARI) {
                    bestARI = tmpARI;
                    bestDB = tmpDB;
                    bestK = k;
                }
            }
            meanDbKmeans += bestDB;
            meanARIKmeans += bestARI;
            meanK += bestK;
            System.out.println("DB score of kMeans:      " + bestDB);
            System.out.println("ARI score of kMeans:     " + bestARI);
            System.out.println("# of clusters of kMeans: " + bestK);
        }
        System.out.println("mean DB score of kMeans:  " + meanDbKmeans/runs);
        System.out.println("mean ARI score of kMeans: " + meanARIKmeans/runs);
        System.out.println("# of clusters of kMeans: " + meanK/runs);

        rnd = new Random(1);
        for (int seed = 1; seed <= runs; ++seed) {
            // step 3 - run PSO algorithm
            //maxK = (int)Math.sqrt(data.length);
            //configuration.maxK = maxK;
            PSO pso = new PSO(problem, ncConstruct, evaluation, configuration, instances, labelsTrue);
            pso.setSeed(rnd.nextInt());
            // constructed clusters
            labelsPred = Utils.adjustLabels(pso.execute());

            // step 4 - measure comparing to true labels
            HashMap<Integer, double[]> centroids = Utils.centroids(data, labelsPred);
            double ARIScore = adjustedRandIndex.measure(labelsTrue, labelsPred);
            double dbScore = Utils.dbIndexScore(centroids, labelsPred, data);
            int numClusters = Utils.distinctNumberOfItems(labelsPred);

            meanARI += ARIScore;
            meanDB += dbScore;
            meanNumClusters += numClusters;

            System.out.println("run: " + seed);
            System.out.println("ARI score of PSO:   " + ARIScore);
            System.out.println("DB score of PSO:    " + dbScore);
            System.out.println("number of clusters: " + numClusters);
            System.out.println(Arrays.toString(labelsPred));
            System.out.println(Arrays.toString(labelsTrue));

            // k-means baseline algorithm
            //KMeans kMeans = new KMeans(data,problem.getN(), problem.getD(), maxK);
            /*SimpleKMeans kMeans = new SimpleKMeans();
            kMeans.setSeed(seed);
            kMeans.buildClusterer(instances);
            kMeans.setNumClusters(Utils.distinctNumberOfItems(labelsTrue));
            centroids = Utils.centroids(data, kMeans.getAssignments());
            int i = 0;
            for (Instance instance: new Instances(instances)) {
                labelsPred[i++] = kMeans.clusterInstance(instance);
            }
            System.out.println("DB score of kMeans:  " + Utils.dbIndexScore(centroids, kMeans.getAssignments(), data));
            System.out.println("ARI score of kMeans: " + adjustedRandIndex.measure(labelsTrue, kMeans.getAssignments()));
            System.out.println("----------------------");*/

            // optional step - write true and constructed labels into a file
        /*Utils.whenWriteStringUsingBufferedWritter_thenCorrect(Arrays.toString(labelsTrue) +
                System.getProperty("line.separator") + "," + Arrays.toString(labelsPred), "data/output.txt");*/

            // optional step - objectives of true clusters
        /*System.out.println("objectives of true clusters: " + Arrays.toString(problem.evaluate(
                new Solution(labelsTrue, Utils.distinctNumberOfItems(labelsTrue)), evaluation, new NCConstruct(data))));*/
        }

        System.out.println("mean ARI score:          " + meanARI/runs);
        System.out.println("mean DB Index score:     " + meanDB/runs);
        System.out.println("mean number of clusters: " + meanNumClusters/runs);
        System.out.println("--------------------------");
    }

    private Instances getData(String filePath, boolean removeFirst, boolean normalize) throws Exception {
        Remove filter;
        Instances data = ConverterUtils.DataSource.read(filePath);
        data.setClassIndex(data.numAttributes() - 1);
        if (removeFirst) {
            filter = new Remove();
            filter.setAttributeIndices("1");
            filter.setInputFormat(data);
            data = Filter.useFilter(data, filter);
            data.setClassIndex(data.numAttributes() - 1);
        }

        if (normalize) {
            Normalize normFilter = new Normalize();
            normFilter.setInputFormat(data);
            data = Filter.useFilter(data, normFilter);
            data.setClassIndex(data.numAttributes() - 1);
        }
        filter = new Remove();
        filter.setAttributeIndices("" + data.numAttributes());
        filter.setInputFormat(data);
        data = Filter.useFilter(data, filter);

        return data;
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
            new PSODriver().run(30, filePath, filePathForWeka, configuration, ',', false);
            //Utils.nominalForm("data/glass.csv");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
