package PSO;

import utils.NCConstruct;
import utils.Utils;
import java.io.IOException;
import java.util.*;
import smile.validation.AdjustedRandIndex;

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
    public void runDummy() {
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
    }

    /**
     * main method to run PSO-based clustering
     * */
    public void run(int runs, String path, PSOConfiguration configuration, char sep, boolean removeFirst) throws IOException {
        int maxK;
        double[][] data;
        int[] labelsTrue;
        int skipLines = 0;

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
        // pick maxK 2-10% of total number of data point
        maxK = (int)Math.sqrt(data.length);

        // step 2 - pick objectives
        NCConstruct ncConstruct = new NCConstruct(data);
        Evaluator.Evaluation[] evaluation = {Evaluator.Evaluation.CONNECTIVITY, Evaluator.Evaluation.COHESION};
        Evaluator evaluator = new Evaluator();
        Problem problem = new Problem(data, evaluator);

        // step 3 - run PSO algorithm
        configuration.maxK = maxK;
        PSO pso = new PSO(problem, ncConstruct, evaluation, configuration);
        // constructed clusters
        int[] labelsPred = pso.execute();

        // step 4 - measure comparing to true labels
        AdjustedRandIndex adjustedRandIndex = new AdjustedRandIndex();
        HashMap<Integer,double[]> centroids = Utils.centroids(data, labelsPred);
        System.out.println("DB index score of PSO: " + Utils.dbIndexScore(centroids,labelsPred,data));
        System.out.println("ARI of PSO algorithm:  " + adjustedRandIndex.measure(labelsTrue, labelsPred));
        System.out.println(Arrays.toString(labelsTrue));
        System.out.println(Arrays.toString(labelsPred));


        // k-means baseline algorithm
        KMeans kMeans = new KMeans(data,problem.getN(), problem.getD(), maxK);
        kMeans.clustering(100);
        System.out.println("DB index score of PSO: " + Utils.dbIndexScore(centroids,kMeans.getLabels(),data));
        System.out.println("ARI of k-means baseline algorithm: " + adjustedRandIndex.measure(labelsTrue, kMeans.getLabels()));

        // optional step - write true and constructed labels into a file
        /*Utils.whenWriteStringUsingBufferedWritter_thenCorrect(Arrays.toString(labelsTrue) +
                System.getProperty("line.separator") + "," + Arrays.toString(labelsPred), "data/output.txt");*/

        // optional step - objectives of true clusters
        /*System.out.println("objectives of true clusters: " + Arrays.toString(problem.evaluate(
                new Solution(labelsTrue, Utils.distinctNumberOfItems(labelsTrue)), evaluation, new NCConstruct(data))));*/
    }


    public static void main(String[] args) throws Exception {
        //new PSODriver().runDummy();
        try {
            // test using UCI 'glass' public data set - https://archive.ics.uci.edu/ml/datasets/glass+identification
            // pick file manually or pass a path string
            boolean pickManually = false;
            String filePath;
            filePath = "data/yeast.csv";
            //filePath = "data/p-yeast.csv";
            String path = pickManually ? Utils.pickAFile(): filePath;
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
            new PSODriver().run(30, path, configuration,',',true);
            //Utils.nominalForm("data/glass.csv");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
