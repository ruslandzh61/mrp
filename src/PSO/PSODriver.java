package PSO;

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

    public void runDummy() {
        int maxK;
        double[][] data;
        int[] labels;
        labels = new int[]{0,0,0,0,0,0,1,1,1,2};
        data = new double[][]{
                {2,2}, {3,3}, {3,1}, {4,2}, {1.6,-0.5},
                {3.01, -1.5}, {-4, 2}, {-2, 2}, {-3, 3},{7,7}
        };
        maxK = 6;
        Evaluator.Evaluation[] evaluation = {Evaluator.Evaluation.CONNECTIVITY, Evaluator.Evaluation.COHESION};
        Evaluator evaluator = new Evaluator();
        Problem problem = new Problem(data, evaluator);
        PSO pso = new PSO(problem, evaluation, maxK);
        int[] labelsPred = pso.execute();
        System.out.println(new AdjustedRandIndex().measure(labels, labelsPred));
    }

    /* most of data points end up in the same cluster */
    public void run(String path) throws IOException {
        int maxK;
        double[][] data;
        int[] labelsTrue;
        int skipLines = 0;

        /* process data */
        // step 1 - read data from file
        List<String[]> dataStr = Utils.readDataFromCustomSeperator(path, skipLines, ',');
        assert (dataStr.size()>0);
        assert (dataStr.get(0).length>0);

        // extract labels
        labelsTrue = Utils.extractLabels(dataStr,dataStr.get(0).length-1);
        //System.out.println(Arrays.toString(labelsTrue));

        // exclude columns from csv file
        int[] excludedColumns = {0,dataStr.get(0).length-1};
        data = Utils.extractAttributes(dataStr, excludedColumns);
        // pick maxK 2-10% of total number of data point
        maxK = data.length/20;

        // step 2 - pick objectives
        Evaluator.Evaluation[] evaluation = {Evaluator.Evaluation.CONNECTIVITY, Evaluator.Evaluation.COHESION};
        Evaluator evaluator = new Evaluator();
        Problem problem = new Problem(data, evaluator);

        // step 3 - run PSO algorithm
        PSO pso = new PSO(problem, evaluation, maxK);
        // constructed clusters
        int[] labelsPred = pso.execute();

        // step 4 - measure comparing to true labels
        AdjustedRandIndex adjustedRandIndex = new AdjustedRandIndex();
        System.out.println("ARI of PSO algorithm: " + adjustedRandIndex.measure(labelsTrue, labelsPred));
        System.out.println(Arrays.toString(labelsTrue));
        System.out.println(Arrays.toString(labelsPred));


        // k-means baseline algorithm
        KMeans kMeans = new KMeans(data,problem.getN(), problem.getD(), maxK);
        kMeans.clustering(100);
        System.out.println("ARI of k-means baseline algorithm: " + adjustedRandIndex.measure(labelsTrue, kMeans.getLabels()));

        // optional step - write true and constructed labels into a file
        /*Utils.whenWriteStringUsingBufferedWritter_thenCorrect(Arrays.toString(labelsTrue) +
                System.getProperty("line.separator") + "," + Arrays.toString(labelsPred), "data/output.csv");*/

        // optional step - objectives of true clusters
        /*System.out.println("objectives of true clusters: " + Arrays.toString(problem.evaluate(
                new Solution(labelsTrue, Utils.distinctNumberOfItems(labelsTrue)), evaluation, new NCConstruct(data))));*/
    }


    public static void main(String[] args) {
        //new PSODriver().runDummy();
        try {
            // pick file manually or pass a path string
            boolean pickManually = false;
            String path = pickManually ? Utils.pickAFile(): "data/glass.csv";
            new PSODriver().run(path);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
