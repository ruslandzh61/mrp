package PSO;

import clustering.*;
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
    private double[][] dataAttrs;
    private int[] labelsTrue;
    AdjustedRandIndex adjustedRandIndex = new AdjustedRandIndex();

    private void processData(Dataset dataset) throws IOException {
        // read file
        char sep = ',';
        List<String[]> dataStr = Utils.readFile(dataset.getPath(), sep);
        if (dataset.getHeader() >= 0 && dataset.getHeader() < dataStr.size()) {
            dataStr.remove(dataset.getHeader());
        }
        assert (dataStr.size()>0);
        assert (dataStr.get(0).length>0);

        // extract true labels
        int D = dataStr.get(0).length;
        int labelCol = D - 1;
        labelsTrue = Utils.extractLabels(dataStr, labelCol);
        //System.out.println(Arrays.toString(labelsTrue));

        // extract attributes
        int[] excludedColumns;
        if (dataset.isRemoveFirst()) {
            excludedColumns = new int[]{0, dataStr.get(0).length - 1};
        } else {
            excludedColumns = new int[]{dataStr.get(0).length - 1};
        }

        dataAttrs = Utils.extractAttributes(dataStr, excludedColumns);
        /*for (double[] record: dataAttrs) {
            System.out.println(Arrays.toString(record));
        }*/

        /* normalize data */
        if (dataset.isNormalize()) {
            Utils.normalize(dataAttrs);
        }
    }

    /**
     * main method to run PSO-based buildClusterer
     * */
    private void run(int runs, Dataset dataset, PSOConfiguration configuration) throws Exception {
        /* process data */
        processData(dataset);
        Instances instances = Utils.getData(dataset);

        // step 2 - pick objectives
        NCConstruct ncConstruct = new NCConstruct(dataAttrs);
        Evaluator.Evaluation[] evaluation = {Evaluator.Evaluation.CONNECTIVITY, Evaluator.Evaluation.COHESION};
        Evaluator evaluator = new Evaluator();
        Problem problem = new Problem(this.dataAttrs, evaluator);
        configuration.maxK = (int) (Math.sqrt(problem.getData().length));
        boolean normObjectives = true;

        Random rnd = new Random(1);
        Reporter reporter = new Reporter(runs);

        for (int run = 1; run <= runs; ++run) {
            Experiment e;

            System.out.println("run: " + run);
            // step 3 - run PSO algorithm
            PSO pso = new PSO(problem, ncConstruct, evaluation, configuration, instances, labelsTrue, normObjectives);
            pso.setSeed(rnd.nextInt());
            // constructed clusters
            int[] labelsPred = Utils.adjustLabels(pso.execute());
            //int[] labelsPredCloned = labelsPred.clone();
            //Utils.removeNoise(labelsPred, this.dataAttrs, 2, 2.0);
            //Utils.adjustAssignments(labelsPred);

            // step 4 - measure comparing to true labels
            HashMap<Integer, double[]> centroids = Utils.centroids(this.dataAttrs, labelsPred);
            double aRIScore = this.adjustedRandIndex.measure(this.labelsTrue, labelsPred);
            double dbScore = Utils.dbIndexScore(centroids, labelsPred, this.dataAttrs);
            double silhScore = Utils.silhoutte(centroids, labelsPred, this.dataAttrs);
            int numClusters = Utils.distinctNumberOfItems(labelsPred);

            e = new Experiment(labelsPred, aRIScore, dbScore, silhScore, numClusters);
            reporter.set(run-1, e);

            System.out.println("ARI score of PSO for run:   " + Utils.doublePrecision(aRIScore, 4));
            System.out.println("DB score of PSO for run:    " + Utils.doublePrecision(dbScore, 4));
            System.out.println("Silhoutte score of PSO run: " + Utils.doublePrecision(silhScore, 4));
            System.out.println("number of clusters for run: " + numClusters);

            // test hill-climber
            /*int[] labelsPredCloned = labelsPred.clone();
            centroids = Utils.centroids(this.dataAttrs, labelsPredCloned);
            double[][] initialCentroids = new double[centroids.size()][];
            initialCentroids = centroids.values().toArray(initialCentroids);
            clustering.KMeans kMeans = new clustering.KMeans(initialCentroids.length, 2.0);
            kMeans.setInitializationMethod(clustering.KMeans.Initialization.HILL_CLIMBER);
            kMeans.setInitial(initialCentroids);
            kMeans.buildClusterer(this.dataAttrs);
            labelsPred = Utils.adjustLabels(kMeans.getLabels());
            Utils.removeNoise(labelsPred, this.dataAttrs, 2, 2.0);
            Utils.adjustAssignments(labelsPred);

            centroids = Utils.centroids(this.dataAttrs, labelsPred);
            aRIScore = adjustedRandIndex.measure(labelsTrue, labelsPred);
            dbScore = Utils.dbIndexScore(centroids, labelsPred, this.dataAttrs);
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

        reporter.compute();
        Experiment mean = reporter.getMean();
        Experiment stdDev = reporter.getStdDev();

        System.out.println("mean and std dev of ARI score:          " + Utils.doublePrecision(mean.getAri(), 4) +
                " +- " + Utils.doublePrecision(stdDev.getAri(), 4));
        System.out.println("mean and std dev of DB Index score:     " + Utils.doublePrecision(mean.getDb(), 4) +
                " +- " + Utils.doublePrecision(stdDev.getDb(), 4));
        System.out.println("mean and std dev of Silhouette score:   " + Utils.doublePrecision(mean.getSilh(), 4) +
                " +- " + Utils.doublePrecision(stdDev.getSilh(), 4));
        System.out.println("mean and std dev of number of clusters: " + Utils.doublePrecision(mean.getK(), 4) +
                " +- " + Utils.doublePrecision(stdDev.getK(), 4));
        System.out.println("--------------------------");
    }

    public static void main(String[] args) throws Exception {
        //new PSODriver().runDummy();
        try {
            // test using UCI 'glass' public data set - https://archive.ics.uci.edu/ml/datasets/glass+identification
            // pick file manually or pass a path string
            int runs = 15;
            Dataset dataset = Dataset.GLASS;
            PSOConfiguration configuration = new PSOConfiguration();
            new PSODriver().run(runs, dataset, configuration);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}