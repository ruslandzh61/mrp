package PSO;

import clustering.*;
import utils.NCConstruct;
import utils.Utils;

import javax.security.auth.login.Configuration;
import java.util.*;

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
public class PSODriver extends Analyzer {

    private PSOConfiguration configuration;

    public void setSeedStartFrom(int seedStartFrom) {
        this.seedStartFrom = seedStartFrom;
    }

    private int seedStartFrom;
    //List<List<Experiment>> iterations;

    public PSODriver(PSOConfiguration aConf) {
        this.configuration = aConf;
    }

    @Override
    protected void setRuns(int runs) {
        super.setRuns(runs);
        //iterations = new ArrayList<>(runs);
    }

    /**
     * main method to run PSO-based buildClusterer
     * */
    public void run() throws Exception {
        assert (reporter != null);
        assert (dataset != null);
        /* process data */
        processData();

        // step 2 - pick objectives
        long startTime = System.currentTimeMillis();
        NCConstruct ncConstruct = new NCConstruct(dataAttrs);
        long endTime = System.currentTimeMillis();
        System.out.println("TIME:" + ((endTime - startTime) / 1000.0)  / 60);

        Evaluator.Evaluation[] evaluations = {Evaluator.Evaluation.CONNECTIVITY, Evaluator.Evaluation.COHESION};
        Evaluator evaluator = new Evaluator();
        Problem problem = new Problem(this.dataAttrs, evaluator);
        configuration.maxK = (int) (Math.sqrt(problem.getData().length));
        Random rnd = new Random(1);

        // skip several seed values in order to start from
        for (int i = 0; i < this.seedStartFrom; ++i) {
            rnd.nextInt();
        }

        for (int run = 1; run <= reporter.size(); ++run) {
            System.out.println("RUN: " + run);
            startTime = System.currentTimeMillis();

            // step 3 - run PSO algorithm
            PSO pso = new PSO(problem, ncConstruct, evaluations, configuration, labelsTrue);
            pso.setSeed(rnd.nextInt());
            // constructed clusters
            int[] labelsPred = Utils.adjustLabels(pso.execute());
            //int[] labelsPredCloned = labelsPred.clone();

            // step 4 - measure comparing to true labels
            Experiment e = measure(labelsPred);
            endTime = System.currentTimeMillis();
            System.out.println("TIME:" + ((endTime - startTime) / 1000.0)  / 60);
            System.out.println("A:" + e.getAri());
            System.out.println("D:" + e.getDb());
            System.out.println("S:" + e.getSilh());
            System.out.println("K:" + e.getK());
            reporter.set(run-1, e);

            //iterations.add(pso.iterationsBest);
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
    }

    public void setConfiguration(PSOConfiguration configuration) {
        this.configuration = configuration;
    }

    public static void main(String[] args) throws Exception {
        System.out.println("PSO");
        PSOConfiguration conf = PSOConfiguration.valueOf(args[0]);
        Dataset dataset =  Dataset.valueOf(args[1]); //Dataset.S1;
        int seedStartFrom = Integer.parseInt(args[2]);
        int runs = Integer.parseInt(args[3]);

        System.out.println("Dataset: " + dataset.name());
        String solutionsFilePath = "results/PSO/pso" + conf.name() + "_" + dataset.name() + "-" + seedStartFrom + "-" + runs + ".txt";

        PSODriver psoDriver = new PSODriver(conf);
        psoDriver.setDataset(dataset);
        psoDriver.setRuns(runs);
        psoDriver.setSeedStartFrom(seedStartFrom);
        psoDriver.run();
        System.out.println("AVERAGE OVER RUNS");
        psoDriver.analyze(true);
        psoDriver.saveResults(solutionsFilePath);
        /*int startIdx = Integer.parseInt(args[0]); // inclusively
        int endIdx = Integer.parseInt(args[1]); // exclusively
        Dataset[] allDatasets = Dataset.values();
        int runs = Integer.parseInt(args[2]);
        int startSeedFrom = Integer.parseInt(args[3]);
        String solutionsFilePath;
        PSOConfiguration conf = PSOConfiguration.CONF1;
        solutionsFilePath = "results/PSO/pso" + conf.name() + "_" + startIdx + "-" + endIdx + "-" + runs + "-" + startSeedFrom + ".txt";
        System.out.println("CONFIGURATION: " + conf.name());
        for (int i = startIdx; i < endIdx; ++i) {
            Dataset dataset = allDatasets[i];
            System.out.println("DATASET: " + dataset.name());
            PSODriver psoDriver = new PSODriver(conf);
            psoDriver.setDataset(dataset);
            psoDriver.setRuns(runs);
            psoDriver.setSeedStartFrom(startSeedFrom);
            psoDriver.run();
            System.out.println("AVERAGE OVER RUNS");
            psoDriver.analyze(true);
            //psoDriver.saveResults(solutionsFilePath);
        }*/
    }
}