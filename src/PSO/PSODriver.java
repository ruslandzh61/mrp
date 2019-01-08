package PSO;

import clustering.*;
import utils.NCConstruct;
import utils.Utils;
import java.util.*;

/**
 * Extends Analyzer class to PSODriver runs experiments on clustering algorithm based on
 * PSO-based algorithm (Armani,2016): 'Multiobjective buildClusterer analysis using particle swarm optimization'
 *
 * MaxiMin algorithm (Li, 2007):
 *      'Better Spread and Convergence: Particle Swarm Multiobjective Optimization Using the Maximin Fitness Function'
 */
public class PSODriver extends Analyzer {
    private PSOConfiguration configuration;

    /**
     * main method to run PSO-based clusterer
     * */
    public void run() throws Exception {
        assert (reporter != null);
        assert (dataset != null);
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
        }
    }

    void setConfiguration(PSOConfiguration configuration) {
        this.configuration = configuration;
    }

    public static void main(String[] args) throws Exception {
        System.out.println("PSO");
        PSOConfiguration conf = PSOConfiguration.valueOf(args[0]);
        Dataset dataset =  Dataset.valueOf(args[1]);
        int seedStartFrom = Integer.parseInt(args[2]);
        int runs = Integer.parseInt(args[3]);

        System.out.println("Dataset: " + dataset.name());
        String solutionsFilePath = "results/PSO/pso" + conf.name() + "_" + dataset.name() + "-" + seedStartFrom + "-" + runs + ".txt";

        PSODriver psoDriver = new PSODriver();
        psoDriver.setConfiguration(conf);
        psoDriver.setDataset(dataset);
        psoDriver.setRuns(runs);
        psoDriver.setSeedStartFrom(seedStartFrom);
        psoDriver.run();
        System.out.println("AVERAGE OVER RUNS");
        psoDriver.analyze(true);
        psoDriver.saveResults(solutionsFilePath);
    }
}