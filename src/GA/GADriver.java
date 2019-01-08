package GA;

import clustering.*;
import utils.NCConstruct;
import utils.Utils;
import weka.clusterers.GenClustPlusPlus;
import weka.core.*;
import java.util.*;

/**
 * Extends Analyzer to run experiments on GenClustPlusPlus algorithm and its modified versions
 */
public class GADriver extends Analyzer {

    private GaConfiguration gaConfiguration;

    void setGaConfiguration(GaConfiguration gaConfiguration) {
        this.gaConfiguration = gaConfiguration;
    }

    public void run() throws Exception {
        assert (reporter != null);
        assert (dataset != null);

        processData();

        MyGenClustPlusPlus cl;
        GenClustPlusPlus gl;
        Random rnd = new Random(1);

        for (int i = 0; i < this.seedStartFrom; ++i) {
            rnd.nextInt();
        }

        // step 2 - pick objectives
        NCConstruct ncConstruct = null;
        Evaluator.Evaluation[] evaluations = null;
        Evaluator evaluator = null;
        if (this.gaConfiguration != GaConfiguration.GA) {
            ncConstruct = new NCConstruct(dataAttrs);
            evaluations = new Evaluator.Evaluation[]{Evaluator.Evaluation.CONNECTIVITY, Evaluator.Evaluation.COHESION};
            evaluator = new Evaluator();
        }
        for (int run = 1; run <= reporter.size(); ++run) {
            System.out.println("RUN: " + run);
            long startTime = System.currentTimeMillis();
            Experiment e;
            int[] labelsPred;
            if (this.gaConfiguration != GaConfiguration.GA) {
                cl = new MyGenClustPlusPlus();
                cl.setSeed(rnd.nextInt());
                cl.setNcConstruct(ncConstruct);
                cl.setEvaluator(evaluator);
                cl.setEvaluations(evaluations);
                cl.setMyData(this.dataAttrs);
                cl.setTrueLabels(labelsTrue);
                cl.setkMeansInit(KMeans.Initialization.RANDOM);
                cl.setHillClimb(true);
                cl.setDistance(1.0);

                cl.setStartChromosomeSelectionGeneration(gaConfiguration.chrSelectionGen);
                cl.setNumGenerations(gaConfiguration.generations);
                cl.setMaximin(gaConfiguration.maximin);
                cl.setFitnessType(gaConfiguration.fitness);
                cl.setEvaluationWeights(gaConfiguration.w);

                cl.buildClusterer(this.wekaData);
                labelsPred = Utils.adjustLabels(cl.getLabels());
            } else {
                gl = new GenClustPlusPlus();
                gl.setStartChromosomeSelectionGeneration(10);
                gl.setNumGenerations(20);
                gl.setSeed(rnd.nextInt());
                gl.buildClusterer(this.wekaData);
                labelsPred = new int[this.wekaData.size()];
                for (int i = 0; i < this.wekaData.size(); ++i) {
                    Instance instance = this.wekaData.get(i);
                    labelsPred[i] = gl.clusterInstance(instance);
                }
                labelsPred = Utils.adjustLabels(labelsPred);
            }

            Utils.removeNoise(labelsPred, dataAttrs, 2, 2.0);
            Utils.adjustAssignments(labelsPred);

            //temp = Arrays.toString(labelsPred);
            //output.append(temp.substring(1, temp.length() - 1)).append(System.getProperty("line.separator"));

            e = this.measure(labelsPred);
            long endTime = System.currentTimeMillis();
            double time = ((endTime - startTime) / 1000.0)  / 60;
            e.setTime(time);
            System.out.println("TIME:" + time);
            System.out.println("A:" + e.getAri());
            System.out.println("D:" + e.getDb());
            System.out.println("S:" + e.getSilh());
            System.out.println("K:" + e.getK());
            reporter.set(run-1, e);
        }
    }

    /**
     *
     * @param dataset Dataset
     * @param runs number of runs
     * @param seedStartFrom seed to start experiments from
     * @throws Exception
     */
    static void testGA(Dataset dataset, int runs, int seedStartFrom) throws Exception {
        String solutionsFilePath = "results/GA/ga" + "_" + dataset.name() + "-" + seedStartFrom + "-" + runs + ".txt";

        GADriver driver = new GADriver();
        driver.setDataset(dataset);
        driver.setRuns(runs);
        driver.setGaConfiguration(GaConfiguration.GA);
        driver.setSeedStartFrom(seedStartFrom);
        driver.run();
        driver.analyze(true);
        driver.saveResults(solutionsFilePath);
    }

    /**
     *
     * Method to run experiments on WEKA GenClustPlusPlus and modified GenClustPlusPlus.
     * Clustering solutions are saved into txt file. Passed arguments are included in file's name.
     * @param args - array of arguments: args[0] is a name of configuration; args[1] is a name of dataset;
     *  args[2] is a seed to start experiments from; args[3] is a number of runs
     */
    public static void main(String[] args) throws Exception {
        String confStr = args[0];
        GaConfiguration conf = GaConfiguration.valueOf(confStr);
        Dataset dataset = Dataset.valueOf(args[1]);
        int seedStartFrom = Integer.parseInt(args[2]);
        int runs = Integer.parseInt(args[3]);
        if (conf != GaConfiguration.GA) {
            String solutionsFilePath = "results/mGA/" + conf.name() + "_" + dataset.name() + "-" + seedStartFrom + "-" + runs + ".txt";
            GADriver driver = new GADriver();
            driver.setDataset(dataset);
            driver.setGaConfiguration(conf);
            driver.setRuns(runs);
            driver.setSeedStartFrom(seedStartFrom);
            driver.run();
            driver.analyze(true);
            driver.saveResults(solutionsFilePath);
        } else {
            testGA(dataset, runs, seedStartFrom);
        }
    }
}