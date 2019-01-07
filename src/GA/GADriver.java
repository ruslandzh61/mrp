package GA;

import PSO.PSOConfiguration;
import clustering.*;
import utils.NCConstruct;
import utils.Utils;
import weka.clusterers.GenClustPlusPlus;
import weka.core.*;
import java.util.*;

import static clustering.Dataset.S4;

/**
 * Created by rusland on 23.11.18.
 */
public class GADriver extends Analyzer {
    public enum  GaConfiguration {
        // C1-C5: close to original
        // c6-c10: maximin with db index, norm objs
        // c11-c15: maximin with db index, do not norm objs
        // c16-c20: maximin with sum, norm objs
        // c21-c25: maximin with sum, do not norm objs
        GA(),
        mgaC1(10,20, true, false, MyGenClustPlusPlus.FITNESS.DBINDEX),
        mgaC2(10,60, true, false, MyGenClustPlusPlus.FITNESS.DBINDEX),
        mgaC3(30,60, true, false, MyGenClustPlusPlus.FITNESS.DBINDEX),
        mgaC4(50,60, true, false, MyGenClustPlusPlus.FITNESS.DBINDEX),
        mgaC5(10,20, true, true, MyGenClustPlusPlus.FITNESS.DBINDEX),
        mgaC6(10,60, true, true, MyGenClustPlusPlus.FITNESS.DBINDEX),
        mgaC7(30,60, true, true, MyGenClustPlusPlus.FITNESS.DBINDEX),
        mgaC8(50,60, true, true, MyGenClustPlusPlus.FITNESS.DBINDEX),
        mgaC9(10,20, false, true, MyGenClustPlusPlus.FITNESS.DBINDEX),
        mgaC10(10,60, false, true, MyGenClustPlusPlus.FITNESS.DBINDEX),
        mgaC11(30,60, false, true, MyGenClustPlusPlus.FITNESS.DBINDEX),
        mgaC12(50,60, false, true, MyGenClustPlusPlus.FITNESS.DBINDEX),
        mgaC13(10,20, true, true, MyGenClustPlusPlus.FITNESS.MULTIOBJECTIVE_SUM),
        mgaC14(10,60, false, true, MyGenClustPlusPlus.FITNESS.MULTIOBJECTIVE_SUM),
        mgaC15(30,60, true, true, MyGenClustPlusPlus.FITNESS.MULTIOBJECTIVE_SUM),
        mgaC16(50,60, true, true, MyGenClustPlusPlus.FITNESS.MULTIOBJECTIVE_SUM),
        mgaC17(10,20, false, true, MyGenClustPlusPlus.FITNESS.MULTIOBJECTIVE_SUM),
        mgaC18(10,60, false, true, MyGenClustPlusPlus.FITNESS.MULTIOBJECTIVE_SUM),
        mgaC19(30,60, false, true, MyGenClustPlusPlus.FITNESS.MULTIOBJECTIVE_SUM),
        mgaC20(50,60, false, true, MyGenClustPlusPlus.FITNESS.MULTIOBJECTIVE_SUM),
        mgaC21(50,60, true, false, MyGenClustPlusPlus.FITNESS.MULTIOBJECTIVE_SUM),
        mgaC22(10,20, false, false, MyGenClustPlusPlus.FITNESS.MULTIOBJECTIVE_SUM),

        mgaC23(10,20, true,true, MyGenClustPlusPlus.FITNESS.MULTIOBJECTIVE_SUM, new double[]{0.9,0.1}),
        mgaC24(10,20, true,true, MyGenClustPlusPlus.FITNESS.MULTIOBJECTIVE_SUM, new double[]{0.6,0.4}),
        mgaC25(10,20, true,true, MyGenClustPlusPlus.FITNESS.MULTIOBJECTIVE_SUM, new double[]{0.2,0.8}),
        mgaC26(10,20, true,true, MyGenClustPlusPlus.FITNESS.MULTIOBJECTIVE_SUM, new double[]{0.4,0.6}),
        mgaC27(10,20, true,true, MyGenClustPlusPlus.FITNESS.MULTIOBJECTIVE_SUM, new double[]{0.8,0.2}),
        mgaC28(10,20, true,true, MyGenClustPlusPlus.FITNESS.MULTIOBJECTIVE_SUM, new double[]{0.1,0.9}),
        mgaC29(10,20, true,true, MyGenClustPlusPlus.FITNESS.MULTIOBJECTIVE_SUM, new double[]{0.05,0.95}),

        mgaC30(10,20, true,false, MyGenClustPlusPlus.FITNESS.MULTIOBJECTIVE_SUM, new double[]{0.9,0.1}),
        mgaC31(10,20, true,false, MyGenClustPlusPlus.FITNESS.MULTIOBJECTIVE_SUM, new double[]{0.6,0.4}),
        mgaC32(10,20, true,false, MyGenClustPlusPlus.FITNESS.MULTIOBJECTIVE_SUM, new double[]{0.2,0.8}),
        mgaC33(10,20, true,false, MyGenClustPlusPlus.FITNESS.MULTIOBJECTIVE_SUM, new double[]{0.4,0.6}),
        mgaC34(10,20, true,false, MyGenClustPlusPlus.FITNESS.MULTIOBJECTIVE_SUM, new double[]{0.8,0.2}),
        mgaC35(10,20, true,false, MyGenClustPlusPlus.FITNESS.MULTIOBJECTIVE_SUM, new double[]{0.1,0.9}),
        mgaC36(10,20, true,false, MyGenClustPlusPlus.FITNESS.MULTIOBJECTIVE_SUM, new double[]{0.7,0.3}),
        mgaC37(10,20, true,false, MyGenClustPlusPlus.FITNESS.MULTIOBJECTIVE_SUM, new double[]{0.3,0.7}),
        mgaC38(10,20, true,false, MyGenClustPlusPlus.FITNESS.MULTIOBJECTIVE_SUM, new double[]{0.05,0.95}),
        mgaC39(10,20, true,false, MyGenClustPlusPlus.FITNESS.MULTIOBJECTIVE_SUM);

        private int chrSelectionGen;
        private int generations;
        private boolean normObjs;
        private boolean maximin;
        private MyGenClustPlusPlus.FITNESS fitness;
        private double[] w;

        GaConfiguration(int chrSelectionGen, int generations, boolean normObjs, boolean maximin, MyGenClustPlusPlus.FITNESS fitness) {
            this(chrSelectionGen,generations,normObjs,maximin, fitness,new double[]{0.5, 0.5});
        }

        GaConfiguration(int chrSelectionGen, int generations, boolean normObjs, boolean maximin,
                        MyGenClustPlusPlus.FITNESS fitness, double[] aW) {
            this.chrSelectionGen = chrSelectionGen;
            this.generations = generations;
            this.normObjs = normObjs;
            this.maximin = maximin;
            this.fitness = fitness;
            this.w = aW.clone();
        }

        GaConfiguration() {
        }
    }
    private GaConfiguration gaConfiguration;

    public void setGaConfiguration(GaConfiguration gaConfiguration) {
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
                cl.setNormalizeObjectives(gaConfiguration.normObjs);
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


        //System.out.println(Arrays.toString(labelsTrue));
        //System.out.println(Arrays.toString(cl.getLabels()));
        //Utils.whenWriteStringUsingBufferedWritter_thenCorrect(output.toString() +
          //      System.getProperty("line.separator"), "../datasets/output.csv");
    }

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

    public static void main(String[] args) throws Exception {
        System.out.println("GA");
        String confStr = args[0];
        GaConfiguration conf = GaConfiguration.valueOf(confStr);
        Dataset dataset = Dataset.valueOf(args[1]);
        int seedStartFrom = Integer.parseInt(args[2]);
        int runs = Integer.parseInt(args[3]);
        if (conf != GaConfiguration.GA) {
            String solutionsFilePath = "results/mGA2/" + conf.name() + "_" + dataset.name() + "-" + seedStartFrom + "-" + runs + ".txt";
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
        /*String config = args[0];
        int startIdx = Integer.parseInt(args[1]); // inclusively
        int endIdx = Integer.parseInt(args[2]); // exclusively
        Dataset[] allDatasets = Dataset.values();
        int runs = Integer.parseInt(args[3]);
        int startSeedFrom = Integer.parseInt(args[4]);
        String solutionsFilePath;
        GaConfiguration conf = GaConfiguration.valueOf(config);
        assert (conf != null);
        solutionsFilePath = "results/mGA/" + conf.name() + "_" + startIdx + "-" + endIdx + "-" + runs + "-" + startSeedFrom + ".txt";
        System.out.println(solutionsFilePath + " will be created");
        for (int i = startIdx; i < endIdx; ++i) {
            long overallStartTime = System.currentTimeMillis();

            Dataset dataset = allDatasets[i];
            System.out.println("DATASET: " + dataset.name());
            GADriver driver = new GADriver(true);
            driver.setDataset(dataset);
            driver.setGaConfiguration(conf);
            driver.setRuns(runs);
            driver.run();
            driver.analyze(true);
            driver.saveResults(solutionsFilePath);

            long overallEndTime = System.currentTimeMillis();
            double overallTime = ((overallEndTime - overallStartTime) / 1000.0)  / 60;
            System.out.println("Overall time for dataset " + dataset.name() + ": " + overallTime);
        }*/

        /*for (Dataset dataset: datasets) {
            System.out.println("DATASET: " + dataset.getPath());
            GADriver gaDriver = new GADriver(false);
            gaDriver.setDataset(dataset);
            gaDriver.setRuns(runs);
            gaDriver.setStartChrSelection(11);
            gaDriver.run();
            gaDriver.analyze(true);
            //gaDriver.saveResults(resultFilePath, solutionsFilePath);
        }*/

        /*runs = 10;
        GaConfiguration[] gaConfigurations = GaConfiguration.values();
        for (GaConfiguration conf: gaConfigurations) {
            System.out.println("configuration: " + conf.name());
            for (Dataset dataset: datasets) {
                System.out.println("DATASET: " + dataset.getPath());
                solutionsFilePath = "results/mGA/tuning/" + conf.name() + ".txt";
                GADriver driver = new GADriver(true);
                driver.setDataset(dataset);
                driver.setGaConfiguration(conf);
                driver.setRuns(runs);
                driver.run();
                driver.analyze(true);
                driver.saveResults(solutionsFilePath);
            }
        }*/
    }
}