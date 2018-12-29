package GA;

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
        mgaC1(11,20, true, false, MyGenClustPlusPlus.FITNESS.DBINDEX),
        mgaC2(11,60, true, false, MyGenClustPlusPlus.FITNESS.DBINDEX),
        mgaC3(30,60, true, false, MyGenClustPlusPlus.FITNESS.DBINDEX),
        mgaC4(50,60, true, false, MyGenClustPlusPlus.FITNESS.DBINDEX),
        mgaC5(11,20, true, true, MyGenClustPlusPlus.FITNESS.DBINDEX),
        mgaC6(11,60, true, true, MyGenClustPlusPlus.FITNESS.DBINDEX),
        mgaC7(30,60, true, true, MyGenClustPlusPlus.FITNESS.DBINDEX),
        mgaC8(50,60, true, true, MyGenClustPlusPlus.FITNESS.DBINDEX),
        mgaC9(11,20, false, true, MyGenClustPlusPlus.FITNESS.DBINDEX),
        mgaC10(11,60, false, true, MyGenClustPlusPlus.FITNESS.DBINDEX),
        mgaC11(30,60, false, true, MyGenClustPlusPlus.FITNESS.DBINDEX),
        mgaC12(50,60, false, true, MyGenClustPlusPlus.FITNESS.DBINDEX),
        mgaC13(11,20, true, true, MyGenClustPlusPlus.FITNESS.MULTIOBJECTIVE_SUM),
        mgaC14(11,60, false, true, MyGenClustPlusPlus.FITNESS.MULTIOBJECTIVE_SUM),
        mgaC15(30,60, true, true, MyGenClustPlusPlus.FITNESS.MULTIOBJECTIVE_SUM),
        mgaC16(50,60, true, true, MyGenClustPlusPlus.FITNESS.MULTIOBJECTIVE_SUM),
        mgaC17(11,20, false, true, MyGenClustPlusPlus.FITNESS.MULTIOBJECTIVE_SUM),
        mgaC18(11,60, false, true, MyGenClustPlusPlus.FITNESS.MULTIOBJECTIVE_SUM),
        mgaC19(30,60, false, true, MyGenClustPlusPlus.FITNESS.MULTIOBJECTIVE_SUM),
        mgaC20(50,60, false, true, MyGenClustPlusPlus.FITNESS.MULTIOBJECTIVE_SUM);
        //mgaC21(50,60, true, false, MyGenClustPlusPlus.FITNESS.MULTIOBJECTIVE_SUM),
        //mgaC22(11,20, false, false, MyGenClustPlusPlus.FITNESS.MULTIOBJECTIVE_SUM);

        int chrSelectionGen;
        int generations;
        boolean normObjs;
        boolean maximin;

        GaConfiguration(int chrSelectionGen, int generations, boolean normObjs, boolean maximin, MyGenClustPlusPlus.FITNESS fitness) {
            this.chrSelectionGen = chrSelectionGen;
            this.generations = generations;
            this.normObjs = normObjs;
            this.maximin = maximin;
            this.fitness = fitness;
        }

        MyGenClustPlusPlus.FITNESS fitness;
    }

    public static String[] confValuesStr() {
        GaConfiguration[] configurations = GaConfiguration.values();
        String[] res = new String[configurations.length];
        for (int i = 0; i < configurations.length; ++i) {
            res[i] = configurations[i].name();
        }
        return res;
    }

    private boolean myGenClust;

    public void setGaConfiguration(GaConfiguration gaConfiguration) {
        this.gaConfiguration = gaConfiguration;
    }

    private GaConfiguration gaConfiguration;

    public void setStartChrSelection(int startChrSelection) {
        this.startChrSelection = startChrSelection;
    }

    private int startChrSelection;

    public GADriver(boolean myGenClust) {
        this.myGenClust = myGenClust;
    }

    public void run() throws Exception {
        assert (reporter != null);
        assert (dataset != null);

        processData();

        MyGenClustPlusPlus cl;
        GenClustPlusPlus gl;
        Random rnd = new Random(1);

        // step 2 - pick objectives

        NCConstruct ncConstruct = null;
        Evaluator.Evaluation[] evaluations = null;
        Evaluator evaluator = null;
        if (myGenClust) {
            ncConstruct = new NCConstruct(dataAttrs);
            evaluations = new Evaluator.Evaluation[]{Evaluator.Evaluation.CONNECTIVITY, Evaluator.Evaluation.COHESION};
            evaluator = new Evaluator();
        }

        for (int run = 1; run <= reporter.size(); ++run) {
            System.out.println("RUN: " + run);
            long startTime = System.currentTimeMillis();
            Experiment e;
            int[] labelsPred;
            if (myGenClust) {
                cl = new MyGenClustPlusPlus();
                cl.setSeed(rnd.nextInt());
                cl.setNcConstruct(ncConstruct);
                cl.setEvaluator(evaluator);
                cl.setEvaluations(evaluations);
                cl.setMyData(this.dataAttrs);
                cl.setTrueLabels(labelsTrue);
                cl.setStartChromosomeSelectionGeneration(gaConfiguration.chrSelectionGen);
                cl.setNumGenerations(gaConfiguration.generations);
                cl.setNormalizeObjectives(gaConfiguration.normObjs);
                cl.setHillClimb(true);
                cl.setMaximin(gaConfiguration.maximin);
                cl.setFitnessType(gaConfiguration.fitness);
                cl.setDistance(2.0);
                cl.buildClusterer(this.wekaData);
                labelsPred = Utils.adjustLabels(cl.getLabels());
            } else {
                gl = new GenClustPlusPlus();
                gl.setStartChromosomeSelectionGeneration(this.startChrSelection);
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
            System.out.println("TIME:" + ((endTime - startTime) / 1000.0)  / 60);
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

    public void setMyGenClust(boolean myGenClust) {
        this.myGenClust = myGenClust;
    }

    public static void main(String[] args) throws Exception {
        int runs = 30;
        String solutionsFilePath = "results/ga11.txt";

        /*for (Dataset dataset:  new Dataset[]{Dataset.S1, Dataset.S2, Dataset.S3, Dataset.S4}) {
            System.out.println("DATASET: " + dataset.getPath());
            GADriver gaDriver = new GADriver(false);
            gaDriver.setDataset(dataset);
            gaDriver.setRuns(runs);
            gaDriver.setStartChrSelection(11);
            gaDriver.run();
            gaDriver.analyze(true);
            gaDriver.saveResults(resultFilePath, solutionsFilePath);
        }*/

        runs = 5;
        Dataset[] datasets = {Dataset.DERMATOLOGY};
        GaConfiguration[] gaConfigurations = {GaConfiguration.mgaC13};
        for (GaConfiguration conf: gaConfigurations) {
            System.out.println("configuration: " + conf.name());
            for (Dataset dataset: datasets) {
                System.out.println("DATASET: " + dataset.getPath());
                solutionsFilePath = "results/" + conf.name() + ".txt";
                GADriver driver = new GADriver(true);
                driver.setDataset(dataset);
                driver.setGaConfiguration(conf);
                driver.setRuns(runs);
                driver.run();
                driver.analyze(true);
                driver.saveResults(solutionsFilePath);
            }
        }
    }
}