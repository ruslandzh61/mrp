package GA;

import clustering.*;
import utils.NCConstruct;
import utils.Utils;
import weka.clusterers.GenClustPlusPlus;
import weka.core.*;
import java.util.*;

/**
 * Created by rusland on 23.11.18.
 */
public class GADriver extends Analyzer {
    private boolean myGenClust;

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
        NCConstruct ncConstruct = new NCConstruct(dataAttrs);
        Evaluator.Evaluation[] evaluations = {Evaluator.Evaluation.CONNECTIVITY, Evaluator.Evaluation.COHESION};
        Evaluator evaluator = new Evaluator();

        for (int run = 1; run <= reporter.size(); ++run) {
            System.out.println("RUN: " + run);
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
                cl.setNormalizeObjectives(false);
                cl.setHillClimb(true);
                cl.setMaximin(true);
                cl.setFitnessType(MyGenClustPlusPlus.FITNESS.MULTIOBJECTIVE_SUM);
                cl.setDistance(2.0);
                cl.buildClusterer(this.wekaData);
                labelsPred = Utils.adjustLabels(cl.getLabels());
            } else {
                gl = new GenClustPlusPlus();
                gl.setStartChromosomeSelectionGeneration(11);
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
        Dataset[] datasets = {Dataset.DERMATOLOGY};
        boolean mGenClust = true;
        int runs = 10;
        if (mGenClust) {
            System.out.println("MODIFIED GENCLUST++");
        } else {
            System.out.println("WEKA GENCLUST++");
        }

        for (Dataset dataset: datasets) {
            System.out.println("DATASET: " + dataset.getPath());
            GADriver gaDriver = new GADriver(mGenClust);
            gaDriver.setDataset(dataset);
            gaDriver.setRuns(runs);
            gaDriver.run();
            gaDriver.analyze(true);
        }
    }
}