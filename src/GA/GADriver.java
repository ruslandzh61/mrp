package GA;

import PSO.Problem;
import clustering.*;
import smile.validation.AdjustedRandIndex;
import utils.NCConstruct;
import utils.Utils;
//import weka.core.*;
import weka.clusterers.GenClustPlusPlus;
import weka.core.*;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.Remove;

import java.io.IOException;
import java.util.*;

/**
 * Created by rusland on 23.11.18.
 */
public class GADriver extends Analyzer {
    private boolean myGenClust;

    public GADriver(boolean myGenClust) {
        this.myGenClust = myGenClust;
    }

    public void run(int runs, Dataset dataset) throws Exception {
        MyGenClustPlusPlus cl;
        GenClustPlusPlus gl;

        processData(dataset);

        Random rnd = new Random(1);
        this.reporter = new Reporter(runs);

        // step 2 - pick objectives
        NCConstruct ncConstruct = new NCConstruct(dataAttrs);
        Evaluator.Evaluation[] evaluations = {Evaluator.Evaluation.CONNECTIVITY, Evaluator.Evaluation.COHESION};
        Evaluator evaluator = new Evaluator();

        for (int run = 1; run <= runs; ++run) {
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
                cl.setFitnessNormalize(true);
                cl.setNormalizeObjectives(true);
                cl.setHillClimb(true);
                cl.setMaximin(true);
                cl.setFitnessType(MyGenClustPlusPlus.FITNESS.DBINDEX);
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

            //Utils.removeNoise(labelsPred, dataArr, 2, 2.0);
            //Utils.adjustAssignments(labelsPred);

            //temp = Arrays.toString(labelsPred);
            //output.append(temp.substring(1, temp.length() - 1)).append(System.getProperty("line.separator"));

            e = this.measure(labelsPred);
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
        Dataset[] datasets = {Dataset.GLASS};
        boolean mGenClust = true;
        int runs = 15;
        if (mGenClust) {
            System.out.println("MODIFIED GENCLUST++");
        } else {
            System.out.println("WEKA GENCLUST++");
        }

        for (Dataset dataset: datasets) {
            System.out.println("DATASET: " + dataset.getPath());
            GADriver gaDriver = new GADriver(mGenClust);
            gaDriver.run(runs, dataset);
            gaDriver.analyze();
        }
    }
}