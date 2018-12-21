package GA;

import PSO.Problem;
import clustering.Dataset;
import clustering.Evaluator;
import clustering.Experiment;
import clustering.Reporter;
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
public class GADriver {
    private double[][] dataAttrs;
    private int[] labelsTrue;

    private AdjustedRandIndex adjustedRandIndex = new AdjustedRandIndex();

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
        System.out.println(Arrays.toString(labelsTrue));

        // extract attributes
        int[] excludedColumns;
        if (dataset.isRemoveFirst()) {
            excludedColumns = new int[]{0, dataStr.get(0).length - 1};
        } else {
            excludedColumns = new int[]{dataStr.get(0).length - 1};
        }

        dataAttrs = Utils.extractAttributes(dataStr, excludedColumns);
        for (double[] record: dataAttrs) {
            System.out.println(Arrays.toString(record));
        }

        /* normalize data */
        if (dataset.isNormalize()) {
            Utils.normalize(dataAttrs);
        }
    }

    public void run(boolean myGenClust, int runs, Dataset dataset) throws Exception {
        MyGenClustPlusPlus cl;
        GenClustPlusPlus gl;
        char sep = ',';

        //StringBuffer output= new StringBuffer();
        //String temp;

        // step 1 - retrieve true labels
        List<String[]> dataStr = Utils.readFile(dataset.getPath(), sep);
        assert (dataStr.size() > 0);
        assert (dataStr.get(0).length > 0);
        labelsTrue = Utils.extractLabels(dataStr, dataStr.get(0).length - 1);

        //temp = Arrays.toString(labelsTrue);
        //output.append(temp.substring(1,temp.length()-1)+System.getProperty("line.separator"));

        // step 2 - preprocess data


        // step 3 - build model
        processData(dataset);
        Instances dataClusterer = Utils.getData(dataset);

        // step 2 - pick objectives
        NCConstruct ncConstruct = new NCConstruct(dataAttrs);
        Evaluator.Evaluation[] evaluations = {Evaluator.Evaluation.CONNECTIVITY, Evaluator.Evaluation.COHESION};
        Evaluator evaluator = new Evaluator();

        System.out.println(Arrays.toString(labelsTrue));
        Random rnd = new Random(1);
        Reporter reporter = new Reporter(runs);

        for (int run = 1; run <= runs; ++run) {
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
                cl.setFitnessType(MyGenClustPlusPlus.FITNESS.MULTIOBJECTIVE_SUM);
                cl.setDistance(2.0);
                cl.buildClusterer(dataClusterer);
                labelsPred = Utils.adjustLabels(cl.getLabels());
            } else {
                gl = new GenClustPlusPlus();
                gl.setStartChromosomeSelectionGeneration(11);
                gl.setSeed(rnd.nextInt());
                gl.buildClusterer(dataClusterer);
                labelsPred = new int[dataClusterer.size()];
                for (int i = 0; i < dataClusterer.size(); ++i) {
                    Instance instance = dataClusterer.get(i);
                    labelsPred[i] = gl.clusterInstance(instance);
                }
                labelsPred = Utils.adjustLabels(labelsPred);
            }

            //Utils.removeNoise(labelsPred, dataArr, 2, 2.0);
            //Utils.adjustAssignments(labelsPred);

            //temp = Arrays.toString(labelsPred);
            //output.append(temp.substring(1, temp.length() - 1)).append(System.getProperty("line.separator"));

            HashMap<Integer, double[]> centroids = Utils.centroids(this.dataAttrs, labelsPred);
            double aRIScore = this.adjustedRandIndex.measure(this.labelsTrue, labelsPred);
            double dbScore = Utils.dbIndexScore(centroids, labelsPred, this.dataAttrs);
            double silhScore = Utils.silhoutte(centroids, labelsPred, this.dataAttrs);
            int numClusters = Utils.distinctNumberOfItems(labelsPred);

            e = new Experiment(labelsPred, aRIScore, dbScore, silhScore, numClusters);
            reporter.set(run-1, e);

            System.out.println("RUN: " + run);
            System.out.println("ARI score of PSO for run:   " + Utils.doublePrecision(aRIScore, 4));
            System.out.println("DB score of PSO for run:    " + Utils.doublePrecision(dbScore, 4));
            System.out.println("Silhoutte score of PSO run: " + Utils.doublePrecision(silhScore, 4));
            System.out.println("number of clusters for run: " + numClusters);
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

        //System.out.println(Arrays.toString(labelsTrue));
        //System.out.println(Arrays.toString(cl.getLabels()));
        //Utils.whenWriteStringUsingBufferedWritter_thenCorrect(output.toString() +
          //      System.getProperty("line.separator"), "../datasets/output.csv");
    }

    public static void main(String[] args) throws Exception {
        // weka doesn't work with other separators other than ','
        // (false, false) - first attribute not removed, not normalized
        Dataset[] datasets = {Dataset.GLASS};
        for (Dataset dataset: datasets) {
            System.out.println("dataset: " + dataset.getPath());
            //System.out.println("WEKA GENCLUST++");
            //new GADriver(false, 15, filePathForWeka, filePath, removeFirst, normalize);
            System.out.println("MY GENCLUST++");
            new GADriver().run(true, 15, dataset);
        }
    }
}