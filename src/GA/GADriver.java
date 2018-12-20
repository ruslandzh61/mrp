package GA;

import clustering.Evaluator;
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

import java.util.*;

/**
 * Created by rusland on 23.11.18.
 */
public class GADriver {

    public GADriver(boolean myGenClust, int runs, String filename, String filenameForTrueLabels,
                    boolean removeFirst,  boolean normalize) throws Exception {
        Instances data;
        MyGenClustPlusPlus cl;
        GenClustPlusPlus gl;
        Remove filter;
        char sep = ',';

        AdjustedRandIndex adjustedRandIndex = new AdjustedRandIndex();
        double meanMyDBWithMyCentroids = 0.0;
        double meanARI = 0.0;
        double meanK = 0.0;
        double[] sdofARI = new double[runs];
        double[] sdofDB = new double[runs];
        double[] sdOfNumClusters = new double[runs];
        int[] labelsTrue, labelsPred;
        StringBuffer output= new StringBuffer();
        String temp;

        // step 1 - retrieve true labels
        List<String[]> dataStr = Utils.readDataFromCustomSeperator(filenameForTrueLabels, sep);
        assert (dataStr.size() > 0);
        assert (dataStr.get(0).length > 0);
        labelsTrue = Utils.extractLabels(dataStr, dataStr.get(0).length - 1);

        temp = Arrays.toString(labelsTrue);
        output.append(temp.substring(1,temp.length()-1)+System.getProperty("line.separator"));

        // retrieve data in two-dim array format
        int[] excludedColumns;
        if (removeFirst) {
            excludedColumns = new int[]{0, dataStr.get(0).length - 1};
        } else {
            excludedColumns = new int[]{dataStr.get(0).length - 1};
        }

        double[][] dataArr = Utils.extractAttributes(dataStr, excludedColumns);

        if (normalize) {
            Utils.normalize(dataArr);
        }

        // step 2 - preprocess data
        data = ConverterUtils.DataSource.read(filename);
        data.setClassIndex(data.numAttributes() - 1);
        if (removeFirst) {
            filter = new Remove();
            filter.setAttributeIndices("1");
            filter.setInputFormat(data);
            data = Filter.useFilter(data, filter);
            data.setClassIndex(data.numAttributes() - 1);
        }

        if (normalize) {
            Normalize normFilter = new Normalize();
            normFilter.setInputFormat(data);
            data = Filter.useFilter(data, normFilter);
            data.setClassIndex(data.numAttributes() - 1);
        }

        filter = new Remove();
        //filter.setAttributeIndicesArray(new int[]{0, data.numAttributes()-1});
        filter.setAttributeIndices("" + data.numAttributes());
        filter.setInputFormat(data);
        Instances dataClusterer = Filter.useFilter(data, filter);
        /*double[][] myData = new double[dataClusterer.size()][];
        int var1Idx = 0;
        for (Instance instance: dataClusterer) {
            myData[var1Idx++] = instance.toDoubleArray();
        }*/

        //dataClusterer.setClassIndex(dataClusterer.numAttributes() - 1);
        //System.out.println(dataClusterer);

        // step 3 - build model
        NCConstruct ncConstruct = new NCConstruct(dataArr);
        Evaluator.Evaluation[] evaluations = {Evaluator.Evaluation.CONNECTIVITY, Evaluator.Evaluation.COHESION};
        Evaluator evaluator = new Evaluator();

        System.out.println(Arrays.toString(labelsTrue));
        Random rnd = new Random(1);
        for (int run = 1; run <= runs; ++run) {
            if (myGenClust) {
                cl = new MyGenClustPlusPlus();
                cl.setSeed(rnd.nextInt());
                cl.setNcConstruct(ncConstruct);
                cl.setEvaluator(evaluator);
                cl.setEvaluations(evaluations);
                cl.setMyData(dataArr);
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

            temp = Arrays.toString(labelsPred);
            output.append(temp.substring(1, temp.length() - 1)).append(System.getProperty("line.separator"));

            HashMap<Integer, double[]> myCentroids = Utils.centroids(dataArr, labelsPred);

            // step 4 - measure
            double ARI = adjustedRandIndex.measure(labelsTrue, labelsPred);
            double myDBWithMyCentroids = Utils.dbIndexScore(myCentroids, labelsPred, dataArr);
            double k = Utils.distinctNumberOfItems(labelsPred);
            meanARI += ARI;
            meanMyDBWithMyCentroids += myDBWithMyCentroids;
            meanK += k;
            sdofARI[run-1] = ARI;
            sdofDB[run-1] = myDBWithMyCentroids;
            sdOfNumClusters[run-1] = k;

            System.out.println("RUN: " + run);
            System.out.println(Arrays.toString(labelsPred));
            System.out.println("ARI score: " + Utils.doublePrecision(ARI, 4));
            System.out.println("DB score:  " + Utils.doublePrecision(myDBWithMyCentroids, 4));
            System.out.println("num of clusters: " + k);
        }

        System.out.println("ARI: " + Utils.doublePrecision(meanARI/runs, 4) + "+-" +
                Utils.doublePrecision(Utils.standardDeviation(sdofARI), 4));
        System.out.println("DB: " + Utils.doublePrecision(meanMyDBWithMyCentroids/runs, 4) + "+-" +
                Utils.doublePrecision(Utils.standardDeviation(sdofDB), 4));
        System.out.println("K: " + Utils.doublePrecision(meanK/runs, 4) + "+-" +
                Utils.doublePrecision(Utils.standardDeviation(sdOfNumClusters), 4));

        //System.out.println(Arrays.toString(labelsTrue));
        //System.out.println(Arrays.toString(cl.getLabels()));
        //Utils.whenWriteStringUsingBufferedWritter_thenCorrect(output.toString() +
          //      System.getProperty("line.separator"), "../datasets/output.csv");
    }

    public static void main(String[] args) throws Exception {
        // weka doesn't work with other separators other than ','
        // (false, false) - first attribute not removed, not normalized
        String[][] datasetPaths = {
                {"data/glass.csv", "data/p-glass.csv"},
                {"data/dermatology.csv", "data/p-dermatology.csv"},
                {"data/flame.csv", "data/p-flame.csv"},
                {"data/compound.csv", "data/p-compound.csv"},
        };
        int[] datasetIDs = {3};
        for (int dID: datasetIDs) {
            System.out.println("dataset: " + datasetPaths[dID][0]);
            String filePath = datasetPaths[dID][0];
            String filePathForWeka = datasetPaths[dID][1];
            boolean removeFirst = false;
            boolean normalize = true;
            //System.out.println("WEKA GENCLUST++");
            //new GADriver(false, 15, filePathForWeka, filePath, removeFirst, normalize);
            System.out.println("MY GENCLUST++");
            new GADriver(true, 15, filePathForWeka, filePath, removeFirst, normalize);
        }
    }
}