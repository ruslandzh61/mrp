package GA;

import clustering.Evaluator;
import smile.validation.AdjustedRandIndex;
import utils.NCConstruct;
import utils.Utils;
//import weka.core.*;
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

    public GADriver(int runs, String filename, String filenameForTrueLabels, char sep,
                    boolean removeFirst,  boolean normalize) throws Exception {
        Instances data;
        MyGenClustPlusPlus cl;
        Remove filter;

        AdjustedRandIndex adjustedRandIndex = new AdjustedRandIndex();
        double meanMyDBWithMyCentroids = 0.0;
        double meanARI = 0.0;
        double meanK = 0.0;
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

        Random rnd = new Random(1);
        for (int run = 1; run <= runs; ++run) {

            cl = new MyGenClustPlusPlus();
            cl.setSeed(rnd.nextInt());
            cl.setNcConstruct(ncConstruct);
            cl.setEvaluator(evaluator);
            cl.setEvaluations(evaluations);
            cl.setMyData(dataArr);
            cl.setTrueLabels(labelsTrue);
            cl.setNormalizeObjectives(true);
            cl.buildClusterer(dataClusterer);

            labelsPred = Utils.adjustLabels(cl.getLabels());
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

            System.out.println("RUN: " + run);
            System.out.println("ARI score: " + Utils.doublePrecision(ARI, 4));
            System.out.println("DB score:  " + Utils.doublePrecision(myDBWithMyCentroids, 4));
            System.out.println("num of clusters: " + k);
        }


        // step 4 - measure comparing to true labels
        //System.out.println("DB Index score: " + cl.calcularDavidBouldin().getResultado());
        System.out.println("mean ARI score: " + Utils.doublePrecision(meanARI/runs, 4));
        System.out.println("mean DB score: " + Utils.doublePrecision(meanMyDBWithMyCentroids/runs, 4));
        System.out.println("mean # of clusters: " + Utils.doublePrecision(meanK/runs, 4));

        //System.out.println(Arrays.toString(labelsTrue));
        //System.out.println(Arrays.toString(cl.getLabels()));
        Utils.whenWriteStringUsingBufferedWritter_thenCorrect(output.toString() +
                System.getProperty("line.separator"), "../datasets/output.csv");
    }

    public static void main(String[] args) throws Exception {
        // weka doesn't work with other separators other than ','
        // (false, false) - first attribute not removed, not normalized

        GADriver gaDriver = new GADriver(15, "data/p-glass.csv","data/glass.csv", ',', false, false);
    }
}