package GA;

import smile.validation.AdjustedRandIndex;
import utils.Utils;
import weka.clusterers.ClusterEvaluation;
import weka.clusterers.DensityBasedClusterer;
import weka.clusterers.EM;
import weka.clusterers.GenClustPlusPlus;
//import weka.core.*;
import weka.core.*;
import weka.core.converters.ConverterUtils;
import weka.core.pmml.jaxbbindings.Cluster;
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

        //dataClusterer.setClassIndex(dataClusterer.numAttributes() - 1);
        //System.out.println(dataClusterer);

        // step 3 - build model
        Random rnd = new Random(1);
        for (int run = 1; run <= runs; ++run) {
            cl = new MyGenClustPlusPlus();
            cl.setSeed(rnd.nextInt());
            cl.buildClusterer(dataClusterer);

            labelsPred = Utils.adjustLabels(cl.getLabels());
            temp = Arrays.toString(labelsPred);
            output.append(temp.substring(1, temp.length() - 1)).append(System.getProperty("line.separator"));

            HashMap<Integer, double[]> myCentroids = Utils.centroids(dataArr, labelsPred);

            // step 4 - measure
            double ARI = adjustedRandIndex.measure(labelsTrue, labelsPred);
            double myDBWithMyCentroids = Utils.dbIndexScore(myCentroids, labelsPred, dataArr);

            meanARI += ARI;
            meanMyDBWithMyCentroids += myDBWithMyCentroids;

            System.out.println("run " + run + ": " + Arrays.toString(labelsPred));
            System.out.println("ARI score: " + ARI);
            System.out.println("DB score:  " + myDBWithMyCentroids);
        }


        // step 4 - measure comparing to true labels
        //System.out.println("DB Index score: " + cl.calcularDavidBouldin().getResultado());
        System.out.println("mean ARI score: " + meanARI/runs);
        System.out.println("mean DB score: " + meanMyDBWithMyCentroids/runs);

        //System.out.println(Arrays.toString(labelsTrue));
        //System.out.println(Arrays.toString(cl.getLabels()));
        Utils.whenWriteStringUsingBufferedWritter_thenCorrect(output.toString() +
                System.getProperty("line.separator"), "../datasets/output.csv");
    }

    public static void main(String[] args) throws Exception {
        // weka doesn't work with other separators other than ','
        // (false, false) - first attribute not removed, not normalized
        GADriver gaDriver = new GADriver(30, "data/p-yeast.csv","data/yeast.csv", ',', true, false);
    }
}
