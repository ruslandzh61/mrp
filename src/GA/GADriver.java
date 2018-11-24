package GA;

import smile.validation.AdjustedRandIndex;
import utils.Utils;
import weka.clusterers.ClusterEvaluation;
import weka.clusterers.DensityBasedClusterer;
import weka.clusterers.EM;
//import weka.clusterers.GenClustPlusPlus;
//import weka.core.*;
import weka.clusterers.GenClustPlusPlus;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.Remove;

import java.util.Arrays;
import java.util.List;

/**
 * Created by rusland on 23.11.18.
 */
public class GADriver {

    public GADriver(String filename) throws Exception {
        ClusterEvaluation eval;
        Instances               data;
        String[]                options;
        GenClustPlusPlus cl;
        double                  logLikelyhood;
        Remove filter;

        data = ConverterUtils.DataSource.read(filename);
        data.setClassIndex(data.numAttributes() - 1);
        /*filter = new Remove();
        filter.setAttributeIndices("1");
        filter.setInputFormat(data);
        data = Filter.useFilter(data, filter);*/

        /*Normalize normFilter = new Normalize();
        normFilter.setInputFormat(data);
        data = Filter.useFilter(data,normFilter);
        data.setClassIndex(data.numAttributes() - 1);*/

        filter = new Remove();
        //filter.setAttributeIndicesArray(new int[]{0, data.numAttributes()-1});
        filter.setAttributeIndices(""+data.numAttributes());
        filter.setInputFormat(data);
        Instances dataClusterer = Filter.useFilter(data, filter);

        //dataClusterer.setClassIndex(dataClusterer.numAttributes() - 1);
        //System.out.println(dataClusterer);

        cl   = new GenClustPlusPlus();
        cl.setSeed(20);
        cl.buildClusterer(dataClusterer);
        eval = new ClusterEvaluation();
        eval.setClusterer(cl);
        eval.evaluateClusterer(new Instances(data));
        System.out.println(eval.clusterResultsToString());

        int[] labelsTrue = new int[data.size()];
        int[] labelsPred = new int[data.size()];
        List<String[]> dataStr = Utils.readDataFromCustomSeperator("data/glass.csv", ',');
        assert (dataStr.size()>0);
        assert (dataStr.get(0).length>0);
        labelsTrue = Utils.extractLabels(dataStr,dataStr.get(0).length-1);
        // extract labels
        for (int i = 0; i <data.size(); ++i) {
            labelsPred[i] = cl.clusterInstance(dataClusterer.get(i));
        }

        // step 4 - measure comparing to true labels
        AdjustedRandIndex adjustedRandIndex = new AdjustedRandIndex();
        System.out.println("ARI of PSO algorithm: " + adjustedRandIndex.measure(labelsTrue, labelsPred));
        System.out.println(Arrays.toString(labelsTrue));
        System.out.println(Arrays.toString(labelsPred));

        Utils.whenWriteStringUsingBufferedWritter_thenCorrect(Arrays.toString(labelsTrue) +
                System.getProperty("line.separator") + "," + Arrays.toString(labelsPred), "data/output.txt");
    }

    public static void main(String[] args) throws Exception {
        new GADriver("data/p-glass.csv");
    }
}