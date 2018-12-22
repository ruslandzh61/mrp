package clustering;

import smile.validation.AdjustedRandIndex;
import utils.Utils;
import weka.core.Instances;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;

/**
 * Created by rusland on 20.12.18.
 */

public abstract class Analyzer {
    protected double[][] dataAttrs;
    protected Instances wekaData;
    protected int[] labelsTrue;
    protected AdjustedRandIndex adjustedRandIndex = new AdjustedRandIndex();
    Experiment[] experiments;
    protected Reporter reporter;

    enum Algorithm {
        KMEANS, GENCLUST, MGENCLUST, MCPSO;
    }

    protected void processData(Dataset dataset) throws Exception {
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
        //System.out.println(Arrays.toString(labelsTrue));

        // extract attributes
        int[] excludedColumns;
        if (dataset.isRemoveFirst()) {
            excludedColumns = new int[]{0, dataStr.get(0).length - 1};
        } else {
            excludedColumns = new int[]{dataStr.get(0).length - 1};
        }

        dataAttrs = Utils.extractAttributes(dataStr, excludedColumns);
        /*for (double[] record: dataAttrs) {
            System.out.println(Arrays.toString(record));
        }*/

        /* normalize data */
        if (dataset.isNormalize()) {
            Utils.normalize(dataAttrs);
        }

        this.wekaData = Utils.getData(dataset);
    }

    public abstract void run(int runs, Dataset dataset) throws Exception;

    public void analyze() {
        this.reporter.compute();
        Experiment mean = this.reporter.getMean();
        Experiment stdDev = this.reporter.getStdDev();

        System.out.println("mean and std dev of ARI score:          " + Utils.doublePrecision(mean.getAri(), 4) +
                " +- " + Utils.doublePrecision(stdDev.getAri(), 4));
        System.out.println("mean and std dev of DB Index score:     " + Utils.doublePrecision(mean.getDb(), 4) +
                " +- " + Utils.doublePrecision(stdDev.getDb(), 4));
        System.out.println("mean and std dev of Silhouette score:   " + Utils.doublePrecision(mean.getSilh(), 4) +
                " +- " + Utils.doublePrecision(stdDev.getSilh(), 4));
        System.out.println("mean and std dev of number of clusters: " + Utils.doublePrecision(mean.getK(), 4) +
                " +- " + Utils.doublePrecision(stdDev.getK(), 4));
        System.out.println("--------------------------");
    }

    protected Experiment measure(int[] labelsPred) {
        HashMap<Integer, double[]> centroids = Utils.centroids(this.dataAttrs, labelsPred);
        double aRIScore = this.adjustedRandIndex.measure(this.labelsTrue, labelsPred);
        double dbScore = Utils.dbIndexScore(centroids, labelsPred, this.dataAttrs);
        double silhScore = Utils.silhoutte(centroids, labelsPred, this.dataAttrs);
        int numClusters = Utils.distinctNumberOfItems(labelsPred);

        System.out.println("ARI score of PSO for run:   " + Utils.doublePrecision(aRIScore, 4));
        System.out.println("DB score of PSO for run:    " + Utils.doublePrecision(dbScore, 4));
        System.out.println("Silhoutte score of PSO run: " + Utils.doublePrecision(silhScore, 4));
        System.out.println("number of clusters for run: " + numClusters);

        return new Experiment(labelsPred, aRIScore, dbScore, silhScore, numClusters);
    }
}
