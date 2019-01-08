package clustering;

import smile.validation.AdjustedRandIndex;
import utils.Silh;
import utils.Utils;
import weka.core.Instances;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

/**
 * Abstart class Analyzer used for conducting experiments on clustering algoritms
 */

public abstract class Analyzer {
    protected Dataset dataset;
    protected double[][] dataAttrs;
    protected Instances wekaData;
    protected int[] labelsTrue;
    protected Reporter reporter;
    protected int seedStartFrom;
    private Experiment mean, stdDev;
    private AdjustedRandIndex adjustedRandIndex = new AdjustedRandIndex();
    private Silh silhoutte = new Silh();

    public abstract void run() throws Exception;

    public void analyze(boolean print) {
        this.reporter.compute();
        mean = this.reporter.getMean();
        stdDev = this.reporter.getStdDev();

        if (print) {
            System.out.println("------- ANALYSIS --------");

            if (this.reporter.size() == 1) {
                System.out.println("C: " + Arrays.toString(this.reporter.get(0).getSolution()));
            }
            System.out.println("A: " + Utils.doublePrecision(mean.getAri(), 4) +
                    " +- " + Utils.doublePrecision(stdDev.getAri(), 4));
            System.out.println("D: " + Utils.doublePrecision(mean.getDb(), 4) +
                    " +- " + Utils.doublePrecision(stdDev.getDb(), 4));
            System.out.println("S: " + Utils.doublePrecision(mean.getSilh(), 4) +
                    " +- " + Utils.doublePrecision(stdDev.getSilh(), 4));
            System.out.println("K: " + Utils.doublePrecision(mean.getK(), 4) +
                    " +- " + Utils.doublePrecision(stdDev.getK(), 4));
        }
    }

    protected void setRuns(int runs) {
        reporter = new Reporter(runs);
    }

    protected void setDataAttrs(double[][] aDataAttrs) {
        this.dataAttrs = aDataAttrs;
    }

    protected void setLabelsTrue(int[] labelsTrue) {
        this.labelsTrue = labelsTrue;
    }

    protected void setDataset(Dataset aDataset) {
        this.dataset = aDataset;
    }

    protected void processData() throws Exception {
        assert (dataset != null);
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
        /*for (double[] record: dataAttrs) {
            System.out.println(Arrays.toString(record));
        }*/

        /* normalize data */
        if (dataset.isNormalize()) {
            Utils.normalize(dataAttrs);
        }

        this.wekaData = Utils.getData(dataset);
    }



    protected void saveResults(String solutionsFilePath) throws Exception {
        StringBuilder solutionsLog = new StringBuilder();

        solutionsLog.append(dataset.name() + " " + reporter.size() + System.lineSeparator());
        for (int j = 0; j < reporter.size(); ++j) {
            solutionsLog.append(reporter.get(j).getTime() + System.lineSeparator());
            solutionsLog.append(Arrays.toString(reporter.get(j).getSolution()) + System.lineSeparator());
        }

        Utils.whenWriteStringUsingBufferedWritter_thenCorrect(solutionsLog.toString(), solutionsFilePath, true);
        //ExcelRW.write(resultFilePath, reporter.getExperiments(), this.dataset);
    }

    protected Experiment getMean() {
        return mean;
    }

    protected Experiment getStdDev() {
        return stdDev;
    }

    protected void setSeedStartFrom(int seedStartFrom) {
        this.seedStartFrom = seedStartFrom;
    }

    protected Experiment measure(int[] labelsPred) {
        HashMap<Integer, double[]> centroids = Utils.centroids(this.dataAttrs, labelsPred);
        double aRIScore = this.adjustedRandIndex.measure(this.labelsTrue, labelsPred);
        double dbScore = Utils.dbIndexScore(centroids, labelsPred, this.dataAttrs);
        double silhScore = silhoutte.compute(centroids, labelsPred, this.dataAttrs);
        int numClusters = Utils.distinctNumberOfItems(labelsPred);

        /*for (int i: centroids.keySet()) {
            System.out.println(Arrays.toString(centroids.get(i)));
        }*/
        /*System.out.println("solution: " + Arrays.toString(labelsPred));
        System.out.println("ARI score of PSO for run: " + Utils.doublePrecision(aRIScore, 4));
        System.out.println("DB score of PSO for run: " + Utils.doublePrecision(dbScore, 4));
        System.out.println("Silhoutte score of PSO run: " + Utils.doublePrecision(silhScore, 4));
        System.out.println("number of clusters for run: " + numClusters);*/

        return new Experiment(labelsPred, aRIScore, dbScore, silhScore, numClusters);
    }
}
