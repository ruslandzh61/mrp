package utils;

import GA.GADriver;
import clustering.Dataset;
import clustering.Experiment;
import clustering.Reporter;
import smile.validation.AdjustedRandIndex;

import javax.xml.crypto.Data;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

/**
 * Generate Excel file with experimental results
 */
public class ResultsGenerator {
    Dataset[] datasets;
    List<double[][]> dataAttrsList;

    AdjustedRandIndex adjustedRandIndex = new AdjustedRandIndex();
    Silh silhoutte = new Silh();

    public ResultsGenerator(Dataset[] aDatasets) throws IOException {
        datasets = aDatasets;
        this.dataAttrsList = new ArrayList<>(datasets.length);
        processDatasetData();
    }

    private void processDatasetData() throws IOException {
        for (int i = 0; i < datasets.length; ++i) {
            Dataset dataset = datasets[i];
            char sep = ',';
            List<String[]> dataStr = Utils.readFile(dataset.getPath(), sep);
            if (dataset.getHeader() >= 0 && dataset.getHeader() < dataStr.size()) {
                dataStr.remove(dataset.getHeader());
            }
            assert (dataStr.size() > 0);
            assert (dataStr.get(0).length > 0);

            // extract true labels
            int D = dataStr.get(0).length;
            int labelCol = D - 1;
            dataset.setLabels(Utils.extractLabels(dataStr, labelCol));

            // extract attributes
            int[] excludedColumns;
            if (dataset.isRemoveFirst()) {
                excludedColumns = new int[]{0, dataStr.get(0).length - 1};
            } else {
                excludedColumns = new int[]{dataStr.get(0).length - 1};
            }

            dataAttrsList.add(Utils.extractAttributes(dataStr, excludedColumns));
            if (dataset.isNormalize()) {
                Utils.normalize(dataAttrsList.get(i));
            }
        }
    }

    /** @param i - dataset index
     * @param labelsPred - data points assignment produced by clustering algorithm
      */
    protected Experiment measure(int i, int[] labelsPred, Dataset dataset) {
        HashMap<Integer, double[]> centroids = Utils.centroids(this.dataAttrsList.get(i), labelsPred);
        double aRIScore = this.adjustedRandIndex.measure(this.datasets[i].getLabels(), labelsPred);
        double dbScore = Utils.dbIndexScore(centroids, labelsPred, this.dataAttrsList.get(i));
        double silhScore = silhoutte.compute(centroids, labelsPred, this.dataAttrsList.get(i));
        int numClusters = Utils.distinctNumberOfItems(labelsPred);
        int kDiff = Math.abs(dataset.getK()-numClusters);

        /*for (int i: centroids.keySet()) {
            System.out.println(Arrays.toString(centroids.get(i)));
        }*/
        /*System.out.println("solution: " + Arrays.toString(labelsPred));
        System.out.println("ARI score of PSO for run: " + Utils.doublePrecision(aRIScore, 4));
        System.out.println("DB score of PSO for run: " + Utils.doublePrecision(dbScore, 4));
        System.out.println("Silhoutte score of PSO run: " + Utils.doublePrecision(silhScore, 4));
        System.out.println("number of clusters for run: " + numClusters);*/

        return new Experiment(labelsPred, aRIScore, dbScore, silhScore, numClusters, kDiff);
    }

    /**
     * Generates results in Excel from clustering solutions stored in txt file;
     * DATASETS SHOULD BE IN THE SAME ORDER AS IN THE TXT FILE AND VALUE OF RUNS SHOULD MATCH NUMBER OF RUNS IN TXT FILE;
     * last two rows in fileName.xls correspond to mean and standard deviation
     * @param folderPath - folder path relative to the project root
     * @param fileName - file name where clustering solutions are stored
     * @param runs - number of runs
     * @param includesRuns - indicates whether txt file stores number of runs next name of corresponding dataset
     * @param includesTrueLabels - indicates whether true labels are stored in the file on the line after dataset name
     * @param includesTime - indicates whether file stores algorithm running time for each run
     * @throws Exception
     */
    public void generate(String folderPath, String fileName, int runs, boolean includesRuns, boolean includesTrueLabels, boolean includesTime) throws Exception { // folder "results/mGA/tuning"
        Experiment[] experiments;
        Experiment[] confMeans = new Experiment[datasets.length];
        Experiment[] confStdDevs = new Experiment[datasets.length];
        System.out.println(folderPath + fileName);
        experiments = new Experiment[runs];
        String filePath = folderPath + fileName + ".txt";
        HashMap<String, int[][]> datasetTosolutions = Utils.readSolutionFromFile(filePath, runs, includesRuns, includesTrueLabels, includesTime, datasets);

        int datasetIdx = 0;
        for (Dataset dataset: datasets) {
            System.out.println(dataset);
            int[][] expSols = datasetTosolutions.get(dataset.name());
            Reporter reporter = new Reporter(expSols.length);
            int expSolIdx = 0;
            for (int[] expSol: expSols) {
                //System.out.println(Arrays.toString(expSol));
                experiments[expSolIdx] = measure(datasetIdx, expSol, dataset);
                reporter.set(expSolIdx, experiments[expSolIdx]);
                ++expSolIdx;
            }
            reporter.compute();

            confMeans[datasetIdx] = reporter.getMean();
            confStdDevs[datasetIdx] = reporter.getStdDev();
            String excelFilePath = folderPath + fileName + ".xls";
            ExcelRW.write(excelFilePath, experiments, datasets[datasetIdx].name());
            utils.Utils.experimentsToCsv(folderPath+dataset.name()+".csv", experiments);
            ++datasetIdx;
        }

        String[][] datasetMeanStdDevsAverage = new String[datasets.length+1][5];
        String[] d;
        for (int i = 0; i < datasets.length; ++i) {
            d = datasetMeanStdDevsAverage[i];
            d[0] = utils.Utils.doublePrecision(confMeans[i].getAri(), 4) + " +- " + utils.Utils.doublePrecision(confStdDevs[i].getAri(), 4);
            d[1] = utils.Utils.doublePrecision(confMeans[i].getDb(), 4) + " +- " + utils.Utils.doublePrecision(confStdDevs[i].getDb(), 4);
            d[2] = utils.Utils.doublePrecision(confMeans[i].getSilh(), 4) + " +- " + utils.Utils.doublePrecision(confStdDevs[i].getSilh(), 4);
            d[3] = utils.Utils.doublePrecision(confMeans[i].getK(), 4) + " +- " + utils.Utils.doublePrecision(confStdDevs[i].getK(), 4);
            d[4] = utils.Utils.doublePrecision(confMeans[i].getKDiff(), 4) + " +- " + utils.Utils.doublePrecision(confStdDevs[i].getKDiff(), 4);
        }
        String excelFilePath = folderPath + "datasetMeanStdDevs.xls";

        // average mean for all datasets
        Reporter reporter = new Reporter(datasets.length);
        for (int j = 0; j < confMeans.length; ++j) {
            reporter.set(j, confMeans[j]);
        }
        reporter.compute();
        Experiment confMeanOverDatasets = reporter.getMean();

        // average std dev for all datasets
        reporter = new Reporter(datasets.length);
        for (int j = 0; j < confStdDevs.length; ++j) {
            reporter.set(j, confStdDevs[j]);
        }
        reporter.compute();
        Experiment confStdDevOverDatasets = reporter.getMean();

        d = datasetMeanStdDevsAverage[datasetMeanStdDevsAverage.length-1];
        d[0] = utils.Utils.doublePrecision(confMeanOverDatasets.getAri(), 4) + " +- " + utils.Utils.doublePrecision(confStdDevOverDatasets.getAri(), 4);
        d[1] = utils.Utils.doublePrecision(confMeanOverDatasets.getDb(), 4) + " +- " + utils.Utils.doublePrecision(confStdDevOverDatasets.getDb(), 4);
        d[2] = utils.Utils.doublePrecision(confMeanOverDatasets.getSilh(), 4) + " +- " + utils.Utils.doublePrecision(confStdDevOverDatasets.getSilh(), 4);
        d[3] = utils.Utils.doublePrecision(confMeanOverDatasets.getK(), 4) + " +- " + utils.Utils.doublePrecision(confStdDevOverDatasets.getK(), 4);
        d[4] = utils.Utils.doublePrecision(confMeanOverDatasets.getKDiff(), 4) + " +- " + utils.Utils.doublePrecision(confStdDevOverDatasets.getKDiff(), 4);

        String[] datasetNames = new String[datasets.length+1];
        for (int i = 0; i < datasets.length; ++i) {
            datasetNames[i] = datasets[i].name();
        }
        datasetNames[datasetNames.length-1] = "Average";
        ExcelRW.write(excelFilePath, datasetNames, datasetMeanStdDevsAverage);
    }

    /**
     * DATASETS SHOULD BE IN THE SAME ORDER AS IN THE TXT FILE AND VALUE OF RUNS SHOULD MATCH NUMBER OF RUNS IN TXT FILE
     * @param args - no arguments
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {
        String folderPath = "results/mGA2/";
        String fileName = "mga";
        int runs = 30;
        Dataset[] datasets = {Dataset.GLASS, Dataset.WDBC, Dataset.FLAME, Dataset.COMPOUND,
                Dataset.PATHBASED, Dataset.JAIN, Dataset.S1, Dataset.S3, Dataset.DIM064, Dataset.DIM256};
        ResultsGenerator resultsGenerator = new ResultsGenerator(datasets);
        resultsGenerator.generate(folderPath, fileName, runs, true, false, true);
    }
}
