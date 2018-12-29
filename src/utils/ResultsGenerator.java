package utils;

import GA.GADriver;
import clustering.Dataset;
import clustering.Experiment;
import clustering.Reporter;
import smile.validation.AdjustedRandIndex;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

/**
 * Created by rusland on 2018-12-29.
 */
public class ResultsGenerator {
    List<Dataset> datasets;
    List<double[][]> dataAttrsList;
    List<int[]> labelsTrueList;
    AdjustedRandIndex adjustedRandIndex = new AdjustedRandIndex();
    Silh silhoutte = new Silh();
    String[] configurations;

    public ResultsGenerator(List<Dataset> aDatasets, String[] aConfigurations) throws IOException {
        datasets = aDatasets;
        this.configurations = aConfigurations;
        processDatasetData();
    }

    private void processDatasetData() throws IOException {
        for (int i = 0; i < datasets.size(); ++i) {
            Dataset dataset = datasets.get(i);
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

    /** @i - dataset index
     *
      */
    protected Experiment measure(int i, int[] labelsPred) {
        HashMap<Integer, double[]> centroids = Utils.centroids(this.dataAttrsList.get(i), labelsPred);
        double aRIScore = this.adjustedRandIndex.measure(this.datasets.get(i).getLabels(), labelsPred);
        double dbScore = Utils.dbIndexScore(centroids, labelsPred, this.dataAttrsList.get(i));
        double silhScore = silhoutte.compute(centroids, labelsPred, this.dataAttrsList.get(i));
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

    // String[] confs = {GADriver.GaConfiguration.mgaC1.name()};//GADriver.GaConfiguration.values();
    public void generate(String folderPath, int runs) throws Exception { // folder "results/mGA/tuning"
        Experiment[] experiments;
        for (String conf: configurations) {
            experiments = new Experiment[runs];
            String filePath = folderPath + conf + ".txt";
            HashMap<String, int[][]> datasetTosolutions = Utils.readSolutionFromFile(filePath, runs, true);
            for (String dataset: datasetTosolutions.keySet()) {
                System.out.println(dataset);
                int[][] expSols = datasetTosolutions.get(dataset);
                int expSolIdx = 0;
                for (int[] expSol: expSols) {
                    System.out.println(Arrays.toString(expSol));
                    int datasetIdx = indexOfDataset(dataset);
                    experiments[expSolIdx] = measure(datasetIdx, expSol);
                }
                String excelFilePath = folderPath + conf + ".xls";
                int datasetIdx = indexOfDataset(dataset);
                ExcelRW.write(excelFilePath, experiments, datasets.get(datasetIdx));
            }
        }
    }

    private int indexOfDataset(String d) {
        for (int i = 0; i < this.datasets.size(); ++i) {
            if (d.equals(this.datasets.get(i).name())) {
                return i;
            }
        }
        return -1;
    }

    public static void main(String[] args) {
        //datasets.add(Dataset.GLASS)
        //ResultsGenerator resultsGenerator = new ResultsGenerator(datasets, GADriver.confValuesStr())
    }
}
