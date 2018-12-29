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
 * Created by rusland on 2018-12-29.
 */
public class ResultsGenerator {
    Dataset[] datasets;
    List<double[][]> dataAttrsList;
    String[] configurations;

    AdjustedRandIndex adjustedRandIndex = new AdjustedRandIndex();
    Silh silhoutte = new Silh();

    public ResultsGenerator(Dataset[] aDatasets, String[] aConfigurations) throws IOException {
        datasets = aDatasets;
        this.dataAttrsList = new ArrayList<>(datasets.length);
        this.configurations = aConfigurations;
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

    /** @i - dataset index
     *
      */
    protected Experiment measure(int i, int[] labelsPred) {
        HashMap<Integer, double[]> centroids = Utils.centroids(this.dataAttrsList.get(i), labelsPred);
        double aRIScore = this.adjustedRandIndex.measure(this.datasets[i].getLabels(), labelsPred);
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
        Experiment[][] confMeans = new Experiment[datasets.length][configurations.length];
        Experiment[][] confStdDevs = new Experiment[datasets.length][configurations.length];
        int confIdx = 0;
        String confMeansPath = folderPath + "mgaTuningMeans" + ".xls";
        String confStdDevPath = folderPath + "mgaTuningStdDev" + ".xls";
        for (String conf: configurations) {
            System.out.println(conf);
            experiments = new Experiment[runs+2];
            String filePath = folderPath + conf + ".txt";
            HashMap<String, int[][]> datasetTosolutions = Utils.readSolutionFromFile(filePath, runs, true);

            int datasetIdx = 0;
            for (Dataset dataset: datasets) {
                System.out.println(dataset);
                int[][] expSols = datasetTosolutions.get(dataset.name());
                Reporter reporter = new Reporter(expSols.length);
                int expSolIdx = 0;
                for (int[] expSol: expSols) {
                    //System.out.println(Arrays.toString(expSol));
                    experiments[expSolIdx] = measure(datasetIdx, expSol);
                    reporter.set(expSolIdx, experiments[expSolIdx]);
                    ++expSolIdx;
                }
                String excelFilePath = folderPath + conf + ".xls";
                reporter.compute();
                experiments[expSolIdx++] = reporter.getMean();
                experiments[expSolIdx] = reporter.getStdDev();
                ExcelRW.write(excelFilePath, experiments, datasets[datasetIdx]);

                confMeans[datasetIdx][confIdx] = reporter.getMean();
                confMeans[datasetIdx][confIdx].setConfiguration(conf);
                confStdDevs[datasetIdx][confIdx] = reporter.getStdDev();
                confStdDevs[datasetIdx][confIdx].setConfiguration(conf);

                ++datasetIdx;
            }
            ++confIdx;
        }

        int datasetIdx = 0;
        for (Dataset dataset: datasets) {
            ExcelRW.write(confMeansPath, confMeans[datasetIdx], dataset);
            ExcelRW.write(confStdDevPath, confStdDevs[datasetIdx], dataset);
            ++datasetIdx;
        }
    }

    public static void main(String[] args) throws Exception {
        Dataset[] datasets = {Dataset.GLASS, Dataset.FLAME, Dataset.DERMATOLOGY, Dataset.COMPOUND, Dataset.WDBC, Dataset.PATHBASED};
        ResultsGenerator resultsGenerator = new ResultsGenerator(datasets, GADriver.confValuesStr());
        resultsGenerator.generate("results/mGA/tuning/", 10);
    }
}
