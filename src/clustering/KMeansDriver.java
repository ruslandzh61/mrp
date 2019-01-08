package clustering;

import utils.Utils;
import weka.clusterers.SimpleKMeans;
import weka.core.EuclideanDistance;
import weka.core.ManhattanDistance;
import weka.core.SelectedTag;

import java.util.Arrays;
import java.util.Random;

/**
 * Extends Analyzer to run experiments on K-Means algorithm
 */
public class KMeansDriver extends Analyzer {
    private boolean isUsekMeansPlusPlus;
    private boolean isUseWekaVersion;
    private double distMeasure;

    void setDistMeasure(double distMeasure) {
        this.distMeasure = distMeasure;
    }

    KMeansDriver() {}

    public void run(int k) throws Exception {
        processData();

        Random rnd = new Random(1);
        int[] labelsPred;
        for (int skipRnd = 0; skipRnd < seedStartFrom; ++skipRnd) {
            rnd.nextInt();
        }
        for (int run = 1; run <= this.reporter.size(); ++run) {
            KMeans kMeans = new KMeans(k, distMeasure);
            kMeans.setSeed(rnd.nextInt());
            if (isUsekMeansPlusPlus) {
                kMeans.setInitializationMethod(KMeans.Initialization.KMEANS_PLUS_PLUS);
            } else {
                kMeans.setInitializationMethod(KMeans.Initialization.RANDOM);
            }
            kMeans.buildClusterer(this.dataAttrs);
            labelsPred = kMeans.getLabels();
            Utils.removeNoise(labelsPred, this.dataAttrs, 2, 2.0);
            Utils.adjustAssignments(labelsPred);
            Experiment e = measure(labelsPred);
            reporter.set(run-1, e);

            System.out.println("A:" + e.getAri());
            System.out.println("D:" + e.getDb());
            System.out.println("S:" + e.getSilh());
            System.out.println("K:" + e.getK());
            System.out.println("----");
        }
    }

    /** Multi-run K-Means
     */
    public void run() throws Exception {
        assert (dataset != null);
        assert (reporter != null);
        processData();

        Random rnd = new Random(1);

        for (int run = 1; run <= reporter.size(); ++run) {
            System.out.println("RESULTS FOR RUN: " + run);
            int[] labelsPred;
            Experiment bestE = new Experiment();
            bestE.setAri(Double.NEGATIVE_INFINITY);
            int minK = 2;
            int maxK = (int) Math.sqrt(this.dataAttrs.length);
            for (int k = minK; k <= maxK; ++k) {
                if (this.isUseWekaVersion) {
                    SimpleKMeans kMeans = new SimpleKMeans();
                    kMeans.setPreserveInstancesOrder(true);
                    SelectedTag selectedTag;
                    if (this.isUsekMeansPlusPlus) {
                        selectedTag = new SelectedTag(SimpleKMeans.KMEANS_PLUS_PLUS, SimpleKMeans.TAGS_SELECTION);
                    } else {
                        selectedTag = new SelectedTag(SimpleKMeans.RANDOM, SimpleKMeans.TAGS_SELECTION);
                    }
                    if (this.distMeasure == 1.0) {
                        kMeans.setDistanceFunction(new ManhattanDistance(this.wekaData));
                    } else {
                        kMeans.setDistanceFunction(new EuclideanDistance(this.wekaData));
                    }
                    kMeans.setInitializationMethod(selectedTag);
                    kMeans.setSeed(rnd.nextInt());
                    kMeans.setNumClusters(k);
                    kMeans.buildClusterer(this.wekaData);
                    labelsPred = kMeans.getAssignments();
                } else {
                    KMeans kMeans = new KMeans(k, distMeasure);
                    kMeans.setSeed(rnd.nextInt());
                    if (isUsekMeansPlusPlus) {
                        kMeans.setInitializationMethod(KMeans.Initialization.KMEANS_PLUS_PLUS);
                    } else {
                        kMeans.setInitializationMethod(KMeans.Initialization.RANDOM);
                    }
                    kMeans.buildClusterer(this.dataAttrs);
                    labelsPred = kMeans.getLabels();
                }

                Utils.removeNoise(labelsPred, this.dataAttrs, 2, 2.0);
                Utils.adjustAssignments(labelsPred);

                Experiment e = measure(labelsPred);
                if (e.getAri() > bestE.getAri()) {
                    bestE = e.clone();
                }

                //System.out.println("solution: " + Arrays.toString(labelsPred));
                System.out.println("temp ARI score of kMeans:     " + Utils.doublePrecision(e.getAri(), 4));
                System.out.println("temp DB score of kMeans:      " + Utils.doublePrecision(e.getDb(), 4));
                System.out.println("temp Silh score of kMeans:      " + Utils.doublePrecision(e.getSilh(), 4));
                System.out.println("temp # of clusters of kMeans: " + e.getK());
            }
            reporter.set(run-1, bestE);

            System.out.println("A:" + Utils.doublePrecision(bestE.getAri(), 4));
            System.out.println("D:" + Utils.doublePrecision(bestE.getDb(), 4));
            System.out.println("S:" + Utils.doublePrecision(bestE.getSilh(), 4));
            System.out.println("K:" + bestE.getK());
        }
    }

    /** run experiments for a dataset using different number of clusters
     * @param path - file path to save results,
     * @param runs - number of runs
     * */
    void runK(String path, int runs) throws Exception {
        StringBuilder solutionsLog = new StringBuilder();
        Random rnd = new Random(1);
        int[] labelsPred;
        processData();

        int minK = 2;
        int maxK = (int) Math.sqrt(this.dataAttrs.length);
        int[] fakeLabels = new int[dataAttrs.length];
        double[][] fakeCentroids = new double[1][];
        fakeCentroids[0] = utils.Utils.centroids(dataAttrs, fakeLabels).get(0);
        solutionsLog.append(1 + " " + utils.Utils.sse(fakeCentroids, fakeLabels, dataAttrs) + System.lineSeparator());
        for (int k = minK; k <= maxK; ++k) {
            double totalSSE = 0.0;
            for (int skipRnd = 0; skipRnd < seedStartFrom; ++skipRnd) {
                rnd.nextInt();
            }
            for (int run = 1; run <= runs; ++run) {
                KMeans kMeans = new KMeans(k, distMeasure);
                kMeans.setSeed(rnd.nextInt());
                if (isUsekMeansPlusPlus) {
                    kMeans.setInitializationMethod(KMeans.Initialization.KMEANS_PLUS_PLUS);
                } else {
                    kMeans.setInitializationMethod(KMeans.Initialization.RANDOM);
                }
                kMeans.buildClusterer(this.dataAttrs);
                labelsPred = kMeans.getLabels();
                Utils.removeNoise(labelsPred, this.dataAttrs, 2, 2.0);
                Utils.adjustAssignments(labelsPred);
                double sse = utils.Utils.sse(kMeans.getCentroids(), labelsPred, dataAttrs);
                totalSSE += sse;
            }

            double avgSSE = totalSSE/runs;
            System.out.println(k + " " + avgSSE);
            solutionsLog.append(k + " " + avgSSE + System.lineSeparator());
        }
        Utils.whenWriteStringUsingBufferedWritter_thenCorrect(solutionsLog.toString(), path, true);
    }

    void setUsekMeansPlusPlus(boolean usekMeansPlusPlus) {
        isUsekMeansPlusPlus = usekMeansPlusPlus;
    }

    /**
     * experiments are run to determine optimal number of clusters in the
     * dataset using Elbow Method for K-Means Clustering
     * @param args - args[1] is a name of dataset;
     *  args[2] is a seed to start experiments from; args[3] is a number of runs
     * @throws Exception
     */
    static void testForElbow(String[] args) throws Exception {
        String path;
        boolean usePlusPlus = false;
        double distMeasure = 1.0;
        Dataset dataset = Dataset.valueOf(args[0]);
        int seedStartFrom = Integer.parseInt(args[1]);
        int runs = Integer.parseInt(args[2]);
        path = "results/k-means/" + dataset.name() + "-" + seedStartFrom + "-" + runs + ".csv";
        KMeansDriver driver = new KMeansDriver();
        driver.setDistMeasure(distMeasure);
        driver.setDataset(dataset);
        driver.setSeedStartFrom(seedStartFrom);
        driver.setUsekMeansPlusPlus(usePlusPlus);
        driver.runK(path, runs);
    }

    /** experiments are run on K-Means using a pre-defined number of clusters, obtained from experimental results
     * utilizing Elbow Method for K-Means clustering.
     */
    static void testKMeans() throws Exception {
        Dataset[] datasets = {Dataset.GLASS, Dataset.JAIN, Dataset.WDBC, Dataset.FLAME, Dataset.COMPOUND,
                Dataset.PATHBASED, Dataset.S1, Dataset.S3, Dataset.DIM064, Dataset.DIM256};
        // ks are determined based on previous experiments using Elbow Method: testForElbow()
        int[] ks = {3, 3, 2, 4, 3, 3, 12, 13, 21, 12};
        int runs = 1;
        int datasetIdx = 8; //Integer.parseInt(args[0]);
        String solutionsFilePath = "results/k-means/" + datasets[datasetIdx].name() + ks[datasetIdx] + ".txt";
        System.out.println(solutionsFilePath + "will be created");
        KMeansDriver driver = new KMeansDriver();
        driver.setUsekMeansPlusPlus(false);
        driver.setDistMeasure(1.0);
        driver.setDataset(datasets[datasetIdx]);
        driver.setRuns(runs);
        driver.setSeedStartFrom(30);
        driver.run(ks[datasetIdx]);
        driver.analyze(true);
        driver.saveResults(solutionsFilePath);
    }

    /**
     * Method to run experiments on K-Means.
     * If arguments are passed, experiments are run to determine optimal number of clusters in the
     * dataset using Elbow Method for K-Means Clustering. Otherwise, that is if no arguments passed,
     * experiments are run on K-Means using a pre-defined number of clusters, obtained from experimental results
     * utilizing Elbow Method for K-Means clustering.
     * Clustering solutions are saved into txt file. Passed arguments are included in file's name.
     * @param args - array of arguments: args[0] is a name of dataset;
     *  args[1] is a seed to start experiments from; args[2] is a number of runs
     **/
    public static void main(String[] args) throws Exception {
        if (args == null || args.length == 0) {
            testKMeans();
        } else {
            testForElbow(args);
        }
    }
}
