package clustering;

import utils.Utils;
import weka.clusterers.SimpleKMeans;
import weka.core.EuclideanDistance;
import weka.core.ManhattanDistance;
import weka.core.SelectedTag;

import java.util.Arrays;
import java.util.Random;

/**
 * Created by rusland on 20.12.18.
 */
public class KMeansDriver extends Analyzer {
    private boolean isUsekMeansPlusPlus;
    private boolean isUseWekaVersion;

    public void setSeedStartFrom(int seedStartFrom) {
        this.seedStartFrom = seedStartFrom;
    }

    private int seedStartFrom;

    public void setDistMeasure(double distMeasure) {
        this.distMeasure = distMeasure;
    }

    private double distMeasure;

    public KMeansDriver(boolean plus, boolean weka) {
        this.isUsekMeansPlusPlus = plus;
        this.isUseWekaVersion = weka;
    }

    public KMeansDriver() {}

    public void runK(String path, int runs) throws Exception {
        StringBuilder solutionsLog = new StringBuilder();
        Random rnd = new Random(1);
        int[] labelsPred;
        processData();

        int minK = 2;
        int maxK = (int) Math.sqrt(this.dataAttrs.length);
        double[][] results = new double[maxK-minK+1][2];
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
            int minK = 2; //(int)(0.02 * dataAttrs.length);
            int maxK = (int) Math.sqrt(this.dataAttrs.length); //(int)(0.1 * dataAttrs.length);
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

    public boolean isUsekMeansPlusPlus() {
        return isUsekMeansPlusPlus;
    }

    public void setUsekMeansPlusPlus(boolean usekMeansPlusPlus) {
        isUsekMeansPlusPlus = usekMeansPlusPlus;
    }

    static void testMultiKmeans() throws Exception {
        int counter = 1; // write counter before writing results to .txt;
        String solutionsFilePath = "results/newDatasets.txt";

        Dataset[] datasets = {Dataset.IS}; //Dataset.AGGREGATION, Dataset.R15, Dataset.JAIN};//Dataset.values();
        int runs = 10;
        boolean usePlusPlus = false;
        boolean useWeka = false;
        double distMeasure = 1.0;

        if (useWeka) {
            if (usePlusPlus) {
                System.out.println("WEKA K-Means");
            } else {
                System.out.println("WEKA K-Means++");
            }
        } else {
            if (usePlusPlus) {
                System.out.println("My K-Means");
            } else {
                System.out.println("My K-Means++");
            }
        }

        for (Dataset dataset: datasets) {
            System.out.println("remove first attribute: " + dataset.isRemoveFirst());
            System.out.println("normalize: " + dataset.isNormalize());
            System.out.println("N=" + dataset.getN() + "; D=" + dataset.getD() + "; K=" + dataset.getK());
            KMeansDriver kMeansDriver = new KMeansDriver(usePlusPlus, useWeka);
            kMeansDriver.setDistMeasure(distMeasure);
            kMeansDriver.setDataset(dataset);
            kMeansDriver.setRuns(runs);
            kMeansDriver.run();
            kMeansDriver.analyze(true);
            //kMeansDriver.saveResults(solutionsFilePath);
            System.out.println("real number of clusters: " + dataset.getK());
        /*System.out.println("my k-means: ");
        PSODriver.runMyKmeans(clustering.KMeans.Initialization.RANDOM, 10, filePath, removeFirst, normalize);
        System.out.println("-------");
        System.out.println("WEKA random k-means");
        PSODriver.runKmeans(SimpleKMeans.RANDOM, 10, filePath, filePathForWeka, removeFirst, normalize);
        System.out.println("-------");
        System.out.println("WEKA kmeans++ k-means");
        PSODriver.runKmeans(SimpleKMeans.KMEANS_PLUS_PLUS, 10, filePath, filePathForWeka, removeFirst, normalize);*/
        }
    }

    public static void main(String[] args) throws Exception {
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
}
