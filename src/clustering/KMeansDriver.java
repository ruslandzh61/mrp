package clustering;

import utils.Utils;
import weka.clusterers.SimpleKMeans;
import weka.core.SelectedTag;
import java.util.Random;

/**
 * Created by rusland on 20.12.18.
 */
public class KMeansDriver extends Analyzer {
    private boolean isUsekMeansPlusPlus;
    private boolean isUseWekaVersion;

    public KMeansDriver(boolean plus, boolean weka) {
        this.isUsekMeansPlusPlus = plus;
        this.isUseWekaVersion = weka;
    }

    public void run(int runs, Dataset dataset) throws Exception {
        processData(dataset);

        Random rnd = new Random(1);
        this.reporter = new Reporter(runs);

        for (int run = 1; run <= runs; ++run) {
            System.out.println("RESULTS FOR RUN: " + run);
            int[] labelsPred;
            Experiment bestE = new Experiment();
            bestE.setAri(Double.NEGATIVE_INFINITY);
            int minK = 2;//(int)(0.02 * data.length);
            int maxK = (int) Math.sqrt(this.dataAttrs.length); //(int)(0.1 * data.length);
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
                    kMeans.setInitializationMethod(selectedTag);
                    kMeans.setSeed(rnd.nextInt());
                    kMeans.setNumClusters(k);
                    kMeans.buildClusterer(this.wekaData);
                    labelsPred = kMeans.getAssignments();
                } else {
                    KMeans kMeans = new KMeans(k, 2.0);
                    kMeans.setSeed(rnd.nextInt());
                    if (isUsekMeansPlusPlus) {
                        kMeans.setInitializationMethod(KMeans.Initialization.KMEANS_PLUS_PLUS);
                    } else {
                        kMeans.setInitializationMethod(KMeans.Initialization.RANDOM);
                    }
                    kMeans.buildClusterer(this.dataAttrs);
                    labelsPred = kMeans.getLabels();
                }

                //Utils.removeNoise(labelsPred, this.dataAttrs, 2, 2.0);
                //Utils.adjustAssignments(labelsPred);

                Experiment e = measure(labelsPred);
                if (e.getAri() > bestE.getAri()) {
                    bestE = e.clone();
                }

                /*System.out.println("solution: " + Arrays.toString(labelsPred));
                System.out.println("temp ARI score of kMeans:     " + Utils.doublePrecision(tmpARI, 4));
                System.out.println("temp DB score of kMeans:      " + Utils.doublePrecision(tmpDB, 4));
                System.out.println("temp Silh score of kMeans:      " + Utils.doublePrecision(silhScore, 4));
                System.out.println("temp # of clusters of kMeans: " + k + " : " + Utils.distinctNumberOfItems(labelsPred) + " : " + kMeans.numberOfClusters());
                System.out.println(Utils.distinctNumberOfItems(labelsPred) == kMeans.numberOfClusters());*/
            }
            reporter.set(run-1, bestE);
        }
    }

    public boolean isUsekMeansPlusPlus() {
        return isUsekMeansPlusPlus;
    }

    public void setUsekMeansPlusPlus(boolean usekMeansPlusPlus) {
        isUsekMeansPlusPlus = usekMeansPlusPlus;
    }

    public static void main(String[] args) throws Exception {
        Dataset yeast = Dataset.PATHBASED;
        Dataset[] datasets = {yeast};
        int runs = 10;
        boolean usePlusPlus = true;
        boolean useWeka = false;

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
            kMeansDriver.run(runs, dataset);
            kMeansDriver.analyze();
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
}
