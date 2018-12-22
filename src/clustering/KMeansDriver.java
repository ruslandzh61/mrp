package clustering;

import smile.validation.AdjustedRandIndex;
import utils.Utils;
import weka.clusterers.SimpleKMeans;
import weka.core.Instances;
import weka.core.SelectedTag;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

/**
 * Created by rusland on 20.12.18.
 */
public class KMeansDriver extends Analyzer {
    private boolean isUseWekaVersion;

    public KMeansDriver(boolean weka) {
        this.isUseWekaVersion = weka;
    }

    public void run(int runs, Dataset dataset) throws Exception {
        processData(dataset);

        Random rnd = new Random(1);
        this.reporter = new Reporter(runs);

        for (int run = 1; run <= runs; ++run) {
            int[] labelsPred;
            Experiment bestE = new Experiment(null, Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY, -1);
            int minK = 2;//(int)(0.02 * data.length);
            int maxK = (int) Math.sqrt(this.dataAttrs.length); //(int)(0.1 * data.length);
            for (int k = minK; k <= maxK; ++k) {
                if (this.isUseWekaVersion) {
                    SimpleKMeans kMeans = new SimpleKMeans();
                    kMeans.setPreserveInstancesOrder(true);
                    SelectedTag selectedTag = new SelectedTag(SimpleKMeans.KMEANS_PLUS_PLUS, SimpleKMeans.TAGS_SELECTION);
                    kMeans.setInitializationMethod(selectedTag);
                    kMeans.setSeed(rnd.nextInt());
                    kMeans.setNumClusters(k);
                    kMeans.buildClusterer(this.wekaData);
                    labelsPred = kMeans.getAssignments();
                } else {
                    KMeans kMeans = new KMeans(k, 2.0);
                    kMeans.setSeed(rnd.nextInt());
                    kMeans.setInitializationMethod(KMeans.Initialization.KMEANS_PLUS_PLUS);
                    kMeans.buildClusterer(this.dataAttrs);
                    labelsPred = kMeans.getLabels();
                }

                //Utils.removeNoise(labelsPred, data, 2, 2.0);
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

        }
    }

    public static void main(String[] args) throws Exception {
        Dataset dataset = Dataset.GLASS;
        boolean weka = false;
        System.out.println("remove id: " + dataset.isRemoveFirst());
        System.out.println("normalize: " + dataset.isNormalize());
        System.out.println("my k-means++: ");
        KMeansDriver kMeansDriver = new KMeansDriver(weka);
        kMeansDriver.run(40, dataset);
        System.out.println("-------");
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
