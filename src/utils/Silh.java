package utils;

import clustering.Cluster;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;


/**
 * Calculates Silhouette Coefficient
 */
public class Silh {
    List<Cluster> clusters;

    public double compute(HashMap<Integer, double[]> aClusters, int[] aLabels, double[][] aData) {
        transform(aClusters,aLabels,aData);
        return calculateMeanCoefficient();
    }

    public double compute(int[] aLabels, double[][] aData) {
        HashMap<Integer, double[]> aClusters = utils.Utils.centroids(aData, aLabels);
        return compute(aClusters, aLabels, aData);
    }
    private void transform(HashMap<Integer, double[]> aClusters, int[] aLabels, double[][] aData) {
        clusters = new ArrayList<>(aClusters.size());
        HashMap<Integer, Integer> mapIDToArr = new HashMap<>();
        int idx = 0;
        for (int id: aClusters.keySet()) {
            clusters.add(new Cluster(id, aClusters.get(id)));
            mapIDToArr.put(id, idx);
            ++idx;
        }

        for (int i = 0; i < aData.length; ++i) {
            int label = aLabels[i];
            int indexToPut = mapIDToArr.get(label);
            clusters.get(indexToPut).add(aData[i]);
        }
        clusters.removeIf(cluster -> cluster.size() < 1);
        /*for (Cluster cluster: clusters) {
            System.out.println(cluster.id() + ": ");
            for (double[] point: cluster.getPoints()) {
                System.out.println(Arrays.toString(point));
            }
        }*/
    }

    private double calculateMeanCoefficient() {
        double sum = 0;
        int N = 0;
        for (Cluster cluster : clusters) {
            int clusterID = cluster.id();
            for (double[] point : cluster.getPoints()) {
                sum += calculateCoefficientForPoint(point, clusterID);
                N++;
            }
        }

        return sum / N;
    }

    private double calculateCoefficientForPoint(double[] onePoint, int clusterLabel) {
        double a = 0;
        double b = Double.MAX_VALUE;
        for (Cluster cluster : clusters) {
            double sum = 0;
            int clusterID = cluster.id();
            int N = cluster.size();
            for (double[] otherPoint : cluster.getPoints()) {
                double dist = Utils.dist(onePoint, otherPoint, 2.0);
                sum += dist;
            }
            double avgDistance = sum / N;
            if(clusterID == clusterLabel) {
                if (N < 2) {
                    a = 0.0D;
                } else {
                    double correction = N / (N - 1.0);
                    a = correction * avgDistance;
                }
            } else {
                b = Math.min(avgDistance, b);
            }
        }
        return (b-a) / Math.max(a, b);
    }

    public static void main(String[] args) {
        double[][] data = {{1, 1, 0}, {10, 10, 10}, {5, 5, 5}, {7, 4, 9}, {0, 1, 1}, {4, 4 ,6}};
        int[] labels = {2, 1, 5, 5, 5, 2};
        HashMap<Integer, double[]> centroids = utils.Utils.centroids(data, labels);
        centroids.put(3, new double[]{2,2,2});
        Silh silh = new Silh();
        System.out.println(silh.compute(centroids, labels, data));
    }
}
