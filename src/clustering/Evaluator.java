package clustering;

import PSO.Solution;
import utils.NCConstruct;
import utils.Utils;

import java.util.*;

/**
 * Created by rusland on 06.10.18.
 */
public class Evaluator {
    public enum Evaluation {
        CONNECTIVITY, COHESION
    }


    /**
     * 'data' and 'ncc' parameters are needed for the evaluation of connectivity function
     * minimize fitness
     * */
    public Double evaluate(int[] solution, Evaluation evaluation, double[][] data, NCConstruct ncc) {
        if (evaluation == Evaluation.CONNECTIVITY) {
            return -connectivity(solution, data, ncc);
        } else if (evaluation == Evaluation.COHESION) {
            /* returns 1 / (value of cohesion objective) to enable maximization of this objective */
            return cohesion(solution, data); //1.0 / cohesion(solution, data);
        }

        return null;
    }

    /**
     * value of connectivity objective.
     * value would fall in the interval [0, 1].
     * Should be maximized.
     * */
    private double connectivity(int[] solution, double[][] data, NCConstruct ncc) {
        HashMap<Integer, Set<Integer>> clustersHp = toHashMap(solution);
        /*NC construction and obtaining sub-clusters*/

        int N = data.length;
        double[] con = new double[clustersHp.size()];
        int i = 0;
        for (Map.Entry<Integer, Set<Integer>> entry : clustersHp.entrySet()) {
            // add a data point itself to neighborhood sub-cluster
            // compute con[i]
            Set<Integer> cluster = entry.getValue();
            double conI = 0;
            for (int j = 0; j < N; ++j) {
                // + intersection of sub-cluster from NC (neighbors of i) with a cluster from solution (pair.getValue)
                List<Integer> neighborhoodJ = ncc.neighbors(j);
                double intersectionPerc = Utils.intersection(cluster, neighborhoodJ);
                intersectionPerc += (cluster.contains(j)) ? 1: 0;
                intersectionPerc /= neighborhoodJ.size()+1;
                conI += intersectionPerc;
            }
            con[i++] = conI/N;
        }
        // i is a number of clusters in solution at this point
        return Utils.sum(con, 1.0)/clustersHp.size();
    }

    /**
     *  value of cohesion objective.
     *  value would fall in the interval [0, Infinity].
     *  Should be minimized.
     *  */
    private double cohesion(int[] solution, double[][] data) {
        // key is id of a cluster; value is set of data points in the cluster
        HashMap<Integer, Set<Integer>> clustersHp = toHashMap(solution);
        double[] coh = new double[clustersHp.size()];
        int i = 0;
        int N = data.length;
        for (Map.Entry<Integer, Set<Integer>> entry : clustersHp.entrySet()) {
            double cohI = 0;
            for (Integer pointJ : entry.getValue()) {
                cohI += cohesionDistance(data, pointJ, entry.getValue());
            }
            coh[i++] = cohI/entry.getValue().size();
        }
        return Utils.sum(coh, 1.0)/clustersHp.size();
    }

    private double cohesionDistance(double[][] data, int p, Set<Integer> neighbors) {
        double max = -1; // distance can't be negative
        for (Integer neighbor : neighbors) {
            double distTo = Utils.dist(data[p], data[neighbor], 2);
            max = (distTo > max) ? distTo : max;
        }
        return max;
    }

    private HashMap<Integer, Set<Integer>> toHashMap(int[] solution) {
        // convert solution to efficient data structure
        HashMap<Integer, Set<Integer>> clustersHp = new HashMap<>(); // key is cluster id, value is set of data points
        for (int i = 0; i < solution.length; ++i) {
            int id = solution[i];
            if (clustersHp.containsKey(id)) {
                clustersHp.get(id).add(i);
            } else {
                Set<Integer> cluster = new HashSet<>();
                cluster.add(i);
                clustersHp.put(id,cluster);
            }
        }
        return clustersHp;
    }

    /*
    // only use if solution is locus-based representation, which should be transformed to normal representation
    private HashMap<Integer, Set<Integer>> toHashMap(Solution solution) {
        // convert solution to efficient data structure
        CC connectedComps = solution.toClusters();
        HashMap<Integer, Set<Integer>> clustersHp = new HashMap<>(); // key is cluster id, value is set of data points
        for (int i = 0; i < connectedComps.N(); ++i) {
            int id = connectedComps.id(i);
            if (clustersHp.containsKey(id)) {
                clustersHp.get(id).add(i);
            } else {
                Set<Integer> cluster = new HashSet<>();
                cluster.add(i);
                clustersHp.put(id,cluster);
            }
        }
        return clustersHp;
    }*/

    public static void main(String[] args) {

        int[] realS = {1,2,3,0,5,4,7,8,6,9}; // 4 clusters
        //int[] realS = {1,2,3,0,0,4,7,8,6,9}; // 3 clusters
        //int[] realS = {1,2,3,5,0,4,7,8,0,9}; // 2 cluster
        //int[] realS = {1,2,3,5,0,4,7,8,0,0}; // 1 cluster
        double[][] data = {{2,2}, {3,3}, {3,1}, {4,2}, {1.6,-0.5}, {3.01, -1.5}, {-4, 2}, {-2, 2}, {-3, 3},{7,7}};

        Solution s = new Solution(realS, Utils.distinctNumberOfItems(realS));
        Evaluator e = new Evaluator();
        NCConstruct ncc = new NCConstruct(data);
        double cost = e.evaluate(realS,Evaluation.CONNECTIVITY,data, ncc);
        System.out.println(cost);

    }
}
