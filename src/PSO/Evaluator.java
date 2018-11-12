package PSO;

import utils.CC;
import utils.NCConstruct;
import utils.Utils;

import java.util.*;

/**
 * Created by rusland on 06.10.18.
 */
public class Evaluator {
    public enum Evaluation {
        DUMMY, CONNECTIVITY, COHESION
    }

    /**
     * data parameter is needed for the evaluation of a connectivity */
    public Double evaluate(Solution solution, Evaluation evaluation, double[][] data, NCConstruct ncc) {
        if (evaluation == Evaluation.CONNECTIVITY) {
            return connectivity(solution, data, ncc);
        } else if (evaluation == Evaluation.COHESION) {
            return cohesion(solution, data);
        }

        return null;
    }

    private double connectivity(Solution solution, double[][] data, NCConstruct ncc) {
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
        return Utils.sum(con)/clustersHp.size();
    }

    /* returns negative value of cohesion function to enable maximization of this value */
    private double cohesion(Solution solution, double[][] data) {
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
        return Utils.sum(coh)/clustersHp.size() * (-1);
    }

    private double cohesionDistance(double[][] data, int p, Set<Integer> neighbors) {
        double max = -1; // distance can't be negative
        for (Integer neighbor : neighbors) {
            double distTo = Utils.dist(data[p], data[neighbor]);
            max = (distTo > max) ? distTo : max;
        }
        return max;
    }

    private double dummy(Solution solution) {
        double x = solution.getxSolutionAt(0); // the "x" part of the location
        double y = solution.getxSolutionAt(1); // the "y" part of the location

        return Math.pow(2.8125 - x + x * Math.pow(y, 4), 2) +
                Math.pow(2.25 - x + x * Math.pow(y, 2), 2) +
                Math.pow(1.5 - x + x * y, 2);
    }

    private HashMap<Integer, Set<Integer>> toHashMap(Solution solution) {
        // convert solution to efficient data structure
        HashMap<Integer, Set<Integer>> clustersHp = new HashMap<>(); // key is cluster id, value is set of data points
        for (int i = 0; i < solution.size(); ++i) {
            int id = solution.getxSolutionAt(i);
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
        double cost = e.evaluate(s,Evaluation.CONNECTIVITY,data, ncc);
        System.out.println(cost);

    }
}
