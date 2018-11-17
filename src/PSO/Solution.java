package PSO;

import utils.*;

/**
 * Created by rusland on 06.10.18.
 * locus-based solution representation
 * e.g. [1,1,1,1,1,2,2,2,3] - transform nine data points to a particle vector
 *      which represents a clustering solution consisting of three clusters
 *      first 5 points belong to cluster with 1, next 3 points - to cluster 2, last point - to cluster 3
 */
public class Solution {
    private int[] solution;
    private double fitness;
    private int k;

    /** number of clusters stays the same if empty clusters are counted as well
    * */

    public Solution(Solution s) {
        this.solution = s.getSolution();
        this.k = s.getK(false);
        this.fitness = s.getFitness();
        checkSolution();
    }

    public Solution(int[] aS, int aK) {
        assert (aK < aS.length);
        solution = aS.clone();
        this.k = aK;
        checkSolution();
        // uncomment if empty clusters should be removed
        //computeCLusters();
    }

    public int getSolutionAt(int idx) {
        return solution[idx];
    }

    public int size() {
        return solution.length;
    }

    public int[] getSolution() {
        return solution.clone();
    }

    /*
    // used only  if solution is in locus based representation, which should transformed to clusters
    public CC toClusters() {
        Graph g = new Graph(solution.length);
        for (int i = 0; i < solution.length; ++i) {
            g.addEdge(i, solution[i]);
        }
        CC connectedComps = new CC(g);

        return connectedComps;
    }*/

    public void setFitness(double fitness) {
        this.fitness = fitness;
    }

    public double getFitness() {
        return fitness;
    }

    public int getK(boolean includeEmptyClusters) {
        if (includeEmptyClusters) {
            return Utils.distinctNumberOfItems(solution);
        } else {
            return k;
        }
    }

    private void checkSolution() {
        for (int kNum: this.solution) {
            assert (kNum < this.k);
        }
    }
}
