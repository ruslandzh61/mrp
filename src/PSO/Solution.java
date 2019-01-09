package PSO;

import utils.*;

import java.util.Arrays;

/**
 * Represents clustering solution
 */
public class Solution implements Comparable<Solution> {
    private int[] solution;
    private double fitness;
    private int k;
    private double[] objectives;

    public Solution(Solution s) {
        this.solution = s.getSolution().clone();
        this.k = s.getK(false);
        this.fitness = s.getFitness();
        this.objectives = s.getObjectives().clone();
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
        return solution;
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

    public int getK(boolean dontIncludeEmptyClusters) {
        if (dontIncludeEmptyClusters) {
            return Utils.distinctNumberOfItems(solution);
        } else {
            return k;
        }
    }

    public double getObjective(int i) {
        return objectives[i];
    }

    public double[] getObjectives() {
        return objectives;
    }

    public void setObjectives(double[] aObjectives) {
        objectives = aObjectives.clone();
    }

    private void checkSolution() {
        for (int kNum: this.solution) {
            assert (kNum < this.k);
        }
    }

    @Override
    public int compareTo(Solution other) {
        return Double.compare(this.fitness, other.fitness);
    }

    @Override
    public boolean equals(Object o) {
        Solution other = (Solution) o;
        int[] otherSol = Utils.adjustLabels(other.getSolution());
        int[] thisSol = Utils.adjustLabels(this.getSolution());
        return Arrays.equals(thisSol, otherSol);
    }
}
