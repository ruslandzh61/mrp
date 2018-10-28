package PSO;

/**
 * Created by rusland on 06.10.18.
 */
public class Solution {
    private double[] solution; // continuos repr
    private int numClusters;
    private int[] realSolution; // transformed from connection vector

    public Solution(double[] solution) {
        this.solution = solution;
    }

    public double get(int idx) {
        return solution[idx];
    }

    public void setSolution(double[] solution) {
        this.solution = solution;
    }
}
