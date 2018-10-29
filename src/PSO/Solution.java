package PSO;

import utils.*;

/**
 * Created by rusland on 06.10.18.
 */
public class Solution {
    private int[] xSolution; // continuos representation, formed from discrete repr
    private double fitness;

    public Solution(int[] aS) {
        xSolution = aS.clone();
    }

    public Solution() {
    }


    public double getxSolutionAt(int idx) {
        return xSolution[idx];
    }

    public void setxSolution(int[] xSolution) {
        this.xSolution = xSolution.clone();
    }

    public int[] getxSolution() {
        return xSolution.clone();
    }

    public CC toClusters() {
        Graph g = new Graph(xSolution.length);
        for (int i = 0; i < xSolution.length; ++i) {
            g.addEdge(i, xSolution[i]);
        }
        CC connectedComps = new CC(g);

        return connectedComps;
    }

    public void setFitness(double fitness) {
        this.fitness = fitness;
    }

    public double getFitness() {
        return fitness;
    }

    public static void main(String[] args) {
        int[] realS = {1,2,3,0,5,4,7,8,6,9};
        Solution s = new Solution();
        s.setxSolution(realS);
        CC cc = s.toClusters();
        for (int i = 0; i < cc.N(); ++i) {
            System.out.print(cc.id(i) + " ");
        }
    }
}
