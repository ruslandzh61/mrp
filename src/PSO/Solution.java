package PSO;

import utils.*;

/**
 * Created by rusland on 06.10.18.
 */
public class Solution {
    private double[] xSolution; // continuos representation, formed from discrete repr
    private int numClusters;
    private int[] ySolution; // transformed from connection vector;

    public Solution() {}

    public Solution(double[] xSolution) {
        this.xSolution = xSolution;

    }

    public double getxSolutionAt(int idx) {
        return xSolution[idx];
    }

    public double[] getxSolution(int idx) {
        return xSolution.clone();
    }

    public int[] getySolution() {
        return ySolution.clone();
    }
    public void setySolution(int[] realS) {
        ySolution = realS.clone();
    }

    public CC toClusters() {
        Graph g = new Graph(ySolution.length);
        for (int i = 0; i < ySolution.length; ++i) {
            g.addEdge(i, ySolution[i]);
        }
        CC connectedComps = new CC(g);
        numClusters = connectedComps.count();

        return connectedComps;
    }

    public void setxSolution(double[] xSolution) {
        this.xSolution = xSolution;
    }

    public static void main(String[] args) {
        int[] realS = {1,2,3,0,5,4,7,8,6,9};
        Solution s = new Solution();
        s.setySolution(realS);
        CC cc = s.toClusters();
        for (int i = 0; i < cc.N(); ++i) {
            System.out.print(cc.id(i) + " ");
        }
    }
}
