package PSO;

import utils.*;

/**
 * Created by rusland on 06.10.18.
 */
public class Solution {
    private int[] xSolution; // continuos representation, formed from discrete repr
    private double fitness;
    private int k;

    /** number of clusters stays the same if empty clusters are counted as well
    * */
    public Solution(int[] aS, int aK) {
        xSolution = aS.clone();

        // if exceeds number of points
        if (aK > xSolution.length) {
            this.k = xSolution.length;
        } else {
            this.k = aK;
        }
        // uncomment if empty clusters should be removed
        //computeCLusters();
    }


    public int getxSolutionAt(int idx) {
        return xSolution[idx];
    }

    public int size() {
        return xSolution.length;
    }

    public int[] getxSolution() {
        return xSolution.clone();
    }

    /*
    // used only  if xSolution is in locus based representation, which should transformed to clusters
    public CC toClusters() {
        Graph g = new Graph(xSolution.length);
        for (int i = 0; i < xSolution.length; ++i) {
            g.addEdge(i, xSolution[i]);
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

    public int getK() {
        return k;
    }

    public int getKWithoutEmptyClusters() {
        return Utils.distinctNumberOfItems(xSolution);
    }

    public static void main(String[] args) {
        /*
        //test toCluster()
        int[] realS = {1,2,3,0,5,4,7,8,6,9};
        Solution s = new Solution();
        s.setxSolution(realS);
        CC cc = s.toClusters();
        for (int i = 0; i < cc.N(); ++i) {
            System.out.print(cc.id(i) + " ");
        }*/
    }
}
