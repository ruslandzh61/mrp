package PSO;

import utils.Utils;

/**
 * Created by rusland on 09.09.18.
 */
public class Problem {
    private int N, D;
    // space constraints
    private double[] dimLow, dimHigh;
    private double velLow, velHigh;

    public double[][] getData() {
        return data;
    }

    private double[][] data;
    private Evaluator evaluator;

    public Problem(double[][] aData, Evaluator aEvaluator, double aVelLow, double aVelHigh) {
        this.data = Utils.deepCopy(aData);
        this.N = data.length;
        this.D = data[0].length;
        evaluator = aEvaluator;
        velLow = aVelLow;
        velHigh = aVelHigh;
        retrieveDimBound();
    }

    public int getN() {
        return N;
    }

    public int getD() { return D;}

    public double getDimLow(int dimIdx) { return dimLow[dimIdx];}

    public double getDimHigh(int dimIdx) { return dimHigh[dimIdx];}

    public double evaluate(Solution solution, Evaluator.Evaluation evaluation) {
        return evaluator.evaluate(solution, evaluation, data);
    }

    public double getVelLow() {
        return velLow;
    }

    public double getVelHigh() {
        return velHigh;
    }

    private void retrieveDimBound() {
        dimLow = new double[D];
        dimHigh = new double[D];
        for (int iD = 0; iD < D; ++iD) {
            dimLow[iD] = Integer.MAX_VALUE;
            dimHigh[iD] = Integer.MIN_VALUE;
        }
        for (int iN = 0; iN < N; ++iN) {
            for (int iD = 0; iD < D; ++iD) {
                if (data[iN][iD] > dimHigh[iD]) {
                    dimHigh[iD] = data[iN][iD];
                }
                if (data[iN][iD] < dimLow[iD]) {
                    dimLow[iD] = data[iN][iD];
                }
            }
        }
    }

    public static void main(String[] args) {
        Evaluator.Evaluation evaluation = Evaluator.Evaluation.CONNECTIVITY;
        double[][] data = {{2, 2}, {3, 3}, {3, 1}, {4, 2}, {1.6, -0.5}, {3.01, -1.5}, {-4, 2}, {-2, 2}, {-3, 3}, {7, 7}};
        Evaluator evaluator = new Evaluator();
        int numDims = 2;
        double velLow = -1;
        double velHigh = 1;
        Problem problem = new Problem(data, evaluator, velLow, velHigh);
        for (int iD = 0; iD < numDims; ++iD) {
            System.out.println(problem.getDimLow(iD) + " " + problem.getDimHigh(iD));
        }
    }
}
