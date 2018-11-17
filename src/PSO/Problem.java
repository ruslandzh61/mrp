package PSO;

import utils.NCConstruct;
import utils.Utils;

/**
 * Created by rusland on 09.09.18.
 */
public class Problem {
    private int N, D;
    // space constraints
    //private double[] dimLow, dimHigh;

    public double[][] getData() {
        return data;
    }

    private double[][] data;
    private Evaluator evaluator;

    public Problem(double[][] aData, Evaluator aEvaluator) {
        this.data = Utils.deepCopy(aData);
        this.N = data.length;
        this.D = data[0].length;
        evaluator = aEvaluator;
        //retrieveDimBound();
    }

    public int getN() {
        return N;
    }

    public int getD() { return D;}

    /*public double getDimLow(int dimIdx) { return dimLow[dimIdx];}

    public double getDimHigh(int dimIdx) { return dimHigh[dimIdx];}*/

    public double[] evaluate(Solution solution, Evaluator.Evaluation[] evaluation, NCConstruct ncc) {
        assert (evaluation.length>0);
        double[] result = new double[evaluation.length];
        for (int iE = 0; iE < evaluation.length; ++iE) {
            result[iE] = evaluator.evaluate(solution, evaluation[iE], data, ncc);
        }
        return result;
    }

    /*private void retrieveDimBound() {
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
    }*/
}
