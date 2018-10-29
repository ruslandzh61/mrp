package PSO;

import utils.Utils;

/**
 * Created by rusland on 09.09.18.
 */
public class Problem {
    private int D;
    // space constraints
    private double[] dimLow, dimHigh;
    private double velLow, velHigh;
    private double[][] data;
    private Evaluator evaluator;

    public Problem(double[][] aData, Evaluator aEvaluator, double[] aDimLow, double[] aDimHigh, double aVelLow, double aVelHigh) {
        this.data = Utils.deepCopy(aData);
        this.D = aData[0].length;
        assert(aDimHigh.length== D);
        assert(aDimLow.length== D);
        evaluator = aEvaluator;
        dimLow = aDimLow;
        dimHigh = aDimHigh;
        velLow = aVelLow;
        velHigh = aVelHigh;
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
}
