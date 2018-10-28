package PSO;

import java.util.List;

/**
 * Created by rusland on 09.09.18.
 */
public class Problem {
    private int numDimensions;
    // space constraints
    private double[] dimLow, dimHigh;
    private double velLow, velHigh;

    private Evaluator evaluator;

    public Problem(int aNumDimensions, Evaluator aEvaluator, double[] aDimLow, double[] aDimHigh, double aVelLow, double aVelHigh) {
        this.numDimensions = aNumDimensions;
        assert(aDimHigh.length==numDimensions);
        assert(aDimLow.length==numDimensions);
        evaluator = aEvaluator;
        dimLow = aDimLow;
        dimHigh = aDimHigh;
        velLow = aVelLow;
        velHigh = aVelHigh;
    }

    public int getNumDimensions() { return numDimensions;}

    public double getDimLow(int dimIdx) { return dimLow[dimIdx];}

    public double getDimHigh(int dimIdx) { return dimHigh[dimIdx];}

    public double evaluate(Solution solution, Evaluator.Evaluation evaluation) {
        return evaluator.evaluate(solution, evaluation);
    }

    public double getVelLow() {
        return velLow;
    }

    public double getVelHigh() {
        return velHigh;
    }
}
