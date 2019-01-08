package PSO;

import clustering.Evaluator;
import utils.NCConstruct;
import utils.Utils;

/**
 * Represents clustering problem
 */
public class Problem {
    private int N, D;

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
    }

    public int getN() {
        return N;
    }

    public int getD() { return D;}

    public double[] evaluate(int[] solution, Evaluator.Evaluation[] evaluation, NCConstruct ncc) {
        assert (evaluation.length>0);
        double[] result = new double[evaluation.length];
        for (int iE = 0; iE < evaluation.length; ++iE) {
            result[iE] = evaluator.evaluate(solution, evaluation[iE], data, ncc);
        }
        return result;
    }
}
