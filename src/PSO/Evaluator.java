package PSO;

/**
 * Created by rusland on 06.10.18.
 */
public class Evaluator {
    public enum Evaluation {
        DUMMY,
    }

    public double evaluate(Solution solution, Evaluation evaluation) {
        if (evaluation == Evaluation.DUMMY) {
            return dummy(solution);
        }

        return Double.MIN_VALUE;
    }

    private double dummy(Solution solution) {
        double x = solution.get(0); // the "x" part of the location
        double y = solution.get(1); // the "y" part of the location

        return Math.pow(2.8125 - x + x * Math.pow(y, 4), 2) +
                Math.pow(2.25 - x + x * Math.pow(y, 2), 2) +
                Math.pow(1.5 - x + x * y, 2);
    }
}
