package PSO;

/**
 * Created by rusland on 27.10.18.
 */
public class PSODriver {
    public static void main(String[] args) {
        Evaluator.Evaluation[] evaluation = {Evaluator.Evaluation.CONNECTIVITY, Evaluator.Evaluation.COHESION};
        double[][] data = {{2,2}, {3,3}, {3,1}, {4,2}, {1.6,-0.5}, {3.01, -1.5}, {-4, 2}, {-2, 2}, {-3, 3},{7,7}};
        Evaluator evaluator = new Evaluator();
        int numDims = 2;
        double velLow = -1;
        double velHigh = 1;
        Problem problem = new Problem(data, evaluator, velLow, velHigh);

        PSO pso = new PSO(problem, evaluation);
        pso.execute();
    }
}
