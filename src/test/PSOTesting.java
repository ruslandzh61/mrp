package test;

import PSO.PSO;
import PSO.PSOConfiguration;
import PSO.Problem;
import clustering.Evaluator;
import smile.validation.AdjustedRandIndex;
import utils.NCConstruct;

/**
 * Created by rusland on 20.12.18.
 */
public class PSOTesting {

    /** simple test using artificial two-dimensional data sets
     *  similar to the one used in (Inkaya, 2014) depicted on Fig. 3:
     *      'Adaptive neighbourhood construction algorithm based on density and connectivity'
     * */
    /*private void runDummy() {
        double[][] data;
        int[] labels;
        labels = new int[]{0,0,0,0,0,0,1,1,1,2};
        data = new double[][]{
                {2,2}, {3,3}, {3,1}, {4,2}, {1.6,-0.5},
                {3.01, -1.5}, {-4, 2}, {-2, 2}, {-3, 3},{7,7}
        };
        NCConstruct ncConstruct = new NCConstruct(data);
        int maxK = 6;

        Evaluator.Evaluation[] evaluation = {Evaluator.Evaluation.CONNECTIVITY, Evaluator.Evaluation.COHESION};
        Evaluator evaluator = new Evaluator();
        Problem problem = new Problem(data, evaluator);

        PSOConfiguration configuration = new PSOConfiguration();
        configuration.maxK = maxK;
        PSO pso = new PSO(problem, ncConstruct, evaluation, configuration);

        int[] labelsPred = pso.execute();

        System.out.println(new AdjustedRandIndex().measure(labels, labelsPred));
    }*/
}
