package PSO;
import utils.Utils;

import java.io.IOException;
import java.util.*;

/**
 * Created by rusland on 27.10.18.
 */
public class PSODriver {

    /* most of data points end up in the same cluster */
    public void run(int psoNum) throws IOException {
        int maxK;
        double[][] data;
        int[] labels;
        // step 1 - pre-process data
        if (psoNum == 0) {

            maxK = 35;
            //System.out.println(Utils.pickAFile());

            List<String[]> dataStr = Utils.readDataFromCustomSeperator("data/glass.csv", 0, ',');
            assert (dataStr.size()>0);

            labels = new int[dataStr.size()];
            for (int i = 0; i < dataStr.size(); ++i) {
                int labelIdx = dataStr.get(0).length-1;
                labels[i] = Integer.parseInt(dataStr.get(i)[labelIdx]);
            }
            System.out.println(Arrays.toString(labels));

                // exclude columns from csv file
            int[] excludedColumns = {0,dataStr.get(0).length-1};
            data = new double[dataStr.size()][dataStr.get(0).length-excludedColumns.length];
            for (int i = 0; i < data.length; ++i) {
                int k = 0;
                for (int j = 0; j < data[0].length; ++j) {
                    if (Arrays.binarySearch(excludedColumns,j) < 0) {
                        data[i][k++] = Double.parseDouble(dataStr.get(i)[j]);
                    }
                }
            }
        } else {
            labels = new int[]{0,0,0,0,0,0,1,1,1,2};
            data = new double[][]{{2,2}, {3,3}, {3,1}, {4,2}, {1.6,-0.5}, {3.01, -1.5}, {-4, 2}, {-2, 2}, {-3, 3},{7,7}};
            maxK = 5;
        }

        Evaluator.Evaluation[] evaluation = {Evaluator.Evaluation.CONNECTIVITY, Evaluator.Evaluation.COHESION};
        Evaluator evaluator = new Evaluator();
        double velLow = -1.0;
        double velHigh = 1.0;
        Problem problem = new Problem(data, evaluator, velLow, velHigh);
        PSO pso = new PSO(problem, evaluation, maxK);
        Utils.whenWriteStringUsingBufferedWritter_thenCorrect(Arrays.toString(labels) +
                System.getProperty("line.separator") + Arrays.toString(pso.execute()), "data/output.csv");
    }

    public static void main(String[] args) {
        try {
            new PSODriver().run(0);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
