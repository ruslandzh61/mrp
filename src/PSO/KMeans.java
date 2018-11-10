package PSO;

import java.util.Random;

/**
 * Created by rusland on 10.11.18.
 */
public class KMeans {
    private int[] label;
    private Problem problem;

    public double[][] getCentroids() {
        return centroids;
    }

    private double[][] centroids;
    private Random rnd = new Random();
    private int k;

    KMeans(Problem aProblem, int aK) {
        label = new int[aProblem.getN()];
        problem = aProblem;
        this.k=aK;
        centroids = new double[k][];
        for (int iR = 0; iR < k; ++iR) {
            centroids[iR] = new double[problem.getD()];
            for (int iD = 0; iD < problem.getD(); ++iD) {
                centroids[iR][iD] = problem.getDimLow(iD) + rnd.nextDouble() * (
                        problem.getDimHigh(iD) - problem.getDimLow(iD));
            }
        }
    }

    public int[] getLabels() {
        return label;
    }

    void nextIter() {
        //assign record to the closest centroid
        for (int i=0; i<problem.getN(); i++){
            label[i] = closest(problem.getData()[i]);
        }
        centroids = updateCentroids();
    }

    private double[][] updateCentroids() {
        // initialize centroids and set to 0
        double [][] newc = new double [k][]; //new centroids
        int [] counts = new int[k]; // sizes of the clusters

        // intialize
        for (int i=0; i<k; i++) {
            counts[i] =0;
            newc[i] = new double [problem.getD()];
            for (int j=0; j<problem.getD(); j++)
                newc[i][j] =0;
        }


        for (int i=0; i<problem.getN(); i++){
            for (int j=0; j<problem.getD(); j++){
                newc[label[i]][j] += problem.getData()[i][j]; // update that centroid by adding the member data record
            }
            counts[label[i]]++;
        }

        // finally get the average
        for (int i=0; i< k; i++){
            for (int j=0; j<problem.getD(); j++){
                newc[i][j] /= counts[i];
            }
        }

        return newc;
    }

    // find the closest centroid for the record v
    private int closest(double[] v){
        double mindist = dist(v, centroids[0]);
        int label = 0;
        for (int i = 1; i < k; i++) {
            double t = dist(v, centroids[i]);
            if (mindist > t) {
                mindist = t;
                label = i;
            }
        }
        return label;
    }

    // compute Euclidean distance between two vectors v1 and v2
    private double dist(double [] v1, double [] v2){
        double sum=0;
        for (int i=0; i<problem.getD(); i++){
            double d = v1[i]-v2[i];
            sum += d*d;
        }
        return Math.sqrt(sum);
    }

    public static void main(String[] args) {
        Evaluator.Evaluation evaluation = Evaluator.Evaluation.CONNECTIVITY;
        double[][] data = {{2, 2}, {3, 3}, {3, 1}, {4, 2}, {1.6, -0.5}, {3.01, -1.5}, {-4, 2}, {-2, 2}, {-3, 3}, {7, 7}};
        Evaluator evaluator = new Evaluator();
        int numDims = 2;
        double velLow = -1;
        double velHigh = 1;
        Problem problem = new Problem(data, evaluator, velLow, velHigh);
        int k = 4;
        KMeans kMeans = new KMeans(problem,k);
        kMeans.nextIter();
        int[] labels = kMeans.getLabels();
        double[][] centroids = kMeans.getCentroids();


        for (int i = 0; i < k; ++i) {
            for (int iD = 0; iD < numDims; ++iD) {
                System.out.print(centroids[i][iD] + " ");
            }
            System.out.println();
        }

        for (int i = 0; i < data.length; ++i) {
            System.out.print(labels[i] + " ");
        }


    }
}
