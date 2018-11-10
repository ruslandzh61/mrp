package PSO;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Created by rusland on 09.09.18.
 */
public class Particle {
    public static double ALPHA = 0.5;
    private Solution solution;
    private Solution pBest;
    private double[] velocity;

    private int[] ySolution; // dummy intermediate discrete representation

    private final Random rnd = new Random();

    public Particle(Solution aSolution) {
        this.solution = aSolution;
    }

    public Particle(Solution aSolution, double[] aVelocity) {
        this.velocity = aVelocity;
        this.solution = aSolution;
    }

    /**
     * update X through. update Y first and then X for next iteration of PSO
     * */
    public void update(Solution gBest) {
        double[] lambda = new double[solution.getxSolution().length];
        int N = solution.getxSolution().length;
        for (int j = 0; j < N; ++j) {
            lambda[j] = ySolution[j] + velocity[j];
        }
        for (int j = 0; j < N; ++j) {
            if (lambda[j] > Particle.ALPHA) {
                ySolution[j] = 1;
            } else if (lambda[j] < -Particle.ALPHA) {
                ySolution[j] = -1;
            } else {
                ySolution[j] = 0;
            }
        }

        // new solution
        int[] newSol = new int[N];
        for (int j = 0; j < N; ++j) {
            if (ySolution[j] == 1) {
                newSol[j] = gBest.getxSolution()[j];
            } else if (ySolution[j] == -1) {
                newSol[j] = pBest.getxSolution()[j];
            } else {
                // randomly select cluster
                newSol[j] = rnd.nextInt(solution.toClusters().count());
            }
        }
        solution.setxSolution(newSol);
    }

    public void computeYFromX(Solution gBest) {
        int N = solution.getxSolution().length;
        for (int j = 0; j < N; ++j) {
            if (solution.getxSolution()[j] == pBest.getxSolution()[j] &&
                    pBest.getxSolution()[j] == gBest.getxSolution()[j]){
                ySolution[j] = (rnd.nextDouble() >= 0.5) ? 1: -1;
            } else if (solution.getxSolution()[j] == pBest.getxSolution()[j]) {
                ySolution[j] = 1;
            } else if (solution.getxSolution()[j] == gBest.getxSolution()[j]) {
                ySolution[j] = -1;
            } else {
                ySolution[j] = 0;
            }
        }
    }

    public double[] getVelocity() {
        return velocity;
    }

    public void setVelocity(double[] velocity) {
        this.velocity = velocity;
    }

    public Solution getSolution() {
        return solution;
    }

    public Solution getpBest() {
        return pBest;
    }

    public void setpBest(Solution pBest) {
        this.pBest = pBest;
    }

    public int[] getySolution() {
        return this.ySolution.clone();
    }

    public void setySolution(int[] aySolution) {
        this.ySolution = aySolution;
    }

    public static void main(String[] args) {
        
    }
}

