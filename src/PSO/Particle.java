package PSO;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Created by rusland on 09.09.18.
 */
public class Particle {
    private Solution solution;
    private Solution pBest;
    private double[] velocity;


    private double[] objectives; // objective function values: connectivity on first and cohesion on second

    private int[] ySolution; // dummy intermediate discrete representation

    private final Random rnd = new Random();

    public Particle(Solution aSolution, double[] aVelocity) {
        this.velocity = aVelocity;
        this.solution = aSolution;
        this.ySolution = new int[solution.getxSolution().length];
    }

    /**
     * update X through Y. update Y first and then X for next iteration of PSO
     * gBest is global best solution at step t-1, pBest is personal best at step t-1 before calling update method
     * */
    public void update(Solution gBest) {
        // step 1 - obtain currect vector value Y at step t-1
        computeYFromX(gBest);
        // step 2 - update velocity at step t, code can be found in PSO; called just before update

        // step 3 - new value of lambla is computed using updated value of velocity
        double[] lambda = new double[solution.getxSolution().length];
        int N = solution.getxSolution().length;
        for (int j = 0; j < N; ++j) {
            lambda[j] = ySolution[j] + velocity[j];
        }

        // step 4 - vector value Y at step is computed
        for (int j = 0; j < N; ++j) {
            if (lambda[j] > PSO.ALPHA) {
                ySolution[j] = 1;
            } else if (lambda[j] < PSO.ALPHA * (-1)) {
                ySolution[j] = -1;
            } else {
                ySolution[j] = 0;
            }
        }

        // step 5 - obtain new solution vector X from vector Y at step t
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

    public double getObjective(int i) {
        return objectives[i];
    }

    public double[] getObjectives() {
        return objectives.clone();
    }

    public void setObjectives(double[] aObjectives) {
        objectives = aObjectives.clone();
    }

    public static void main(String[] args) {
        
    }
}

