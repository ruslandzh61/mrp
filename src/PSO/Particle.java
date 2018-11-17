package PSO;

import utils.Utils;

import java.util.Random;

/**
 * Created by rusland on 09.09.18.
 * represents extended version of particle, defined in (Jarboui, 2007):
 *      'Combinatorial particle swarm optimization (CPSO) for partitional clustering problem'
 * Extended particle stores real solution 'solution' and intermediate dummy solution vector
 * This version is able to deal with a combinatorial representation of PSO
 */
public class Particle {
    public static double ALPHA = 0.5;
    /* solution stores real reprsentation */
    private Solution solution;
    private Solution pBest;
    private double[] velocity;
    private double[] objectives; // objective function values: connectivity on first and cohesion on second

    private int[] dummySol; // dummy intermediate discrete representation
    private final Random rnd = new Random();

    public Particle(Solution aSolution, double[] aVelocity) {
        this.velocity = aVelocity;
        this.solution = aSolution;
        this.dummySol = new int[solution.getSolution().length];
    }

    /**
     * update real solution vector X through intermediate solution vector Y. update Y first and then X for step t
     * gBest is global best solution at step t-1, pBest is personal best at step t-1 before calling update method
     * */
    public void update(Solution gBest) {
        assert (gBest != null);

        int[] sol = solution.getSolution();
        // step 1 - obtain currect dummy intermediate solution vector at step t-1
        calcIntermediateSol(gBest);
        //System.out.println("yk-1: " + Arrays.toString(dummySol));

        // step 2 - update velocity at step t, code can be found in PSO; called just before update
        // step 3 - new value of lambla is computed using updated value of velocity
        double[] lambda = new double[sol.length];
        int N = solution.getSolution().length;
        for (int j = 0; j < N; ++j) {
            lambda[j] = dummySol[j] + velocity[j];
        }

        //System.out.println("lam: " + Arrays.toString(lambda));

        // step 4 - vector value Y at step is computed
        for (int j = 0; j < N; ++j) {
            if (lambda[j] > ALPHA) {
                dummySol[j] = 1;
            } else if (lambda[j] < -ALPHA) {
                dummySol[j] = -1;
            } else {
                dummySol[j] = 0;
            }
        }
        //System.out.println("yk: " + Arrays.toString(dummySol));
        // step 5 - obtain new real solution vector from intermediate solution vector

        int[] newSol = new int[N];
        for (int j = 0; j < N; ++j) {
            if (dummySol[j] == 1) {
                newSol[j] = gBest.getSolutionAt(j);
            } else if (dummySol[j] == -1) {
                newSol[j] = pBest.getSolutionAt(j);
            } else {
                // randomly select cluster
                //use only if solution is in locus-based representation
                //newSol[j] = rnd.nextInt(solution.toClusters().count());
                int randomCluster = rnd.nextInt(solution.getK(false));
                newSol[j] = randomCluster;
            }
        }

        // get highest cluster id
        int maxK = 0;
        for (int i = 0; i < N; ++i) {
            if (maxK < newSol[i]) {
                maxK = newSol[i];
            }
        }
        solution = new Solution(newSol, maxK+1);
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

    public void setSolution(int[] sol) {
        this.solution = new Solution(sol.clone(), solution.getK(false));
    }

    public int getDummySolAt(int dimIdx) {
        return dummySol[dimIdx];
    }

    // pBest is not used in client PSO code, since gBest is calculated using records of particle's objectives
    public Solution getpBest() {
        return pBest;
    }

    public void setpBest(Solution pBest) {
        this.pBest = new Solution(pBest.getSolution(),pBest.getK(false));
    }

    public double getObjective(int i) {
        return objectives[i];
    }

    public double[] getObjectives() {
        return objectives;
    }

    public void setObjectives(double[] aObjectives) {
        objectives = aObjectives.clone();
    }

    private void calcIntermediateSol(Solution gBest) {
        int N = solution.getSolution().length;
        for (int j = 0; j < N; ++j) {
            if (solution.getSolutionAt(j) == pBest.getSolutionAt(j) &&
                    pBest.getSolutionAt(j) == gBest.getSolutionAt(j)){
                dummySol[j] = (rnd.nextDouble() >= 0.5) ? 1: -1;
            } else if (solution.getSolutionAt(j) == pBest.getSolutionAt(j)) {
                dummySol[j] = 1;
            } else if (solution.getSolutionAt(j) == gBest.getSolutionAt(j)) {
                dummySol[j] = -1;
            } else {
                dummySol[j] = 0;
            }
        }
    }


    public static void main(String[] args) {
        int[] s = {0,0,0,1,2,1};
        int[] pbest = {1,0,2,1,2,0};
        int[] gbest = {2,1,2,0,1,1};
        double[] vel = {-1.2,-0.9,0.5,0.7,0.9,0.3};
        Solution sol = new Solution(s,3);
        Solution pbestSol = new Solution(pbest,3);
        Solution gbestSol = new Solution(gbest,3);

        Particle p = new Particle(sol,vel);
        // pretend velocity is updated
        p.setVelocity(vel);
        p.setpBest(pbestSol);
        p.update(gbestSol);
    }

}

