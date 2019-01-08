package PSO;

import utils.Utils;

import java.util.Random;
import java.util.Set;

/**
 * represents extended version of particle, defined in (Jarboui, 2007):
 *      'Combinatorial particle swarm optimization (CPSO) for partitional buildClusterer problem'
 * Extended particle stores real solution 'solution' and intermediate dummy solution vector
 * This version is able to deal with a combinatorial representation of PSO
 */
public class Particle implements Comparable<Particle> {
    public static double ALPHA = 0.5;
    /* solution stores real reprsentation */
    private Solution solution;
    private Solution pBest;
    private double[] velocity;

    private int[] dummySol; // dummy intermediate discrete representation
    private Random rnd;
    private int seed;
    private static int default_seed = 10;
    private Solution prev = null;

    public Particle(Solution aSolution, double[] aVelocity) {
        assert (aVelocity != null);
        assert (aSolution != null);
        assert (aSolution.getSolution().length==aVelocity.length);

        this.velocity = aVelocity;
        this.solution = aSolution;
        this.dummySol = new int[solution.getSolution().length];
        this.setpBest(this.solution);
        setSeed(default_seed);
    }

    public Particle(Particle particle) {
        this.solution = new Solution(particle.getSolution());
        if (particle.getpBest() != null) {
            this.setpBest(particle.getpBest());
        }
        if (particle.prev != null) {
            this.setpBest(particle.prev);
        }
        this.velocity = particle.getVelocity().clone();
        this.dummySol = particle.getDummySol().clone();
        setSeed(particle.getSeed());
    }

    public Solution getPrev() {
        return prev;
    }

    /**
     * update real solution vector X through intermediate solution vector Y. update Y first and then X for step t
     * gBest is global best solution at step t-1, pBest is personal best at step t-1 before calling update method
     * */
    public void update(Solution gBest, int maxK) {
        assert (gBest != null);
        this.prev = new Solution(this.solution);

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
        int distNumK = Utils.distinctNumberOfItems(solution.getSolution());
        Integer[] distClusters = Utils.distinctItems(solution.getSolution()).toArray(new Integer[distNumK]);
        for (int j = 0; j < N; ++j) {
            if (dummySol[j] == 1) {
                newSol[j] = gBest.getSolutionAt(j);
            } else if (dummySol[j] == -1) {
                newSol[j] = pBest.getSolutionAt(j);
            } else {
                // randomly select cluster
                //use only if solution is in locus-based representation
                //newSol[j] = rnd.nextInt(solution.toClusters().count());
                // randomly assign to existing cluster
                int rndIdx = rnd.nextInt(distClusters.length);
                int randomCluster = distClusters[rndIdx];
                newSol[j] = randomCluster;
            }
        }

        // check for maxK bound constraint
        for (int i = 0; i < N; ++i) {
            // if violates maxK bound constraint - set to random cluster
            if (newSol[i] >= maxK) {
                newSol[i] = rnd.nextInt(maxK);
            }
        }

        // get number of clusters, since number of clusters might increase after updating the particle using global best
        int maxLabel = -1;
        for (int i = 0; i < N; ++i) {
            if (maxLabel < newSol[i]) {
                maxLabel = newSol[i];
            }
        }

        solution = new Solution(newSol, maxLabel+1);
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

    public int[] getDummySol() {
        return dummySol;
    }

    // pBest is not used in client PSO code, since gBest is calculated using records of particle's objectives
    public Solution getpBest() {
        return pBest;
    }

    public void setpBest(Solution aPBest) {
        this.pBest = new Solution(aPBest.getSolution().clone(),aPBest.getK(false));
        this.pBest.setFitness(aPBest.getFitness());
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

    @Override
    public int compareTo(Particle other) {
        return this.solution.compareTo(other.getSolution());
    }

    public int getSeed() {
        return seed;
    }

    public void setSeed(int seed) {
        rnd = new Random(seed);
        this.seed = seed;
    }
}

