package PSO;

import utils.NCConstruct;
import utils.Pareto;
import utils.Utils;

import java.util.*;

/**
 * Created by rusland on 09.09.18.
 * implements PSO algorithm
 */
public class PSO {
    static double C1 = 1.42;
    static double C2 = 1.63;
    static double MAX_W = 0.9;
    static double MIN_W = 0.4;
    static int MAX_ITERATION = 1000;
    static int NUM_ITER_IMRPOV = 100;
    boolean pickLeaderRandomly = false;
    int numOfObj;
    int maxK;
    int swarmSize;
    double w;
    int totalNumOfIterations, numOfIterWithoutImprov;

    private List<Particle> swarm;
    /* record of objectives of personal best solution are stored
        for the purpose of not calculating them every time pBest is updated */
    private double[][] pBestObjective;
    private Problem problem;
    /* solutions will be evaluated on objectives stored in evaluation array */
    private Evaluator.Evaluation[] evaluation;
    private Solution gBestSolution;
    private Set<Integer> paretoFrontier;
    private Solution paretoLeaderSolution;
    /* NCConstruct is used for evaluating connectivy objective function */
    private NCConstruct ncc;
    private Pareto pareto = new Pareto();
    /* idealObjectives represent utopia point */
    private double[] idealObjectives;
    private Random generator = new Random();
    private VelocityCalculator velocityCalculator = new VelocityCalculator(C1, C2);

    public PSO(Problem aProblem, Evaluator.Evaluation[] aEvaluation, int maxK) {
        problem = aProblem;
        evaluation = aEvaluation;
        numOfObj = evaluation.length;
        this.maxK = maxK;
        this.swarmSize = 2 * maxK;

        swarm = new ArrayList<>();
        pBestObjective = new double[swarmSize][numOfObj];
        for (int iP = 0; iP < swarmSize; ++iP) {
            for (int iO = 0; iO < numOfObj; ++iO) {
                pBestObjective[iP][iO] = Double.NEGATIVE_INFINITY;
            }
        }

        idealObjectives = new double[evaluation.length];
        for (int iO = 0; iO < numOfObj; ++iO) {
            idealObjectives[iO] = Double.NEGATIVE_INFINITY;
        }
        ncc = new NCConstruct(problem.getData());
    }

    public int[] execute() {
        initializeSwarm();

        totalNumOfIterations = 0;
        numOfIterWithoutImprov = 0;

        while(totalNumOfIterations < MAX_ITERATION && numOfIterWithoutImprov < NUM_ITER_IMRPOV) { // && err > ProblemSet.ERR_TOLERANCE) {
            System.out.println("ITERATION " + totalNumOfIterations + ": ");
            update();
            totalNumOfIterations++;
            System.out.println("without improv: " + numOfIterWithoutImprov);
        }

        System.out.println("\nSolution found at iteration " + (totalNumOfIterations - 1) + ", the solutions is:");
        System.out.print("     Best ");
        for (int dimIdx = 0; dimIdx < problem.getN(); ++dimIdx) {
            System.out.print(gBestSolution.getSolutionAt(dimIdx) + " ");
        }
        System.out.println();

        System.out.print("     objectives: ");
        double[] objs = problem.evaluate(gBestSolution, evaluation, ncc);
        for (double obj: objs) {
            System.out.print(obj + " ");
        }
        System.out.println();

        System.out.print("     ideal: ");
        for (int dimIdx = 0; dimIdx < numOfObj; ++dimIdx) {
            System.out.print(idealObjectives[dimIdx] + " ");
        }
        System.out.println();
        return gBestSolution.getSolution();
    }

    public void initializeSwarm() {
        Particle p;
        for(int i = 0; i < swarmSize; i++) {
            // step 1 - randomize particle location using k-means
            int clusterNum = generator.nextInt(maxK-5) + 5;
            // k-means centroids are initialized and point are assigned to a particular centroid
            KMeans kMeans = new KMeans(problem.getData(),problem.getN(),problem.getD(),clusterNum);
            // perform one iteration of k-mean
            //kMeans.oneIter();
            // perform complete k-means clustering
            //kMeans.clustering(100);
            Solution solution = new Solution(kMeans.getLabels(), clusterNum);

            // step 2 -randomize velocity in the defined range
            double[] vel = new double[problem.getN()];
            for (int dimIdx = 0; dimIdx < problem.getN(); ++dimIdx) {
                //vel[dimIdx] = MIN_VEL + generator.nextDouble() * (
                //        MAX_VEL - MIN_VEL);
                vel[dimIdx] = 0.0;
            }
            p = new Particle(solution, vel);
            swarm.add(p);
        }
    }

    private void update() {
        // step 1 - Evaluate objective functions for each particle
        for(int iP = 0; iP < swarmSize; iP++) {
            Particle p = swarm.get(iP);
            p.setObjectives(problem.evaluate(swarm.get(iP).getSolution(), evaluation, ncc));
        }

        // step 2 - store non-dominated particles in paretoFrontier
        Map<Integer, double[]> mapIdxToObjectives = new HashMap<>();
        for (int iP = 0; iP < swarmSize; ++iP) {
            mapIdxToObjectives.put(iP, swarm.get(iP).getObjectives().clone());
        }
        paretoFrontier = pareto.extractParetoNondominated(mapIdxToObjectives);

        /*System.out.println("pareto frontier");
        for (int iPotentialL: paretoFrontier) {
            int[] potLeaderSol = swarm.get(iPotentialL).getSolution().getSolution();
            System.out.println(Arrays.toString(swarm.get(iPotentialL).getObjectives()) +
                    ": " + Arrays.toString(potLeaderSol));
        }*/

        // step 3 - Update Personal Best of each particle
        for(int i = 0; i< swarmSize; i++) {
            double[] curIterObjs = swarm.get(i).getObjectives();
            if(pareto.testDominance(curIterObjs, pBestObjective[i])) {
                //System.out.println(Arrays.toString(curIterObjs) + " > " + Arrays.toString(pBestObjective[i]));
                pBestObjective[i] = swarm.get(i).getObjectives();
                swarm.get(i).setpBest(swarm.get(i).getSolution());
                //System.out.println("pbest updated at iteration: " + totalNumOfIterations);
            } else {
                //System.out.println(Arrays.toString(curIterObjs) + " <= " + Arrays.toString(pBestObjective[i]));
            }
        }

        // step 4 - Select Leader from NonDomRepos
        // randomly or utopia point
        updateGBest();

        // step 5 - Update velocity and particles
        for(int i = 0; i< swarmSize; i++) {
            w = (MAX_W - MIN_W) * (MAX_ITERATION- totalNumOfIterations) / MAX_ITERATION + MIN_W;
            velocityCalculator.setW(w);
            double[] newVel = velocityCalculator.calculate(swarm.get(i));
            swarm.get(i).setVelocity(newVel);
            Utils.checkClusterLabels(paretoLeaderSolution.getSolution(), paretoLeaderSolution.getK(false));
            swarm.get(i).update(paretoLeaderSolution);
        }

        // step - print best in iteration totalNumOfIterations
        System.out.print("     Best: ");
        for (int dimIdx = 0; dimIdx < problem.getN(); ++dimIdx) {
            System.out.print(paretoLeaderSolution.getSolutionAt(dimIdx) + " ");
        }
        System.out.println();

        System.out.print("     Value: ");
        double[] objs = problem.evaluate(paretoLeaderSolution, evaluation, ncc);
        for (double obj: objs) {
            System.out.print(obj + " ");
        }
        System.out.println();
    }

    /** Select global best solution according to certain criteria - proximity to utopia point */
    private void updateGBest() {
        // optional: if a leader is selected randomly
        updateUtopiaPoint();

        // step 1 -  pick a leader in pareto set closest to utopia point
        int iLeader = pickALeader();

        // step 2 - update global best
        // I'M NOT SURE WHETHER GLOBAL BEST SHOULD BE UPDATED ONLY WHEN NEW LEADER DOMINATES IT OR REGARDLESS OF THAT
        paretoLeaderSolution = new Solution(swarm.get(iLeader).getSolution());

        double[] finalBestObjs;
        if (gBestSolution != null) {
            finalBestObjs = problem.evaluate(paretoLeaderSolution, evaluation, ncc);
        } else {
            finalBestObjs = new double[numOfObj];
            for (int i = 0; i < numOfObj; ++i) {
                finalBestObjs[i] = Double.NEGATIVE_INFINITY;
            }
        }
        double[] gBestObjs = problem.evaluate(paretoLeaderSolution,evaluation,ncc);
        if (pareto.testDominance(gBestObjs, finalBestObjs)) {
            gBestSolution = new Solution(paretoLeaderSolution);
            numOfIterWithoutImprov = 0;
        } else {
            numOfIterWithoutImprov++;
        }
    }

    private int pickALeader() {
        int iLeader = -1;
        if (pickLeaderRandomly) {
            iLeader = (int) paretoFrontier.toArray()[generator.nextInt(paretoFrontier.size())];
        } else {
            double minDist = Double.POSITIVE_INFINITY;
            for (int iP : paretoFrontier) {
                double[] cur = swarm.get(iP).getObjectives();
                double distToUtopia = Utils.dist(cur, idealObjectives);
                if (distToUtopia < minDist) {
                    iLeader = iP;
                    minDist = distToUtopia;
                }
            }
        }

        return iLeader;
    }

    private void updateUtopiaPoint() {
        // update utopia point
        for (int i = 0; i < swarmSize; ++i) {
            for (int iO = 0; iO < evaluation.length; ++iO) {
                double obj = swarm.get(i).getObjective(iO);
                if (obj > idealObjectives[iO]) {
                    idealObjectives[iO] = obj;
                }
            }
        }
    }
}
