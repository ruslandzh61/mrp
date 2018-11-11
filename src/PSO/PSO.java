package PSO;

import utils.NCConstruct;
import utils.Pareto;
import utils.Utils;

import java.util.*;

/**
 * Created by rusland on 09.09.18.
 */
public class PSO {
    private int MAX_ITERATION = 1000;
    private double C1 = 1.42;
    private double C2 = 1.63;
    private double MAX_W = 0.9;
    private double MIN_W = 0.4;
    private double w;
    static double ALPHA=0.5;
    private static int MAX_K = 20;
    private static int SWARM_SIZE = 2 * MAX_K;
    private static int NUM_OBJ = 2;
    private int t;

    private List<Particle> swarm = new ArrayList<>();
    private double[][] pBestObjective = new double[SWARM_SIZE][NUM_OBJ];
    private Problem problem;
    private Evaluator.Evaluation[] evaluation;
    private Solution gBestSolution;
    private Set<Integer> paretoFrontier;
    private NCConstruct ncc;
    private Random generator = new Random();
    private Pareto pareto = new Pareto();
    private double[] idealObjectives;

    public PSO(Problem aProblem, Evaluator.Evaluation[] aEvaluation) {
        problem = aProblem;
        evaluation = aEvaluation;
        int[] gBest = new int[problem.getD()];
        for (int i = 0; i < problem.getD(); ++i) {
            gBest[i] = Integer.MIN_VALUE;
        }
        gBestSolution = new Solution(gBest);
        idealObjectives = new double[evaluation.length];
    }

    public void execute() {
        initializeSwarm();

        t = 0;
        //double err = 9999;

        while(t < MAX_ITERATION) { // && err > ProblemSet.ERR_TOLERANCE) {
            System.out.println("ITERATION " + t + ": ");
            update();
            t++;
        }

        System.out.println("\nSolution found at iteration " + (t - 1) + ", the solutions is:");
        for (int dimIdx = 0; dimIdx < problem.getD(); ++dimIdx) {
            System.out.println("     Best " + dimIdx + ": " + gBestSolution.getxSolutionAt(dimIdx));
        }
    }

    public void initializeSwarm() {
        ncc = new NCConstruct(problem.getData());
        Particle p;
        for(int i=0; i<SWARM_SIZE; i++) {
            int clusterNum = generator.nextInt(MAX_K);

            // randomize location inside a space defined in Problem Set

            KMeans kMeans = new KMeans(problem,clusterNum);
            kMeans.nextIter();
            Solution solution = new Solution(kMeans.getLabels());

            // randomize velocity in the range defined in Problem Set
            double[] vel = new double[problem.getD()];
            for (int dimIdx = 0; dimIdx < problem.getD(); ++dimIdx) {
                vel[dimIdx] = problem.getVelLow() + generator.nextDouble() * (
                        problem.getVelHigh() - problem.getVelLow());
            }
            p = new Particle(solution, vel);
            swarm.add(p);
        }
    }

    private void update() {
        // step 1 - Evaluate objective functions for each particle
        for(int iP = 0; iP < SWARM_SIZE; iP++) {
            swarm.get(iP).setObjectives(problem.evaluate(swarm.get(iP).getSolution(), evaluation));
        }

        // step 2 - store non-dominated particles in paretoFrontier
        Map<Integer, double[]> mapIdxToObjectives = new HashMap<>();
        for (int iP = 0; iP < SWARM_SIZE; ++iP) {
            mapIdxToObjectives.put(iP, swarm.get(iP).getObjectives());
        }
        paretoFrontier = pareto.extractParetoNondominated(mapIdxToObjectives);

        // step 3 - Update Personal Bests
        for(int i=0; i<SWARM_SIZE; i++) {
            if(pareto.testDominance(swarm.get(i).getObjectives(), pBestObjective[i])) {
                pBestObjective[i] = swarm.get(i).getObjectives();
            }
        }

        // step 4 - Select Leader randomly from NonDomRepos
        /* randomly
        int ileader = (int)paretoFrontier.toArray()[generator.nextInt(paretoFrontier.size())];
        Solution gBest = swarm.get(ileader).getSolution();
        */
        updateGBest();

        // step 5 - Update velocity and particles
        for(int i=0; i<SWARM_SIZE; i++) {
            updateVelocity(i);
            swarm.get(i).update(gBestSolution);
        }
        for (int dimIdx = 0; dimIdx < problem.getD(); ++dimIdx) {
            System.out.println("     Best " + dimIdx + ": " + gBestSolution.getxSolutionAt(dimIdx));
        }
        System.out.print("     Value: ");
        for (double obj: problem.evaluate(gBestSolution, evaluation)) {
            System.out.print(obj + " ");
        }
        System.out.println();
    }

    private void updateVelocity(int idxParticle) {
        Particle p = swarm.get(idxParticle);
        double r1 = generator.nextDouble();
        double r2 = generator.nextDouble();
        double[] newVel = new double[problem.getD()];
        w = (MAX_W - MIN_W) * (MAX_ITERATION-t) / MAX_ITERATION + MIN_W;
        for (int dimIdx = 0; dimIdx < problem.getD(); ++dimIdx) {
            newVel[dimIdx] = (w * p.getVelocity()[dimIdx]) + (r1 * C1) *
                    (-1 - p.getySolution()[dimIdx]) +
                    (r2 * C2) * (1 - p.getySolution()[dimIdx]);
        }
        p.setVelocity(newVel);
    }

    private void updateGBest() {
        // update utopia point
        for (int i = 0; i < SWARM_SIZE; ++i) {
            for (int iO = 0; iO < evaluation.length; ++iO) {
                double obj = swarm.get(i).getObjectives()[iO];
                if (obj > idealObjectives[iO]) {
                    idealObjectives[iO] = obj;
                }
            }
        }

        // find closest to utopia point
        double minDist = Integer.MAX_VALUE;
        int iLeader = -1;
        for (int iP: paretoFrontier) {
            double distToUtopia = Utils.dist(swarm.get(iP).getObjectives(),idealObjectives);
            if (distToUtopia < minDist) {
                iLeader = iP;
                minDist = distToUtopia;
            }
        }
        gBestSolution = swarm.get(iLeader).getSolution();
    }


    private static int getMinPos(double[] list) {
        int pos = 0;
        double minValue = list[0];

        for(int i=0; i<list.length; i++) {
            if(list[i] < minValue) {
                pos = i;
                minValue = list[i];
            }
        }

        return pos;
    }

    public static void main(String[] args) {

    }
}
