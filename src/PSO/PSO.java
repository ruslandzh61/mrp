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
    private int maxK;
    private int swarmSize = 2 * maxK;
    private static int NUM_OBJ = 2;
    private static int NUM_ITER_IMRPOV = 100;
    private int t, numIterWithoutImprov;

    private List<Particle> swarm;
    /* record of objectives of personal best solution are stored
        for the purpose of not calculating them every time pBest is updated */
    private double[][] pBestObjective;
    private Problem problem;
    private Evaluator.Evaluation[] evaluation;
    private Solution gBestSolution;
    private Set<Integer> paretoFrontier;
    private NCConstruct ncc;
    private Random generator = new Random();
    private Pareto pareto = new Pareto();
    private double[] idealObjectives;

    public PSO(Problem aProblem, Evaluator.Evaluation[] aEvaluation, int maxK) {
        problem = aProblem;
        evaluation = aEvaluation;
        this.maxK = maxK;
        this.swarmSize = 2 * maxK;

        swarm = new ArrayList<>();
        pBestObjective = new double[swarmSize][NUM_OBJ];
        for (int iP = 0; iP < swarmSize; ++iP) {
            for (int iO = 0; iO < NUM_OBJ; ++iO) {
                pBestObjective[iP][iO] = Double.NEGATIVE_INFINITY;
            }
        }

        int[] gBest = new int[problem.getD()];
        for (int i = 0; i < problem.getD(); ++i) {
            gBest[i] = Integer.MIN_VALUE;
        }
        gBestSolution = new Solution(gBest,-1);

        idealObjectives = new double[evaluation.length];
        for (int iO = 0; iO < NUM_OBJ; ++iO) {
            idealObjectives[iO] = Double.NEGATIVE_INFINITY;
        }
        ncc = new NCConstruct(problem.getData());
    }

    public int[] execute() {
        initializeSwarm();

        t = 0;
        numIterWithoutImprov = 0;
        //double err = 9999;

        while(t < MAX_ITERATION && numIterWithoutImprov < NUM_ITER_IMRPOV) { // && err > ProblemSet.ERR_TOLERANCE) {
            System.out.println("ITERATION " + t + ": ");
            update();
            t++;
            System.out.println("without improv: " + numIterWithoutImprov);
        }

        System.out.println("\nSolution found at iteration " + (t - 1) + ", the solutions is:");
        System.out.print("     Best ");
        for (int dimIdx = 0; dimIdx < problem.getN(); ++dimIdx) {
            System.out.print(gBestSolution.getxSolutionAt(dimIdx) + " ");
        }
        System.out.println();

        System.out.print("     objectives: ");
        double[] objs = problem.evaluate(gBestSolution, evaluation, ncc);
        for (double obj: objs) {
            System.out.print(obj + " ");
        }
        System.out.println();

        System.out.print("     ideal: ");
        for (int dimIdx = 0; dimIdx < NUM_OBJ; ++dimIdx) {
            System.out.print(idealObjectives[dimIdx] + " ");
        }
        System.out.println();

        return gBestSolution.getxSolution();
    }

    public void initializeSwarm() {
        Particle p;
        for(int i = 0; i< swarmSize; i++) {
            int clusterNum = generator.nextInt(maxK -1)+1;

            // randomize location inside a space defined in Problem Set

            KMeans kMeans = new KMeans(problem,clusterNum);
            kMeans.nextIter();
            Solution solution = new Solution(kMeans.getLabels(), clusterNum);

            // randomize velocity in the range defined in Problem Set
            double[] vel = new double[problem.getN()];
            for (int dimIdx = 0; dimIdx < problem.getN(); ++dimIdx) {
                vel[dimIdx] = problem.getVelLow() + generator.nextDouble() * (
                        problem.getVelHigh() - problem.getVelLow());
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
            mapIdxToObjectives.put(iP, swarm.get(iP).getObjectives());
        }
        paretoFrontier = pareto.extractParetoNondominated(mapIdxToObjectives);

        // step 3 - Update Personal Bests
        for(int i = 0; i< swarmSize; i++) {
            if(pareto.testDominance(swarm.get(i).getObjectives(), pBestObjective[i])) {
                pBestObjective[i] = swarm.get(i).getObjectives();
                swarm.get(i).setpBest(swarm.get(i).getSolution());
            }
        }

        // step 4 - Select Leader randomly from NonDomRepos
        /* randomly
        int ileader = (int)paretoFrontier.toArray()[generator.nextInt(paretoFrontier.size())];
        Solution gBest = swarm.get(ileader).getSolution();
        */
        updateGBest();

        // step 5 - Update velocity and particles
        for(int i = 0; i< swarmSize; i++) {
            updateVelocity(i);
            swarm.get(i).update(gBestSolution);
        }

        // step - print best in iteration t
        System.out.print("     Best: ");
        for (int dimIdx = 0; dimIdx < problem.getN(); ++dimIdx) {
            System.out.print(gBestSolution.getxSolutionAt(dimIdx) + " ");
        }
        System.out.println();

        System.out.print("     Value: ");
        double[] objs = problem.evaluate(gBestSolution, evaluation, ncc);
        for (double obj: objs) {
            System.out.print(obj + " ");
        }
        System.out.println();
    }

    private void updateVelocity(int idxParticle) {
        Particle p = swarm.get(idxParticle);
        double r1 = generator.nextDouble();
        double r2 = generator.nextDouble();
        double[] newVel = new double[problem.getN()];
        w = (MAX_W - MIN_W) * (MAX_ITERATION-t) / MAX_ITERATION + MIN_W;
        for (int dimIdx = 0; dimIdx < problem.getN(); ++dimIdx) {
            newVel[dimIdx] = (w * p.getVelocity()[dimIdx]) + (r1 * C1) *
                    (-1 - p.getySolution()[dimIdx]) +
                    (r2 * C2) * (1 - p.getySolution()[dimIdx]);
        }
        p.setVelocity(newVel);
    }

    private void updateGBest() {
        // update utopia point
        for (int i = 0; i < swarmSize; ++i) {
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
        double[] gBestObjs;
        if (t != 0) {
            gBestObjs = problem.evaluate(gBestSolution, evaluation, ncc);
        } else {
            gBestObjs = new double[NUM_OBJ];
            for (int i = 0; i < gBestObjs.length; ++i) {
                gBestObjs[i] = Double.NEGATIVE_INFINITY;
            }
        }
        if (pareto.testDominance(swarm.get(iLeader).getObjectives(), gBestObjs)) {
            Solution sTemp = swarm.get(iLeader).getSolution();
            gBestSolution = new Solution(sTemp.getxSolution(), sTemp.getK());
            numIterWithoutImprov = 0;
        } else {
            numIterWithoutImprov++;
        }
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
