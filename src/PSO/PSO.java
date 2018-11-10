package PSO;

import utils.NCConstruct;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Created by rusland on 09.09.18.
 */
public class PSO {
    private int MAX_ITERATION = 1000;
    private int PROBLEM_DIMENSION;
    private double C1 = 2.0;
    private double C2 = 2.0;
    private double W_UPPERBOUND = 0.9;
    private double W_LOWERBOUND = 0.4;
    private double w;
    private double alhpa=0.5;
    private static int MAX_K = 20;
    private static int SWARM_SIZE = 2 * MAX_K;

    private List<Particle> swarm = new ArrayList<>();
    private double[] pBestFitness = new double[SWARM_SIZE];
    private List<Solution> pBestSolution = new ArrayList<>();
    private double gBestFitness;
    private Solution gBestSolution;
    private double[] fitnessValueList = new double[SWARM_SIZE];
    private Problem problem;
    private Evaluator.Evaluation evaluation;
    private List<Particle> paretoFrontier = new ArrayList<>();
    private NCConstruct ncc;
    Random generator = new Random();

    public PSO(Problem aProblem, Evaluator.Evaluation aEvaluation) {
        problem = aProblem;
        evaluation = aEvaluation;
        PROBLEM_DIMENSION = problem.getD();
    }

    public void execute() {
        initializeSwarm();
        updateFitnessList();

        for(int i=0; i<SWARM_SIZE; i++) {
            pBestFitness[i] = fitnessValueList[i];
            pBestSolution.add(swarm.get(i).getSolution());
        }

        int t = 0;
        //double err = 9999;

        while(t < MAX_ITERATION) { // && err > ProblemSet.ERR_TOLERANCE) {
            System.out.println("ITERATION " + t + ": ");
            update(t++);
        }

        System.out.println("\nSolution found at iteration " + (t - 1) + ", the solutions is:");
        for (int dimIdx = 0; dimIdx < PROBLEM_DIMENSION; ++dimIdx) {
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
            double[] vel = new double[PROBLEM_DIMENSION];
            for (int dimIdx = 0; dimIdx < PROBLEM_DIMENSION; ++dimIdx) {
                vel[dimIdx] = problem.getVelLow() + generator.nextDouble() * (
                        problem.getVelHigh() - problem.getVelLow());
            }
            p = new Particle(solution, vel);
            swarm.add(p);
        }
    }

    private void update(int iterNum) {
        /*
        // step 1 - update pBestFitness
        for(int i=0; i<SWARM_SIZE; i++) {
            if(fitnessValueList[i] < pBestFitness[i]) {
                pBestFitness[i] = fitnessValueList[i];
                pBestSolution.set(i, swarm.get(i).getSolution());
            }
        }

        // step 2 - update gBestFitness
        int bestParticleIndex = PSO.getMinPos(fitnessValueList);
        if(iterNum == 0 || fitnessValueList[bestParticleIndex] < gBestFitness) {
            gBestFitness = fitnessValueList[bestParticleIndex];
            gBestSolution = swarm.get(bestParticleIndex).getSolution();
        }

        w = W_UPPERBOUND - (((double) iterNum) / MAX_ITERATION) * (W_UPPERBOUND - W_LOWERBOUND);

        for(int i=0; i<SWARM_SIZE; i++) {
            double r1 = generator.nextDouble();
            double r2 = generator.nextDouble();

            Particle p = swarm.get(i);

            // step 3 - update velocity
            double[] newVel = new double[PROBLEM_DIMENSION];
            for (int dimIdx = 0; dimIdx < PROBLEM_DIMENSION; ++dimIdx) {
                newVel[dimIdx] = (w * p.getVelocity()[dimIdx]) +
                        (r1 * C1) * (pBestSolution.get(i).getxSolutionAt(dimIdx) - p.getSolution().getxSolutionAt(dimIdx)) +
                        (r2 * C2) * (gBestSolution.getxSolutionAt(dimIdx) - p.getSolution().getxSolutionAt(dimIdx));
            }

            p.setVelocity(newVel);

            // step 4 - update location
            double[] newSol = new double[PROBLEM_DIMENSION];
            for (int dimIdx = 0; dimIdx < PROBLEM_DIMENSION; ++dimIdx) {
                newSol[dimIdx] = p.getSolution().getxSolutionAt(dimIdx) + newVel[dimIdx];
            }

            Solution solution = new Solution(newSol);
            p.setSolution(solution);
        }

        //err = ProblemSet.evaluate(gBestSolution) - 0; // minimizing the functions means it's getting closer to 0

        for (int dimIdx = 0; dimIdx < PROBLEM_DIMENSION; ++dimIdx) {
            System.out.println("     Best " + dimIdx + ": " + gBestSolution.getxSolutionAt(dimIdx));
        }
        System.out.println("     Value: " + problem.evaluate(gBestSolution, evaluation));

        updateFitnessList();*/
    }

    private void updateVelocity(int idxParticle) {
        Particle p = swarm.get(idxParticle);
        double r1 = generator.nextDouble();
        double r2 = generator.nextDouble();
        double[] newVel = new double[PROBLEM_DIMENSION];
        for (int dimIdx = 0; dimIdx < PROBLEM_DIMENSION; ++dimIdx) {
            newVel[dimIdx] = (w * p.getVelocity()[dimIdx]) + (r1 * C1) *
                    (-1 - p.getySolution()[dimIdx]) +
                    (r2 * C2) * (1 - p.getySolution()[dimIdx]);
        }
        p.setVelocity(newVel);
    }


    public void updateFitnessList() {
        for(int i=0; i<SWARM_SIZE; i++) {
            swarm.get(i).getSolution().setFitness(problem.evaluate(swarm.get(i).getSolution(), evaluation));
            fitnessValueList[i] = swarm.get(i).getSolution().getFitness();
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
}
