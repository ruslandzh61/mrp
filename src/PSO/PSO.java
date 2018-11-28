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
    PSOConfiguration conf;
    private int swarmSize;
    private int curIterationNum, numOfIterWithoutImprov;

    private List<Particle> swarm;
    /* record of objectives of personal best solution are stored
        for the purpose of not calculating them every time pBest is updated */
    private double[][] pBestObjective;
    private Problem problem;
    /* solutions will be evaluated on objectives stored in evaluation array */
    private Evaluator.Evaluation[] evaluation;
    //private Solution gBestSolution;
    private Set<Solution> paretoOptimal;
    /* NCConstruct is used for evaluating connectivy objective function */
    private NCConstruct ncc;
    private Pareto pareto = new Pareto();
    /* idealObjectives represent utopia point */
    private double[] idealObjectives;
    private Random generator = new Random();
    private VelocityCalculator velocityCalculator;

    public PSO(Problem aProblem, NCConstruct aNCconstruct, Evaluator.Evaluation[] aEvaluation, PSOConfiguration configuration) {
        this.problem = aProblem;
        this.ncc = aNCconstruct;
        this.evaluation = aEvaluation;
        this.conf = configuration;

        this.velocityCalculator = new VelocityCalculator(conf.c1, conf.c2);
        this.swarmSize = 2 * conf.maxK;
        this.swarm = new ArrayList<>();

        int numOfObj = evaluation.length;
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
    }

    public int[] execute() {
        initializeSwarm();

        curIterationNum = 0;
        numOfIterWithoutImprov = 0;

        while(curIterationNum < conf.maxIteration && numOfIterWithoutImprov < conf.maxIterWithoutImprovement) { // && err > ProblemSet.ERR_TOLERANCE) {
            System.out.println("ITERATION " + curIterationNum + ": ");
            update();
            curIterationNum++;
            System.out.println("size of pareto-optimal set: " + paretoOptimal.size());
            System.out.println("without improv: " + numOfIterWithoutImprov);
        }
        Solution leader = pickALeader();
        System.out.println("\nSolution found at iteration " + (curIterationNum - 1) + ", the solutions is:");
        System.out.print("     Best ");
        for (int dimIdx = 0; dimIdx < problem.getN(); ++dimIdx) {
            System.out.print(leader.getSolutionAt(dimIdx) + " ");
        }
        System.out.println();

        System.out.print("     objectives: ");
        double[] objs = problem.evaluate(leader, evaluation, ncc);
        for (double obj: objs) {
            System.out.print(obj + " ");
        }
        System.out.println();

        System.out.print("     ideal found objectives: ");
        for (int dimIdx = 0; dimIdx < evaluation.length; ++dimIdx) {
            System.out.print(idealObjectives[dimIdx] + " ");
        }
        System.out.println();
        System.out.println("pareto-optimal set:");
        for (Solution s: paretoOptimal) {
            System.out.println(Arrays.toString(s.getObjectives()));
        }
        return leader.getSolution();
    }

    public void initializeSwarm() {
        Particle p;
        for(int i = 0; i < swarmSize; i++) {
            // step 1 - randomize particle location using k-means
            int clusterNum = generator.nextInt(conf.maxK-1) + 2;
            // k-means centroids are initialized and point are assigned to a particular centroid
            KMeans kMeans = new KMeans(problem.getData(),problem.getN(),problem.getD(),clusterNum);
            // perform one iteration of k-mean
            //kMeans.oneIter();
            // perform complete k-means clustering
            kMeans.clustering(50);
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
            p.getSolution().setObjectives(problem.evaluate(swarm.get(iP).getSolution(), evaluation, ncc));
        }

        // step 2 - include non-dominated particles in pareto-optimal set
        // map particle position in swarm to particle objectives
        Map<Integer, double[]> mapIdxToObjectives = new HashMap<>();

        for (int iP = 0; iP < swarmSize; ++iP) {
            mapIdxToObjectives.put(iP, swarm.get(iP).getSolution().getObjectives().clone());
        }

        if (paretoOptimal == null) {
            // if paretoOptimal has not been initialized yet
            paretoOptimal = new HashSet<>();
            Set<Integer> idxPareto = pareto.extractParetoNondominated(mapIdxToObjectives);
            for (int idx: idxPareto) {
                // number of solutions in pareto-optimal set cannot be more than P_MAX
                if (paretoOptimal.size() >= conf.pMax) {
                    System.out.println("exceeded maximum allowed size of pareto-optimal set");
                    break;
                }
                // for optimization purposes copy of solution is not created and original one is inserted instead,
                // so that same solution object won't be included again in pareto-optimal set
                paretoOptimal.add(swarm.get(idx).getSolution());
            }
            numOfIterWithoutImprov++;
        } else {
            // otherwise
            // consider each particle to be included into pareto-optimal set
            int prevSize = paretoOptimal.size();
            for (Particle p: swarm) {
                if (paretoOptimal.size() > conf.pMax) {
                    System.out.println("exceeded maximum allowed size of pareto-optimal set");
                    break;
                }
                if (!dominatedByParetoSet(p.getSolution(), paretoOptimal)) {
                    paretoOptimal.add(p.getSolution());
                }
            }

            if (paretoOptimal.size() != prevSize) {
                // changed
                numOfIterWithoutImprov = 0;
            } else {
                //didn't change
                numOfIterWithoutImprov++;
            }
        }

        /*System.out.println("pareto frontier");
        for (int iPotentialL: paretoFrontier) {
            int[] potLeaderSol = swarm.get(iPotentialL).getSolution().getSolution();
            System.out.println(Arrays.toString(swarm.get(iPotentialL).getObjectives()) +
                    ": " + Arrays.toString(potLeaderSol));
        }*/

        // step 3 - Update Personal Best of each particle
        for(int i = 0; i< swarmSize; i++) {
            double[] curIterObjs = swarm.get(i).getSolution().getObjectives();
            if(pareto.testDominance(curIterObjs, pBestObjective[i], false)) {
                //System.out.println(Arrays.toString(curIterObjs) + " > " + Arrays.toString(pBestObjective[i]));
                pBestObjective[i] = swarm.get(i).getSolution().getObjectives();
                swarm.get(i).setpBest(swarm.get(i).getSolution());
                //System.out.println("pbest updated at iteration: " + curIterationNum);
            } else {
                //System.out.println(Arrays.toString(curIterObjs) + " <= " + Arrays.toString(pBestObjective[i]));
            }
        }

        // step 4 - Select Leader from NonDomRepos
        // randomly or utopia point
        Solution paretoLeaderSolution = getLeader();

        // step 5 - Update velocity and particles
        for(int i = 0; i< swarmSize; i++) {
            double w = (conf.maxW - conf.minW) * (conf.maxIteration - curIterationNum) / conf.maxIteration + conf.minW;
            velocityCalculator.setW(w);
            double[] newVel = velocityCalculator.calculate(swarm.get(i));
            swarm.get(i).setVelocity(newVel);
            Utils.checkClusterLabels(paretoLeaderSolution.getSolution(), paretoLeaderSolution.getK(false));
            // update and check for boundaries: k cannot be more than kMax
            swarm.get(i).update(paretoLeaderSolution, conf.maxK);
        }

        // step - print best in iteration curIterationNum
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
    private Solution getLeader() {
        // optional: if a leader is selected randomly
        updateUtopiaPoint();

        // step 1 -  pick a leader in pareto set closest to utopia point
        return pickALeader();
    }

    private Solution pickALeader() {
        Solution leader = null;
        if (conf.pickLeaderRandomly) {
            int iLeader = generator.nextInt(paretoOptimal.size());
            int i = 0;
            for (Solution s: paretoOptimal) {
                if (i == iLeader) {
                    leader = s;
                    break;
                }
                i++;
            }
        } else {
            double minDist = Double.POSITIVE_INFINITY;
            for (Solution s: paretoOptimal) {
                double[] cur = s.getObjectives();
                double distToUtopia = Utils.dist(cur, idealObjectives);
                if (distToUtopia < minDist) {
                    leader = s;
                    minDist = distToUtopia;
                }
            }
        }

        assert (leader != null);

        return leader;
    }

    private void updateUtopiaPoint() {
        // update utopia point
        for (int i = 0; i < swarmSize; ++i) {
            for (int iO = 0; iO < evaluation.length; ++iO) {
                double obj = swarm.get(i).getSolution().getObjective(iO);
                if (obj > idealObjectives[iO]) {
                    idealObjectives[iO] = obj;
                }
            }
        }
    }

    private boolean dominatedByParetoSet(Solution solution, Set<Solution> paretoOptimal) {
        List<Solution> dominatedBySolution = new ArrayList<>();
        for (Solution s: paretoOptimal) {
            if (pareto.testDominance(s.getObjectives(), solution.getObjectives(), true)) {
                return true;
            } else if (pareto.testDominance(solution.getObjectives(), s.getObjectives(), false)) {
                dominatedBySolution.add(s);
            }
        }

        // if not dominated remove all solutions in pareto-optimal set that are dominated by 'solution'
        paretoOptimal.removeAll(dominatedBySolution);
        return false;
    }
}
