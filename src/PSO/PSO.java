package PSO;

import smile.validation.AdjustedRandIndex;
import utils.NCConstruct;
import utils.Pareto;
import utils.Utils;
import weka.clusterers.SimpleKMeans;
import weka.core.Instances;
import weka.core.SelectedTag;

import java.util.*;

/**
 * Created by rusland on 09.09.18.
 * implements PSO algorithm
 */
public class PSO {
    PSOConfiguration conf;
    private int swarmSize;
    private int curIterationNum, numOfIterWithoutImprov;

    private List<Particle> psoList;
    /* record of objectives of personal best solution are stored
        for the purpose of not calculating them every time pBest is updated */
    private double[][] pBestObjective;
    private Problem problem;
    /* solutions will be evaluated on objectives stored in evaluation array */
    private Evaluator.Evaluation[] evaluation;
    //private Solution gBestSolution;
    private List<Particle> nonDomPSOList;
    /* NCConstruct is used for evaluating connectivy objective function */
    private NCConstruct ncc;
    //private Pareto pareto = new Pareto();
    /* objBestCoordinates represent utopia point */
    private double[] objBestCoordinates, objWorstCoordinates;
    private Random generator;
    private VelocityCalculator velocityCalculator;
    private int prevParetoSize = 0;
    private static int seed_default = 10;
    private Instances instances;
    private int[] labelsTrue;
    AdjustedRandIndex adjustedRandIndex = new AdjustedRandIndex();

    public PSO(Problem aProblem, NCConstruct aNCconstruct, Evaluator.Evaluation[] aEvaluation,
               PSOConfiguration configuration, Instances aInstances, int[] aLabelsTrue) {
        this.problem = aProblem;
        this.ncc = aNCconstruct;
        this.evaluation = aEvaluation;
        this.conf = configuration;
        this.instances = aInstances;
        this.labelsTrue = aLabelsTrue;

        setSeed(seed_default);

        this.velocityCalculator = new VelocityCalculator(conf.c1, conf.c2);
        velocityCalculator.setSeed(generator.nextInt());
        this.swarmSize = 2 * conf.maxK;
        this.psoList = new ArrayList<>();

        int numOfObj = evaluation.length;
        pBestObjective = new double[swarmSize][numOfObj];
        for (int iP = 0; iP < swarmSize; ++iP) {
            for (int iO = 0; iO < numOfObj; ++iO) {
                pBestObjective[iP][iO] = Double.POSITIVE_INFINITY;
            }
        }

        objBestCoordinates = new double[evaluation.length];
        for (int iO = 0; iO < numOfObj; ++iO) {
            objBestCoordinates[iO] = Double.POSITIVE_INFINITY;
        }
        objWorstCoordinates = new double[evaluation.length];
        for (int iO = 0; iO < numOfObj; ++iO) {
            objWorstCoordinates[iO] = Double.NEGATIVE_INFINITY;
        }
    }

    public int[] execute() throws Exception {
        initializeSwarm();

        curIterationNum = 0;
        numOfIterWithoutImprov = 0;

        while(curIterationNum < conf.maxIteration && numOfIterWithoutImprov < conf.maxIterWithoutImprovement) { // && err > ProblemSet.ERR_TOLERANCE) {
            //System.out.println("ITERATION " + curIterationNum + ": ");
            update();
            curIterationNum++;
            //System.out.println("size of pareto-optimal set: " + nonDomPSOList.size());
            //System.out.println("without improv: " + numOfIterWithoutImprov);
        }
        // update nonDomPSOList
        nonDomPSOList = determineParetoSet(psoList);
        updateUtopiaPoint();
        Particle leader = pickALeader(false);
        /*System.out.println("\nSolution found at iteration " + (curIterationNum - 1) + ", the solutions is:");
        System.out.print("     Best ");
        for (int dimIdx = 0; dimIdx < problem.getN(); ++dimIdx) {
            System.out.print(leader.getSolution().getSolutionAt(dimIdx) + " ");
        }
        System.out.println();

        System.out.print("     objectives: ");
        double[] objs = problem.evaluate(leader.getSolution(), evaluation, ncc);
        for (double obj: objs) {
            System.out.print(obj + " ");
        }
        System.out.println();

        System.out.print("     ideal found objectives: ");
        for (int dimIdx = 0; dimIdx < evaluation.length; ++dimIdx) {
            System.out.print(objBestCoordinates[dimIdx] + " ");
        }
        System.out.println();
        System.out.println("pareto-optimal set:");
        for (Particle s: nonDomPSOList) {
            System.out.println(Arrays.toString(s.getSolution().getObjectives()));
        }*/
        return leader.getSolution().getSolution();
    }

    public void initializeSwarm() throws Exception {
        Particle p;
        for(int i = 0; i < swarmSize; i++) {
            // step 1 - randomize particle location using k-means
            int clusterNum = generator.nextInt(conf.maxK - 1) + 2;
            // k-means centroids are initialized and point are assigned to a particular centroid

            Solution solution = new Solution(kMeansAssignments(clusterNum), clusterNum);

            // step 2 -randomize velocity in the defined range
            double[] vel = new double[problem.getN()];
            for (int dimIdx = 0; dimIdx < problem.getN(); ++dimIdx) {
                //vel[dimIdx] = MIN_VEL + generator.nextDouble() * (
                //        MAX_VEL - MIN_VEL);
                vel[dimIdx] = 0.0;
            }
            p = new Particle(solution, vel);
            p.setSeed(generator.nextInt());
            psoList.add(p);
        }
    }

    private int[] kMeansAssignments(int clusterNum) {
        KMeans kMeans = new KMeans(problem.getData(), problem.getN(), problem.getD(), clusterNum, generator.nextInt());
        kMeans.setSeed(generator.nextInt());
        // perform one iteration of k-mean
        //kMeans.oneIter();
        // perform complete k-means clustering
        kMeans.clustering(100);
        return kMeans.getLabels();
    }
    private int[] kMeansAssignments(Instances instances, int k) throws Exception {
        SimpleKMeans kMeans = new SimpleKMeans();
        kMeans.setSeed(generator.nextInt());
        SelectedTag selectedTag = new SelectedTag(SimpleKMeans.RANDOM, SimpleKMeans.TAGS_SELECTION);
        kMeans.setInitializationMethod(selectedTag);
        kMeans.setPreserveInstancesOrder(true);
        //kMeans.setMaxIterations(50);
        kMeans.setNumClusters(k);
        kMeans.buildClusterer(instances);
        return kMeans.getAssignments();
    }

    private void update() {
        // step 1 - Evaluate objective functions for each particle
        for(int iP = 0; iP < swarmSize; iP++) {
            Particle p = psoList.get(iP);
            p.getSolution().setObjectives(problem.evaluate(psoList.get(iP).getSolution(), evaluation, ncc));
        }

        // step 2 - determine pareto-optimal set
        nonDomPSOList = determineParetoSet(psoList);
        /* in case there are no strongly non-dominated particles */
        if (nonDomPSOList.size() == 0) {
            for (int i = 0; i < psoList.size(); ++i) {
                if (psoList.get(i).getSolution().getFitness() == 0) {
                    nonDomPSOList.add(new Particle(psoList.get(i)));
                    break;
                }
            }
        }
        assert (nonDomPSOList != null && nonDomPSOList.size() > 0);
        Collections.sort(nonDomPSOList);


        // step 3 - Update Personal Best of each particle
        for(int i = 0; i < swarmSize; i++) {
            double[] curIterObjs = psoList.get(i).getSolution().getObjectives().clone();
            if(Pareto.testDominance(curIterObjs, pBestObjective[i], false)) {
                pBestObjective[i] = curIterObjs;
                psoList.get(i).setpBest(psoList.get(i).getSolution());
            }
        }

        // step 4 - Select Leader from NonDomRepos
        // randomly or utopia point
        // according maxiMinPSO algorithm global best value should be chosen for each dimension separately.
        // However, cluster numbers from different solution vectors do not relate to each other.
        // Therefore, as it's mentioned in MCPSO a single leader is chosen randomly from pareto set

        // step 5 - Update velocity and particles
        // clone psoList to nextPopList
        List<Particle> nextPopList = new ArrayList<Particle>(swarmSize * 2);
        for (int i = 0; i < psoList.size(); ++i) {
            nextPopList.add(new Particle(psoList.get(i)));
        }

        // update cloned particles
        double w = (conf.maxW - conf.minW) * (conf.maxIteration - curIterationNum) / conf.maxIteration + conf.minW;
        velocityCalculator.setW(w);
        updateUtopiaPoint();
        for(int i = 0; i< nextPopList.size(); i++) {
            Particle paretoLeaderSolution = pickALeader(true);
            double[] newVel = velocityCalculator.calculate(nextPopList.get(i));
            nextPopList.get(i).setVelocity(newVel);
            Utils.checkClusterLabels(paretoLeaderSolution.getSolution().getSolution(),
                    paretoLeaderSolution.getSolution().getK(false));
            // update and check for boundaries: k cannot be more than kMax
            nextPopList.get(i).update(paretoLeaderSolution.getSolution(), conf.maxK);
        }

        // evaluate before determining pareto set
        // step 1 - Evaluate objective functions for each particle
        for(int iP = 0; iP < nextPopList.size(); iP++) {
            Particle p = nextPopList.get(iP);
            p.getSolution().setObjectives(problem.evaluate(nextPopList.get(iP).getSolution(), evaluation, ncc));
        }

        // copy main psoList particles to nextPopList
        for (int i = 0; i < psoList.size(); ++i) {
            nextPopList.add(new Particle(psoList.get(i)));
        }

        List<Particle> nonDomFromNextPSOList = determineParetoSet(nextPopList);

        // copy non-dominated solutions to next generation
        psoList = new ArrayList<Particle>();
        for (int i = 0; i < nonDomFromNextPSOList.size(); ++i) {
            if (i >= conf.pMax) {
                break;
            }
            psoList.add(new Particle(nonDomFromNextPSOList.get(i)));
        }

        int roomToFill = swarmSize - psoList.size();
        // first - remove non-dominated
        nextPopList.removeIf(particle -> (particle.getSolution().getFitness() < 0));
        // now randomly pick nextPopList
        int i = 0;
        while (i < roomToFill) {
            int idxPick  = generator.nextInt(nextPopList.size());
            if (nextPopList.get(idxPick) != null) {
                psoList.add(nextPopList.get(idxPick));
                i++;
                nextPopList.remove(idxPick);
                // nextPopList.set(idxPick, null);
            }
        }

        // naive method to identify whether pareto set changed
        if (this.nonDomPSOList.size() == prevParetoSize) {
            numOfIterWithoutImprov++;
        } else {
            numOfIterWithoutImprov = 0;
        }
        prevParetoSize = this.nonDomPSOList.size();

        // step - print best in iteration curIterationNum
        /*updateUtopiaPoint();
        System.out.println("utopia:   " + Arrays.toString(objBestCoordinates));
        System.out.println("dystopia: " + Arrays.toString(objWorstCoordinates));
        double meanARI = 0.0;
        for (Particle s: nonDomPSOList) {
            //System.out.println("normalized: " + Arrays.toString(normalize(s.getSolution().getObjectives())));
            System.out.print(Arrays.toString(s.getSolution().getObjectives()));
            System.out.print(" -- Fintess: " + s.getSolution().getFitness());
            double ari = adjustedRandIndex.measure(labelsTrue, s.getSolution().getSolution());
            meanARI += ari;
            System.out.println(" -- ARI: " + ari + " -- distToUtopia: ");
                    //Utils.dist(normalize(s.getSolution().getObjectives()), new double[]{1.0, 1.0}));
        }
        System.out.println("mean ARI for iter: " + meanARI/nonDomPSOList.size());
        System.out.println();*/
    }

    private static List<Particle> determineParetoSet(List<Particle> particleList) {
        List<Particle> nonDomList = new ArrayList<Particle>();
        for (int i = 0; i < particleList.size(); ++i) {
            double maxiMin = Double.NEGATIVE_INFINITY;
            double[] objI = particleList.get(i).getSolution().getObjectives();
            double[] objJ;
            for (int j = 0; j < particleList.size(); ++j) {
                if (i == j) continue;
                objJ = particleList.get(j).getSolution().getObjectives();
                assert (objI.length==objJ.length);

                double min = Double.POSITIVE_INFINITY;
                for (int m = 0; m < objI.length; ++m) {
                    double curDiff = objI[m] - objJ[m];
                    if (curDiff < min) {
                        min = curDiff;
                    }
                }
                if (min > maxiMin) {
                    maxiMin = min;
                }
            }
            particleList.get(i).getSolution().setFitness(maxiMin);
            // max > 0 then solution i is strongly dominated
            if (maxiMin < 0) {
                nonDomList.add(particleList.get(i));
            }
        }
        return nonDomList;
    }

    /** Select global best solution according to certain criteria - proximity to utopia point */

    private Particle pickALeader(boolean pickLeaderRandomly) {
        // update best found solution so far
        Particle leader = null;
        if (pickLeaderRandomly) {
            int iLeader;
            if (conf.numTopParticlesToPickForLeader <= 0.0 || conf.numTopParticlesToPickForLeader > 1.0
                    || conf.numTopParticlesToPickForLeader > nonDomPSOList.size()) {
                iLeader=generator.nextInt(nonDomPSOList.size());
            } else {
                int tmp = (int)(conf.numTopParticlesToPickForLeader * nonDomPSOList.size());
                if (tmp == 0) {
                    iLeader = 0;
                } else {
                    iLeader = generator.nextInt(tmp);
                }
            }
            leader = nonDomPSOList.get(iLeader);
        } else {

            double minDist = Double.POSITIVE_INFINITY;
            assert (nonDomPSOList != null && nonDomPSOList.size() > 0);
            //double[] utopiaCoords = new double[]{1.0, 1.0};
            for (Particle s: nonDomPSOList) {
                double[] cur = s.getSolution().getObjectives();
                //double[] normCur = normalize(cur);
                //double distToUtopia = Utils.dist(normCur, utopiaCoords);
                double distToUtopia = Utils.dist(cur, objBestCoordinates);
                if (distToUtopia < minDist) {
                    leader = s;
                    minDist = distToUtopia;
                }
            }
        }

        assert (leader != null);

        return leader;
    }

    /*private double[] normalize(double[] cur) {
        assert (objBestCoordinates.length == cur.length);
        double[] normCur = cur.clone();
        for (int i = 0; i < normCur.length; ++i) {
            normCur[i] = (cur[i] - objBestCoordinates[i]) / (objWorstCoordinates[i] - objBestCoordinates[i]);
        }
        return normCur;
    }*/

    private void updateUtopiaPoint() {
        // update utopia point
        for (int i = 0; i < swarmSize; ++i) {
            for (int iO = 0; iO < evaluation.length; ++iO) {
                double obj = psoList.get(i).getSolution().getObjective(iO);
                if (obj < objBestCoordinates[iO]) {
                    objBestCoordinates[iO] = obj;
                }
                if (obj > objWorstCoordinates[iO]) {
                    objWorstCoordinates[iO] = obj;
                }
            }
        }
    }

    public void setSeed(int seed) {
        generator = new Random(seed);
    }

    public static Solution dummySolution() {
        Random rnd = new Random();
        return new Solution(new int[]{rnd.nextInt(), rnd.nextInt()}, rnd.nextInt());
    }
    public static double[] dummyArr() {
        Random rnd = new Random();
        return new double[]{rnd.nextDouble(), rnd.nextDouble()};
    }

    public static void main(String[] args) {
        /*double[] obj1 = new double[]{0,2};
        double[] obj2 = new double[]{0.5,0.5};
        double[] obj3 = new double[]{2,0};
        Solution s1 = dummySolution();
        s1.setObjectives(obj1);
        Solution s2 = dummySolution();
        s2.setObjectives(obj2);
        Solution s3 = dummySolution();
        s3.setObjectives(obj3);
        Particle p1 = new Particle(s1,dummyArr());
        Particle p2 = new Particle(s2,dummyArr());
        Particle p3 = new Particle(s3,dummyArr());
        List<Particle> list = new ArrayList<Particle>();
        list.add(p1);
        list.add(p2);
        list.add(p3);
        for (Particle particle: determineParetoSet(list)) {
            System.out.println(particle.getSolution().getFitness());
        }*/
    }
}
