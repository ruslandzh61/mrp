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
    private static final double THRESHOLD = -0.000001;
    private static final double MIN_VEL = -1.0, MAX_VEL = 1.0;
    private static final int MIN_SWARM_SIZE = 20;
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
    private int minSizeOfCluster;

    AdjustedRandIndex adjustedRandIndex = new AdjustedRandIndex();

    public PSO(Problem aProblem, NCConstruct aNCconstruct, Evaluator.Evaluation[] aEvaluation,
               PSOConfiguration configuration, Instances aInstances, int[] aLabelsTrue) {
        this.problem = aProblem;
        this.ncc = aNCconstruct;
        this.evaluation = aEvaluation;
        this.conf = configuration;
        this.instances = aInstances;
        this.labelsTrue = aLabelsTrue;
        minSizeOfCluster = (int) (Math.sqrt(problem.getN())/2);
        setSeed(seed_default);

        this.velocityCalculator = new VelocityCalculator(conf.c1, conf.c2);
        velocityCalculator.setSeed(generator.nextInt());
        this.swarmSize = 2 * conf.maxK;
        if (swarmSize < MIN_SWARM_SIZE) {
            swarmSize = MIN_SWARM_SIZE;
        }
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

        while(curIterationNum < conf.maxIteration && numOfIterWithoutImprov < conf.maxIterWithoutImprovement) {
            if (this.curIterationNum % 20 == 0) {
                System.out.println("ITERATION " + curIterationNum + ": ");
            }
            update();
            if (this.curIterationNum % 20 == 0) {
                System.out.println("without improv: " + numOfIterWithoutImprov);
            }
            curIterationNum++;
        }
        // update nonDomPSOList
        nonDomPSOList = determineParetoSet(psoList, objBestCoordinates, objWorstCoordinates);
        Particle leader = pickALeader(false);
        Utils.removeNoise(leader.getSolution().getSolution(), problem.getData(), minSizeOfCluster, 2.0);
        Utils.adjustAssignments(leader.getSolution().getSolution());

        System.out.println("--- PSO list start ---");
        printParticlesPerformace(psoList);
        System.out.println("--- PSO list end -----");
        System.out.println("NON-DOMINATED SET");
        printParticlesPerformace(nonDomPSOList);
        System.out.println("------------------");

        System.out.println("Solution found at iteration " + curIterationNum);
        double[] objs = problem.evaluate(leader.getSolution().getSolution(), evaluation, ncc);
        System.out.println("value of objectives: " + Arrays.toString(objs));
        System.out.println("norm value of objectives: " + Arrays.toString(
                Utils.normalize(objs, this.objBestCoordinates, this.objWorstCoordinates)));
        System.out.println("utopia point: " + Arrays.toString(objBestCoordinates));
        System.out.println("dystopia point: " + Arrays.toString(objWorstCoordinates));

        return leader.getSolution().getSolution();
    }

    private void printParticlesPerformace(List<Particle> particles) {
        double meanARI = 0.0;
        double meanDB = 0.0;
        double meanK = 0.0;
        double meanKWithout = 0.0;
        for (Particle s: particles) {
            //System.out.println("normalized: " + Arrays.toString(normalize(s.getSolution().getObjectives())));
            //System.out.print(Arrays.toString(s.getSolution().getObjectives()));
            //System.out.print(" -- Fintess: " + s.getSolution().getFitness());
            double ari = adjustedRandIndex.measure(labelsTrue, s.getSolution().getSolution());
            double db = Utils.dbIndexScore(Utils.centroids(problem.getData(), s.getSolution().getSolution()),
                    s.getSolution().getSolution(), problem.getData());
            meanARI += ari;
            meanDB += db;
            meanK += s.getSolution().getK(false);
            meanKWithout += s.getSolution().getK(true);
            System.out.println("ARI score: " + ari);
            System.out.println("value of objectives: " + Arrays.toString(s.getSolution().getObjectives()));
            System.out.println("norm value of objectives: " + Arrays.toString(Utils.normalize(
                    s.getSolution().getObjectives(), objBestCoordinates, objWorstCoordinates)));
            /*System.out.println(" -- ARI: " + ari + " -- DB: " + db + " -- distToUtopia: " +
                    Utils.dist(s.getSolution().getObjectives(), objBestCoordinates));
                    //Utils.dist(normalize(s.getSolution().getObjectives()), new double[]{1.0, 1.0}));*/
        }
        System.out.println("mean ARI for iter: " + meanARI / particles.size());
        System.out.println("mean DB for iter: " + meanDB / particles.size());
        System.out.println("mean k: " + meanK / particles.size());
        System.out.println("mean k without: " + meanKWithout / particles.size());
    }

    public void initializeSwarm() throws Exception {
        Particle p;
        for(int i = 0; i < swarmSize; i++) {
            // step 1 - randomize particle location using k-means
            int clusterNum = generator.nextInt(conf.maxK - 2 + 1) + 2;
            // k-means centroids are initialized and point are assigned to a particular centroid

            Solution solution = new Solution(kMeansAssignments(instances, clusterNum), clusterNum);

            // step 2 -randomize velocity in the defined range
            double[] vel = new double[problem.getN()];
            for (int dimIdx = 0; dimIdx < problem.getN(); ++dimIdx) {
                vel[dimIdx] = MIN_VEL + generator.nextDouble() * (
                        MAX_VEL - MIN_VEL);
            }
            p = new Particle(solution, vel);
            p.setSeed(generator.nextInt());
            psoList.add(p);
        }

        for(int iP = 0; iP < swarmSize; iP++) {
            p = psoList.get(iP);
            p.getSolution().setObjectives(problem.evaluate(psoList.get(iP).getSolution().getSolution(), evaluation, ncc));
        }
        updateUtopiaPoint(psoList);


        System.out.println("INITIAL POPULATION:");
        for(int i = 0; i < swarmSize; i++) {
            Solution sol = psoList.get(i).getSolution();
            System.out.println(Utils.doublePrecision(adjustedRandIndex.measure(sol.getSolution(), labelsTrue), 6));
            System.out.println(Arrays.toString(sol.getObjectives()));
        }
        System.out.println("--------------------");
    }

    private int[] kMeansAssignments(int k) {
        KMeans kMeans = new KMeans(problem.getData(), problem.getN(), problem.getD(), k, 2.0);
        kMeans.setSeed(generator.nextInt());
        kMeans.setPlus(true);
        // perform one iteration of k-mean
        //kMeans.oneIter();
        // perform complete k-means buildClusterer
        kMeans.buildClusterer(500);
        int[] labelsPred = kMeans.getLabels();
        Utils.removeNoise(labelsPred, problem.getData(), minSizeOfCluster, 2.0);
        Utils.adjustAssignments(labelsPred);
        return labelsPred;
    }

    private int[] kMeansAssignments(Instances instances, int k) throws Exception {
        SimpleKMeans kMeans = new SimpleKMeans();
        kMeans.setSeed(generator.nextInt());
        SelectedTag selectedTag = new SelectedTag(SimpleKMeans.KMEANS_PLUS_PLUS, SimpleKMeans.TAGS_SELECTION);
        kMeans.setInitializationMethod(selectedTag);
        kMeans.setPreserveInstancesOrder(true);
        kMeans.setMaxIterations(500);
        kMeans.setNumClusters(k);
        kMeans.buildClusterer(instances);
        int[] labelsPred = kMeans.getAssignments();
        Utils.removeNoise(labelsPred, problem.getData(), minSizeOfCluster, 2.0);
        Utils.adjustAssignments(labelsPred);
        return labelsPred;
    }

    private void update() {
        // step 1 - Evaluate objective functions for each particle
        for(int iP = 0; iP < swarmSize; iP++) {
            Particle p = psoList.get(iP);
            p.getSolution().setObjectives(problem.evaluate(psoList.get(iP).getSolution().getSolution(), evaluation, ncc));
        }

        // step 2 - determine pareto-optimal set
        nonDomPSOList = determineParetoSet(psoList, objBestCoordinates, objWorstCoordinates);
        /* in case there are no strongly non-dominated particles */
        if (nonDomPSOList.size() == 0) {
            for (int i = 0; i < psoList.size(); ++i) {
                if (psoList.get(i).getSolution().getFitness() >= THRESHOLD) {
                    nonDomPSOList.add(new Particle(psoList.get(i)));
                    break;
                }
            }
        }
        if (nonDomPSOList.size() == 0) {
            nonDomPSOList.add(psoList.get(generator.nextInt(swarmSize)));
        }

        assert (nonDomPSOList != null);
        assert (nonDomPSOList.size() != 0);
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

        for (Particle aNextPopList : nextPopList) {
            Particle paretoLeaderSolution = pickALeader(true);
            double[] newVel = velocityCalculator.calculate(aNextPopList);
            aNextPopList.setVelocity(newVel);
            Utils.checkClusterLabels(paretoLeaderSolution.getSolution().getSolution(),
                    paretoLeaderSolution.getSolution().getK(false));
            // update and check for boundaries: k cannot be more than kMax
            aNextPopList.update(paretoLeaderSolution.getSolution(), conf.maxK);
        }

        // evaluate before determining pareto set
        for (Particle p : nextPopList) {
            p.getSolution().setObjectives(problem.evaluate(p.getSolution().getSolution(), evaluation, ncc));
        }
        updateUtopiaPoint(nextPopList);

        // copy main psoList particles to nextPopList
        for (Particle aPsoList : psoList) {
            nextPopList.add(new Particle(aPsoList));
        }

        List<Particle> nonDomFromNextPSOList = determineParetoSet(nextPopList, objBestCoordinates, objWorstCoordinates);

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
        nextPopList.removeIf(particle -> (particle.getSolution().getFitness() < THRESHOLD));
        // now randomly pick nextPopList
        int i = 0;
        while (i < roomToFill) {
            int idxPick  = generator.nextInt(nextPopList.size());
            if (nextPopList.get(idxPick) != null) {
                psoList.add(nextPopList.get(idxPick));
                i++;
                nextPopList.remove(idxPick);
            }
        }

        // naive method to identify whether pareto set changed
        if (this.nonDomPSOList.size() == prevParetoSize) {
            numOfIterWithoutImprov++;
        } else {
            numOfIterWithoutImprov = 0;
        }
        prevParetoSize = this.nonDomPSOList.size();

        /*if (this.curIterationNum % 20 == 0) {
            System.out.println("utopia:   " + Arrays.toString(objBestCoordinates));
            System.out.println("dystopia: " + Arrays.toString(objWorstCoordinates));
        }*/
    }

    private static List<Particle> determineParetoSet(
            List<Particle> particleList, double[] objsLow, double[] objsHigh) {
        List<Particle> nonDomList = new ArrayList<Particle>();
        for (int i = 0; i < particleList.size(); ++i) {
            double maxiMin = Double.NEGATIVE_INFINITY;
            double[] objI = Utils.normalize(particleList.get(i).getSolution().getObjectives(), objsLow, objsHigh);
            double[] objJ;
            for (int j = 0; j < particleList.size(); ++j) {
                if (i == j) continue;
                objJ = Utils.normalize(particleList.get(j).getSolution().getObjectives(), objsLow, objsHigh);
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
            if (maxiMin < THRESHOLD) {
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
            assert (conf.numTopParticlesToPickForLeader > 0.0 || conf.numTopParticlesToPickForLeader <= 1.0);
            /*if (conf.numTopParticlesToPickForLeader <= 0.0 || conf.numTopParticlesToPickForLeader > 1.0
                    || conf.numTopParticlesToPickForLeader > nonDomPSOList.size()) {
                iLeader=generator.nextInt(nonDomPSOList.size());
            } else {*/
            int tmp = (int)(conf.numTopParticlesToPickForLeader * nonDomPSOList.size());
            if (tmp == 0) {
                iLeader = 0;
            } else {
                iLeader = generator.nextInt(tmp);
            }
            leader = nonDomPSOList.get(iLeader);
        } else {

            double minDist = Double.POSITIVE_INFINITY;
            assert (nonDomPSOList != null && nonDomPSOList.size() > 0);
            double[] utopiaCoords = new double[]{0.0, 0.0};
            for (Particle s: nonDomPSOList) {
                double[] cur = s.getSolution().getObjectives();
                double[] normCur = Utils.normalize(cur, this.objBestCoordinates, objWorstCoordinates);
                double distToUtopia = Utils.dist(normCur, utopiaCoords, 2.0);
                //double distToUtopia = Utils.dist(cur, objBestCoordinates, 2.0);
                if (distToUtopia < minDist) {
                    leader = s;
                    minDist = distToUtopia;
                }
            }
        }

        assert (leader != null);

        return leader;
    }

    private void updateUtopiaPoint(List<Particle> swarm) {
        // update utopia point
        for (int i = 0; i < swarm.size(); ++i) {
            for (int iO = 0; iO < evaluation.length; ++iO) {
                double obj = swarm.get(i).getSolution().getObjective(iO);
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
