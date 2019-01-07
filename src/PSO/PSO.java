package PSO;

import clustering.*;
import smile.validation.AdjustedRandIndex;
import utils.NCConstruct;
import utils.Pareto;
import utils.Silh;
import utils.Utils;
import weka.core.Instances;

import java.util.*;

/**
 * Created by rusland on 09.09.18.
 * implements PSO algorithm
 */
public class PSO {
    private static final double THRESHOLD = -0.0001;
    private static final double MIN_VEL = -1.0, MAX_VEL = 1.0;
    private static final int SWARM_SIZE_MULTIPLIER = 2;
    private PSOConfiguration conf;
    private int swarmSize;
    private int curIterationNum, numOfIterWithoutImprov;

    private List<Particle> psoList;
    //record of objectives of personal best solution are stored
    //    for the purpose of not calculating them every time pBest is updated
    private Problem problem;
    // solutions will be evaluated on objectives stored in evaluation array
    private Evaluator.Evaluation[] evaluation;
    //private Solution gBestSolution;
    // NCConstruct is used for evaluating connectivy objective function
    private NCConstruct ncc;
    //private Pareto pareto = new Pareto();
    // objBestCoordinates represent utopia point
    private double[] objBestCoordinates, objWorstCoordinates;
    private Random generator;
    private VelocityCalculator velocityCalculator;
    private Solution prevBest;
    private int[] labelsTrue;
    private static int seed_default = 10;
    private static int minSizeOfCluster = 2;

    private AdjustedRandIndex adjustedRandIndex = new AdjustedRandIndex();

    public PSO(Problem aProblem, NCConstruct aNCconstruct, Evaluator.Evaluation[] aEvaluation,
               PSOConfiguration configuration, int[] aLabelsTrue) {
        this.problem = aProblem;
        this.ncc = aNCconstruct;
        this.evaluation = aEvaluation;
        this.conf = configuration;
        this.labelsTrue = aLabelsTrue;
        setSeed(seed_default);

        this.velocityCalculator = new VelocityCalculator(conf.c1, conf.c2);
        velocityCalculator.setSeed(generator.nextInt());
        this.swarmSize = SWARM_SIZE_MULTIPLIER * (conf.maxK - 2 + 1);

        this.psoList = new ArrayList<>();

        int numOfObj = evaluation.length;

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
        /* check whether k-means can produce results */
        boolean canProduce = false;
        for (Particle p: psoList) {
            if (Utils.distinctNumberOfItems(p.getSolution().getSolution()) > 1) {
                canProduce = true;
            }
        }
        if (!canProduce) {
            throw new Exception("Can't produce clustering");
        }

        curIterationNum = 0;
        numOfIterWithoutImprov = 0;

        while(curIterationNum < conf.maxIteration && numOfIterWithoutImprov < conf.maxIterWithoutImprovement) {
            update();
            curIterationNum++;
        }
        // update nonDomPSOList to pick a clusterInstance to utopia point solution out of non-dominated set
        Particle leader;
        if (this.conf.maximin) {
            List<Particle> nonDomPSOList = determineParetoSet(this.psoList);
            if (nonDomPSOList.size() == 0) {
                nonDomPSOList = new ArrayList<Particle>();
                for (Particle p: psoList) {
                    nonDomPSOList.add(new Particle(p));
                }
            }
            setFitness(nonDomPSOList);

            Collections.sort(nonDomPSOList);
            leader = pickALeader(nonDomPSOList, false);
        } else {
            Collections.sort(this.psoList);
            leader = this.psoList.get(0);
        }
        Utils.removeNoise(leader.getSolution().getSolution(), problem.getData(), minSizeOfCluster, 2.0);
        Utils.adjustAssignments(leader.getSolution().getSolution());

        /*System.out.println("--- PSO list start ---");
        printParticlesPerformace(psoList, true);
        System.out.println("--- PSO list end -----");
        System.out.println("NON-DOMINATED SET");
        printParticlesPerformace(nonDomPSOList, true);*/
        //System.out.println("------------------");

        /*System.out.println("Solution found at iteration " + curIterationNum);
        double[] objs = problem.evaluate(leader.getSolution().getSolution(), evaluation, ncc);
        System.out.println("value of objectives: " + Arrays.toString(objs));
        double[] objCloned = objs.clone();
        Utils.normalize(objCloned, this.objBestCoordinates, this.objWorstCoordinates);
        System.out.println("norm value of objectives: " + Arrays.toString(objCloned));
        System.out.println("utopia point: " + Arrays.toString(objBestCoordinates));
        System.out.println("dystopia point: " + Arrays.toString(objWorstCoordinates));*/

        return leader.getSolution().getSolution();
    }

    private void setFitness(List<Particle> particleList) {
        double[][] objs = objectivesFromParticles(particleList);
        for (int i = 0; i < particleList.size(); ++i) {
            double f = fitness(objs[i]);
            assert (f >= 0);
            particleList.get(i).getSolution().setFitness(f);
        }
    }

    private Analyzer printParticlesPerformace(List<Particle> particles) {
        Analyzer analyzer = new Analyzer() {
            @Override
            public void run() throws Exception {
                this.reporter = new Reporter(particles.size());
                this.labelsTrue = PSO.this.labelsTrue;
                this.dataAttrs = PSO.this.problem.getData();
                int i = 0;
                for (Particle s: particles) {
                    Experiment e = measure(s.getSolution().getSolution());
                    this.reporter.set(i, e);
                    ++i;
                }
            }
        };
        try {
            analyzer.run();
            analyzer.analyze(true);
            return analyzer;
        } catch (Exception e) {
            System.out.println("failed to analyze solution at iteration:" + this.curIterationNum);
            return null;
        }
        /*System.out.println(" -- ARI: " + ari + " -- DB: " + db + " -- distToUtopia: " +
                Utils.dist(s.getSolution().getObjectives(), objBestCoordinates));
                //Utils.dist(normalize(s.getSolution().getObjectives()), new double[]{1.0, 1.0}));*/
    }

    public void initializeSwarm() throws Exception {
        Particle p;
        if (conf.equalClusterNumDistribution) {
            for(int i = 2; i <= this.conf.maxK; i++) {
                for (int j = 0; j < SWARM_SIZE_MULTIPLIER; ++j) {
                    // step 1 - randomize particle location using k-means
                    int clusterNum = i; //generator.nextInt(conf.maxK - 2 + 1) + 2;
                    // k-means centroids are initialized and point are assigned to a particular centroid
                    Solution solution = new Solution(kMeansAssignments(clusterNum), clusterNum);

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
            }
        } else {
            for (int i = 0; i < this.swarmSize; i++) {
                // step 1 - randomize particle location using k-means
                int clusterNum;
                if (i < 9) {
                    clusterNum = i + 2;
                } else {
                    clusterNum = generator.nextInt(conf.maxK - 2 + 1) + 2;
                }
                // k-means centroids are initialized and point are assigned to a particular centroid
                Solution solution = new Solution(kMeansAssignments(clusterNum), clusterNum);

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
        }
        assert (psoList.size() == swarmSize);

        /*for(int iP = 0; iP < swarmSize; iP++) {
            p = psoList.get(iP);
            p.getSolution().setObjectives(problem.evaluate(psoList.get(iP).getSolution().getSolution(), evaluation, ncc));
        }*/
        //updateUtopiaPoint(psoList);


        /*System.out.println("INITIAL POPULATION:");
        for(int i = 0; i < swarmSize; i++) {
            Solution sol = psoList.get(i).getSolution();
            System.out.println(Utils.doublePrecision(adjustedRandIndex.measure(sol.getSolution(), labelsTrue), 6));
            System.out.println(Arrays.toString(sol.getObjectives()));
        }
        System.out.println("--------------------");*/
    }

    private int[] kMeansAssignments(int k) throws Exception {
        if (k > Math.sqrt(problem.getN())) {
            throw new Exception("too many clusters");
        }
        KMeans kMeans = new KMeans(k, 2.0);
        kMeans.setSeed(generator.nextInt());
        kMeans.setInitializationMethod(KMeans.Initialization.KMEANS_PLUS_PLUS);
        // perform one iteration of k-mean
        //kMeans.oneIter();
        // perform complete k-means buildClusterer
        kMeans.buildClusterer(problem.getData());
        int[] labelsPred = kMeans.getLabels();
        //Utils.removeNoise(labelsPred, problem.getData(), minSizeOfCluster, 2.0);
        //Utils.adjustAssignments(labelsPred);
        return labelsPred;
    }

    private void update() {
        // step 1 - Evaluate objective functions for each particle
        for(int iP = 0; iP < swarmSize; iP++) {
            Particle p = psoList.get(iP);
            p.getSolution().setObjectives(problem.evaluate(psoList.get(iP).getSolution().getSolution(), evaluation, ncc));
        }
        /* update before  */
        updateUtopiaPoint(psoList);

        if (this.conf.maximin) {
            setFitness(psoList);

            // step 2 - determine pareto-optimal set
            List<Particle> nonDomPSOList = determineParetoSet(psoList);
            List<Particle> psoListCloned = null;
            if (nonDomPSOList.size() != 0) {
                // compute fitness
                setFitness(nonDomPSOList);
                Collections.sort(nonDomPSOList);
            } else {
                psoListCloned = new ArrayList<Particle>();
                for (Particle p: psoList) {
                    psoListCloned.add(new Particle(p));
                }
                setFitness(psoListCloned);
                Collections.sort(psoListCloned);
            }

            // step 3 - Update Personal Best of each particle
            for (int i = 0; i < swarmSize; i++) {
                double[] curIterObjs = problem.evaluate(psoList.get(i).getSolution().getSolution(), evaluation, ncc);//psoList.get(i).getSolution().getObjectives().clone();
                utils.Utils.normalize(curIterObjs, objBestCoordinates, objWorstCoordinates);
                Solution prev = psoList.get(i).getPrev();
                double[] prevObjs = null;
                if (prev != null) {
                    prevObjs = problem.evaluate(prev.getSolution(), evaluation, ncc);
                    utils.Utils.normalize(prevObjs, objBestCoordinates, objWorstCoordinates);
                }

                if (prev == null) {
                    // before first update; prev not set yet
                    psoList.get(i).setpBest(psoList.get(i).getSolution());
                } else if (Pareto.testDominance(prevObjs, curIterObjs, false)) {
                    // reverse order, since objectives should be minimized in this case

                    // set to prev if dom
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

            for (Particle aNextPopList: nextPopList) {
                Particle paretoLeaderSolution;
                if (nonDomPSOList.size() != 0) {
                    paretoLeaderSolution = pickALeader(nonDomPSOList, true);
                } else {
                    paretoLeaderSolution = pickALeader(psoListCloned, true);
                }
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

            // copy main psoList particles to nextPopList
            for (Particle aPsoList : psoList) {
                nextPopList.add(new Particle(aPsoList));
            }

            updateUtopiaPoint(nextPopList);
            List<Particle> nonDomFromNextPSOList = determineParetoSet(nextPopList);
            setFitness(nonDomFromNextPSOList);

            // copy non-dominated solutions to next generation
            psoList = new ArrayList<Particle>();
            for (int i = 0; i < nonDomFromNextPSOList.size(); ++i) {
                if (i >= this.swarmSize) {
                    break;
                }
                psoList.add(nonDomFromNextPSOList.get(i));
            }

            int roomToFill = swarmSize - psoList.size();
            // first - remove non-dominated
            nextPopList.removeIf(particle -> (particle.getSolution().getFitness() < THRESHOLD));

            psoListCloned = new ArrayList<Particle>();
            for (Particle p: nextPopList) {
                psoListCloned.add(new Particle(p));
            }
            setFitness(psoListCloned);
            Collections.sort(psoListCloned);
            int idx = 0;
            while (idx < roomToFill) {
                psoList.add(psoListCloned.get(idx));
                idx++;
            }

        /* recompute non-dom set to compare with previous one */
            nonDomPSOList = determineParetoSet(psoList);
            if (nonDomPSOList.size() != 0) {
                setFitness(nonDomPSOList);
                Collections.sort(nonDomPSOList);
            } else {
                psoListCloned = new ArrayList<Particle>();
                for (Particle p: psoList) {
                    psoListCloned.add(new Particle(p));
                }
                setFitness(psoListCloned);
                Collections.sort(psoListCloned);
            }

            // naive method to identify whether pareto set changed
            //if (this.nonDomPSOList.size() == prevParetoSize) {
            Solution curBest;
            if (nonDomPSOList.size() != 0) {
                curBest = pickALeader(nonDomPSOList, false).getSolution();
            } else {
                curBest = pickALeader(psoListCloned, false).getSolution();
            }
            if (prevBest != null && curBest.equals(prevBest)) {
                numOfIterWithoutImprov++;
            } else {
                numOfIterWithoutImprov = 0;
            }
            prevBest = new Solution(curBest);
        } else {
            // adjust fitness after objective normalization
            setFitness(psoList);

            // update pBest
            for (int i = 0; i < swarmSize; i++) {
                double curFitness = psoList.get(i).getSolution().getFitness();
                Solution prev = psoList.get(i).getSolution();
                double pBestFitness;
                if (prev != null) {
                    double[] prevObjs = problem.evaluate(prev.getSolution(), evaluation, ncc);
                    Utils.normalize(prevObjs, objBestCoordinates, objWorstCoordinates);
                    pBestFitness = fitness(prevObjs);
                } else {
                    pBestFitness = Double.POSITIVE_INFINITY;
                }
                if (prev == null) {
                    psoList.get(i).setpBest(psoList.get(i).getSolution());
                } else if (curFitness < pBestFitness) {
                    psoList.get(i).setpBest(psoList.get(i).getSolution());
                }
            }

            List<Particle> nextPopList = new ArrayList<Particle>(swarmSize * 2);
            for (int i = 0; i < psoList.size(); ++i) {
                nextPopList.add(new Particle(psoList.get(i)));
            }

            // update cloned particles
            double w = (conf.maxW - conf.minW) * (conf.maxIteration - curIterationNum) / conf.maxIteration + conf.minW;
            velocityCalculator.setW(w);

            List<Particle> popToPickLeader = new ArrayList<Particle>(swarmSize);
            for (int i = 0; i < psoList.size(); ++i) {
                popToPickLeader.add(new Particle(psoList.get(i)));
            }
            setFitness(popToPickLeader);
            Collections.sort(popToPickLeader);

            for (Particle aNextPopParticle : nextPopList) {
                int tmp = (int)(conf.numTopParticlesToPickForLeader * popToPickLeader.size());
                int iLeader = 0;
                if (tmp != 0) {
                    iLeader = generator.nextInt(tmp);
                }
                Particle paretoLeaderSolution = popToPickLeader.get(iLeader);
                double[] newVel = velocityCalculator.calculate(aNextPopParticle);
                aNextPopParticle.setVelocity(newVel);
                Utils.checkClusterLabels(paretoLeaderSolution.getSolution().getSolution(),
                        paretoLeaderSolution.getSolution().getK(false));
                // update and check for boundaries: k cannot be more than kMax
                aNextPopParticle.update(paretoLeaderSolution.getSolution(), conf.maxK);
            }

            // evaluate
            for (Particle p : nextPopList) {
                double[] tmpObj = problem.evaluate(p.getSolution().getSolution(), evaluation, ncc);
                p.getSolution().setObjectives(tmpObj);
            }
            //update objective boundaries
            updateUtopiaPoint(nextPopList);
            //adjust fitness
            setFitness(nextPopList);

            // copy main psoList particles to nextPopList
            for (Particle aParticle : psoList) {
                nextPopList.add(new Particle(aParticle));
            }

            Collections.sort(nextPopList);
            psoList = new ArrayList<Particle>(this.swarmSize);
            for (int i = 0; i < this.swarmSize; ++i) {
                psoList.add(nextPopList.get(i));
            }

            Solution curBest = psoList.get(0).getSolution();
            if (prevBest != null && curBest.equals(prevBest)) {
                numOfIterWithoutImprov++;
            } else {
                numOfIterWithoutImprov = 0;
            }
            prevBest = new Solution(curBest);

            /*int[] labelsPred = psoList.get(0).getSolution().getSolution();
            HashMap<Integer, double[]> centroids = Utils.centroids(this.problem.getData(), labelsPred);
            double aRIScore = this.adjustedRandIndex.measure(this.labelsTrue, labelsPred);
            double dbScore = Utils.dbIndexScore(centroids, labelsPred, this.problem.getData());
            double silhScore = new Silh().compute(centroids, labelsPred, this.problem.getData());
            int numClusters = Utils.distinctNumberOfItems(labelsPred);
            System.out.println("A: " + aRIScore);
            System.out.println("D: " + dbScore);
            System.out.println("S: " + silhScore);
            System.out.println("K: " + numClusters);*/

        }
    }

    // fitness is minimized in this case
    private double fitness(double[] objs) {
        for (double obj: objs) {
            assert (obj >= 0);
        }
        return utils.Utils.sum(objs, this.conf.weights, 2.0);
    }

    private List<Particle> determineParetoSet(List<Particle> particleList) {
        double[][] objList = objectivesFromParticles(particleList);

        double[] fitness = Utils.determineParetoSet(objList);
        List<Particle> result = new ArrayList<>();
        for (int i = 0; i < particleList.size(); ++i) {
            if (result.size() >= this.conf.pMax) {
                break;
            }
            // max > 0 then solution i is strongly dominated
            if (fitness[i] < THRESHOLD) {
                result.add(particleList.get(i));
            }
        }
        return result;
    }

    private double[][] objectivesFromParticles(List<Particle> particles) {
        double[][] objsResult = new double[particles.size()][];
        for (int i = 0; i < particles.size(); ++i) {
            objsResult[i] = problem.evaluate(particles.get(i).getSolution().getSolution(), evaluation, ncc); //particles.get(i).getSolution().getObjectives().clone();
        }
        utils.Utils.normalize(objsResult, objBestCoordinates, objWorstCoordinates);

        return objsResult;
    }

    /** Select global best solution according to certain criteria - proximity to utopia point */
    private Particle pickALeader(List<Particle> particleList, boolean pickLeaderRandomly) {
        // update best found solution so far
        assert (particleList != null && particleList.size() != 0);
        Particle leader = null;
        if (pickLeaderRandomly) {
            int iLeader;
            assert (conf.numTopParticlesToPickForLeader > 0.0 || conf.numTopParticlesToPickForLeader <= 1.0);

            int tmp = (int)(conf.numTopParticlesToPickForLeader * particleList.size());
            if (tmp == 0) {
                iLeader = 0;
            } else {
                iLeader = generator.nextInt(tmp);
            }
            leader = particleList.get(iLeader);
        } else {
            assert (particleList.size() > 0);
            double[][] objs = objectivesFromParticles(particleList);
            assert (objs.length > 0);
            int leaderIdx = -1;
            leaderIdx = Utils.pickClosestToUtopia(objs, new double[]{0.0, 0.0}, this.conf.weights, 2.0);
            assert (leaderIdx != -1);
            leader = particleList.get(leaderIdx);
        }

        assert (leader != null);

        return leader;
    }

    private void updateUtopiaPoint(List<Particle> swarm) {
        // update utopia point
        for (int i = 0; i < swarm.size(); ++i) {
            double[] objs = problem.evaluate(swarm.get(i).getSolution().getSolution(), evaluation, ncc);
            for (int iO = 0; iO < evaluation.length; ++iO) {
                double obj = objs[iO]; //swarm.get(i).getSolution().getObjective(iO);
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
