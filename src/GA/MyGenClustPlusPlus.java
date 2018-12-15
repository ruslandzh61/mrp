//
// Source code recreated from a .class file by IntelliJ IDEA
// (powered by Fernflower decompiler)
//

package GA;

import java.util.*;

import PSO.Evaluator;
import PSO.Particle;
import smile.validation.AdjustedRandIndex;
import utils.NCConstruct;
import weka.classifiers.rules.DecisionTableHashKey;
import weka.clusterers.Canopy;
import weka.clusterers.RandomizableClusterer;
import weka.clusterers.SimpleKMeans;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.DenseInstance;
import weka.core.DistanceFunction;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.ManhattanDistance;
import weka.core.SelectedTag;
import weka.core.Tag;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.Capabilities.Capability;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

public class MyGenClustPlusPlus extends RandomizableClusterer implements TechnicalInformationHandler {
    private static final long serialVersionUID = -7247404718496233612L;
    private Instances m_data;
    private int m_numberOfClusters = 0;
    private int m_numberOfGenerations = 60;
    private ManhattanDistance m_distFunc;
    private int m_initialPopulationSize = 30;
    private int m_maxKMeansIterationsInitial = 60;
    private int m_maxKMeansIterationsQuick = 15;
    private int m_maxKMeansIterationsFinal = 50;
    private double m_duplicateThreshold = 0.0D;
    private int m_startChromosomeSelectionGeneration = 11;
    private MyGenClustPlusPlus.MKMeans m_bestChromosome;
    private double m_bestFitness;
    private int m_bestFitnessIndex;
    protected MyGenClustPlusPlus.MKMeans m_builtClusterer = null;
    private Random m_rand = null;
    private boolean m_dontReplaceMissing = false;
    public static final int SUPPLIED = 4;
    public static final Tag[] TAGS_SELECTION_MK = new Tag[]{new Tag(0, "Random"), new Tag(1, "k-means++"), new Tag(2, "Canopy"), new Tag(3, "Farthest first"), new Tag(4, "Supplied Centroids")};

    public void setNcConstruct(NCConstruct ncConstruct) {
        this.ncConstruct = ncConstruct;
    }

    private NCConstruct ncConstruct;
    private AdjustedRandIndex adjustedRandIndex = new AdjustedRandIndex();
    private MyGenClustPlusPlus.MKMeans[] finalpopulation;
    private Evaluator evaluator = new Evaluator();
    private Evaluator.Evaluation[] evaluations;
    private double[][] myData;

    public void setTrueLabels(int[] trueLabels) {
        this.trueLabels = trueLabels;
    }

    private int[] trueLabels;
    private double[] objBestCoordinates, objWorstCoordinates;
    private static final double THRESHOLD = 0.000001;

    public void setMyData(double[][] myData) {
        this.myData = myData;
    }


    public void setEvaluator(Evaluator evaluator) {
        this.evaluator = evaluator;
    }

    public void setEvaluations(Evaluator.Evaluation[] evaluations) {
        this.evaluations = evaluations;
        int numOfObj = evaluations.length;
        objBestCoordinates = new double[numOfObj];
        for (int iO = 0; iO < numOfObj; ++iO) {
            objBestCoordinates[iO] = Double.POSITIVE_INFINITY;
        }
        objWorstCoordinates = new double[evaluations.length];
        for (int iO = 0; iO < numOfObj; ++iO) {
            objWorstCoordinates[iO] = Double.NEGATIVE_INFINITY;
        }
    }


    public int numberOfClusters() {
        return this.m_numberOfClusters;
    }

    public MyGenClustPlusPlus() {
        this.m_SeedDefault = 10;
        this.setSeed(this.m_SeedDefault);
    }

    public void buildClusterer(Instances data) throws Exception {
        this.m_rand = new Random((long)this.getSeed());
        this.getCapabilities().testWithFail(data);
        this.m_data = new Instances(data);

        this.m_distFunc = new ManhattanDistance(this.m_data);
        /* Component 3: generate initial population */
        MyGenClustPlusPlus.MKMeans[] initialPopulation = this.generateInitialPopulation(this.m_data);
        /* evaluate and update utopia point */
        double[][] tmpObjectives = new double[initialPopulation.length][];
        for (int i = 0; i < initialPopulation.length; ++i) {
            tmpObjectives[i] = evaluate(initialPopulation[i].getAssignments());
        }
        updateUtopiaPoint(tmpObjectives);

        assert (initialPopulation != null);

        this.m_bestFitness = -1.0D / 0.0;
        this.m_bestFitnessIndex = -2147483648;

        /* identify best chromosome */
        for(int selectedPopulation = 0; selectedPopulation < initialPopulation.length; ++selectedPopulation) {
            double previous = this.fitness(initialPopulation[selectedPopulation]);
            if(previous > this.m_bestFitness) {
                if(previous == 1.0D / 0.0) {
                    this.m_builtClusterer = initialPopulation[selectedPopulation];
                    this.m_numberOfClusters = initialPopulation[selectedPopulation].getClusterCentroids().size();
                    System.out.println("failed in selection of initial population out of pre-initial population");
                    return;
                }

                this.m_bestFitness = previous;
                this.m_bestFitnessIndex = selectedPopulation;
            }
        }
        this.m_bestChromosome = new MyGenClustPlusPlus.MKMeans(initialPopulation[this.m_bestFitnessIndex]);

        /* select initial population out of pre-initial population probabilistically
         * When a value of k is chosen, the next available solution in order of descending fitness
         * is added to initial population. */
        MyGenClustPlusPlus.MKMeans[] mainPopulation = this.probabilisticSelection(initialPopulation);
        MyGenClustPlusPlus.MKMeans[] prevPopulation = new MyGenClustPlusPlus.MKMeans[mainPopulation.length];

        int newBestIndex;
        MyGenClustPlusPlus.MKMeans finalRun;
        /* 60 generations */
        for(int generationIdx = 0; generationIdx <= this.m_numberOfGenerations; ++generationIdx) {
            MyGenClustPlusPlus.MKMeans[] resultingPopulation;
            int f;
            // step 1 - crossover or prob. cloning
            if((generationIdx + 1) % 10 != 0) {
                resultingPopulation = this.crossover(mainPopulation);
            } else {
                /* improve At every 10th iteration (i.e. the 10th, 20th, 30th etc.)
                we perform a probabilistic cloning of the chromosomes.
                We clone chromosomes with probability proportional to
                the fitness of the chromosomes as per the wheel technique. */

                // step 1 - probabilistic cloning
                resultingPopulation = this.probabilisticCloning(mainPopulation);
                // step 2 - elitism
                resultingPopulation = this.elitism(resultingPopulation);

                /* step 3 - polishing. undergo a short length (15 iterations) MK-Means.
                 * The application of a few iterations of the hill-climber of MK-Means is justified
                 * because generally the K-Means hill-climber reaches a reasonably good buildClusterer solution
                 * within a small number of iterations ( Rahman et al., 2014 ).
                 * Moreover, the aim of the MK-Means here is not to produce a final buildClusterer solutions,
                 * but to repair any slight fitness damage caused by the mutation operation
                 * to maintain a population of high achievers.
                 * We refer to the application of the hill-climber as polishing. */
                for(newBestIndex = 0; newBestIndex < resultingPopulation.length; ++newBestIndex) {
                    do {
                        finalRun = new MyGenClustPlusPlus.MKMeans();
                        finalRun.setSeed(this.m_rand.nextInt());
                        finalRun.setInitializationMethod(new SelectedTag(4, TAGS_SELECTION_MK));
                        finalRun.setInitial(resultingPopulation[newBestIndex].getClusterCentroids());

                        finalRun.setMaxIterations(this.m_maxKMeansIterationsQuick);
                        finalRun.setDontReplaceMissingValues(this.m_dontReplaceMissing);
                        finalRun.setPreserveInstancesOrder(true);
                        finalRun.buildClusterer(this.m_data, this.m_distFunc);
                        if(finalRun.getClusterCentroids().numInstances() <= 1) {
                            f = this.m_rand.nextInt((int)(Math.sqrt((double)data.size()) - 2.0D)) + 2;
                            MyGenClustPlusPlus.MKMeans t = new MyGenClustPlusPlus.MKMeans();
                            t.setSeed(this.m_rand.nextInt());
                            t.setNumClusters(f);
                            t.setDontReplaceMissingValues(this.m_dontReplaceMissing);
                            t.setPreserveInstancesOrder(true);
                            t.buildClusterer(data, this.m_distFunc);
                            resultingPopulation[newBestIndex] = new MyGenClustPlusPlus.MKMeans(t);
                        }
                    } while(finalRun.getClusterCentroids().numInstances() <= 1);

                    resultingPopulation[newBestIndex] = finalRun;
                }
            }
            /* step 2 (4 of prob. cloning) - elitism */
            resultingPopulation = this.elitism(resultingPopulation);
            /* step 3 (5 of prob. cloning) - mutation */
            resultingPopulation = this.mutation(resultingPopulation);
            /* step 4 (6 of prob. cloning) - elitism */
            resultingPopulation = this.elitism(resultingPopulation);
            if(this.m_builtClusterer != null) {
                System.out.println("failed after crossover/probabilistic cloning");
                return;
            }

            if(generationIdx <= this.m_startChromosomeSelectionGeneration) {
                /* copy resulting population into main population instance */
                for(newBestIndex = 0; newBestIndex < resultingPopulation.length; ++newBestIndex) {
                    mainPopulation[newBestIndex] = new MyGenClustPlusPlus.MKMeans(resultingPopulation[newBestIndex]);
                }
            } else {
                /* start chromosome selection at this point
                 * merges all chromosomes between two populations of size s:
                 * the last (most recent) population and the resulting generation from all operations.
                 * This second type of elitism chooses the highest fitted s chromosomes from the merged populations.
                 * This restricts radical disruption by the genetic operators,
                 * but it still enables their exploratory nature to drive the genetic search. */
                /* fitness corresponding to resulting population */
                MyGenClustPlusPlus.FitnessContainer[] var18 = new MyGenClustPlusPlus.FitnessContainer[resultingPopulation.length];
                /* fitness corresponding to previous population */
                MyGenClustPlusPlus.FitnessContainer[] var19 = new MyGenClustPlusPlus.FitnessContainer[prevPopulation.length];

                for(f = 0; f < resultingPopulation.length; ++f) {
                    var18[f] = new MyGenClustPlusPlus.FitnessContainer(this.fitness(resultingPopulation[f]), resultingPopulation[f]);
                }

                for(f = 0; f < prevPopulation.length; ++f) {
                    var19[f] = new MyGenClustPlusPlus.FitnessContainer(this.fitness(prevPopulation[f]), prevPopulation[f]);
                }

                MyGenClustPlusPlus.FitnessContainer[] var22 = new MyGenClustPlusPlus.FitnessContainer[resultingPopulation.length * 2];

                int var21;
                for(var21 = 0; var21 < resultingPopulation.length; ++var21) {
                    var22[var21] = var18[var21];
                }

                var21 = 0;

                int i;
                for(i = resultingPopulation.length; i < resultingPopulation.length * 2; ++i) {
                    var22[i] = var19[var21++];
                }

                Arrays.sort(var22, Collections.reverseOrder());
                for(i = 0; i < resultingPopulation.length; ++i) {
                    mainPopulation[i] = new MyGenClustPlusPlus.MKMeans(var22[i].clustering);
                }
            }

            /* update previous population */
            for(newBestIndex = 0; newBestIndex < mainPopulation.length; ++newBestIndex) {
                prevPopulation[newBestIndex] = new MyGenClustPlusPlus.MKMeans(mainPopulation[newBestIndex]);
            }
            System.out.println("best objective coordinates: " + Arrays.toString(objBestCoordinates));
        }
        /* FINAL step - choose best performing chromosome to use as
         * the initial solution of a final full-length MK-Means to deliver the final buildClusterer solution */
        double var17 = 4.9E-324D;
        newBestIndex = -1;

        this.finalpopulation = mainPopulation;
        /* evaluate population */
        double[][] chromosomeIdxToObjs = new double[mainPopulation.length][];
        for (int i = 0; i < mainPopulation.length; ++i) {
            chromosomeIdxToObjs[i] = evaluate(mainPopulation[i].getAssignments());
        }
        /* update utopia point */
        updateUtopiaPoint(chromosomeIdxToObjs);
        /* MaxiMin strategy */
        HashSet<Integer> nonDomIndices = determineParetoSet(chromosomeIdxToObjs);
        if (nonDomIndices.size() > 0) {
            newBestIndex = pickALeader(nonDomIndices, chromosomeIdxToObjs);
        } else {

        }
        /* choose final cluster solution */
        System.out.println(Arrays.toString(nonDomIndices.toArray(new Integer[nonDomIndices.size()])));

        measureFinalPop(myData, trueLabels);
        System.out.println("--------------");
        /*for(int var20 = 0; var20 < mainPopulation.length; ++var20) {
            double var23 = this.fitness(mainPopulation[var20]);
            if(var23 > var17) {
                var17 = var23;
                newBestIndex = var20;
            }
        }*/
        this.m_bestChromosome = new MyGenClustPlusPlus.MKMeans(mainPopulation[newBestIndex]);
        this.m_bestFitness = var17;
        finalRun = new MyGenClustPlusPlus.MKMeans();
        finalRun.setSeed(this.m_rand.nextInt());
        finalRun.setInitializationMethod(new SelectedTag(4, TAGS_SELECTION_MK));
        finalRun.setInitial(this.m_bestChromosome.getClusterCentroids());
        finalRun.setDontReplaceMissingValues(this.m_dontReplaceMissing);
        finalRun.setPreserveInstancesOrder(true);
        finalRun.setMaxIterations(this.m_maxKMeansIterationsFinal);
        finalRun.buildClusterer(this.m_data, this.m_distFunc);
        this.m_builtClusterer = finalRun;
        this.m_numberOfClusters = this.m_builtClusterer.getClusterCentroids().size();
    }

    private static HashSet<Integer> determineParetoSet(double[][] objectives) {
        double[] maxiMinScores = new double[objectives.length];
        HashSet<Integer> nonDomList = new HashSet<>();
        for (int i = 0; i < objectives.length; ++i) {
            double maxiMin = Double.NEGATIVE_INFINITY;
            double[] objI = objectives[i];
            double[] objJ;
            for (int j = 0; j < objectives.length; ++j) {
                if (i == j) continue;
                objJ = objectives[j];
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
            maxiMinScores[i] = maxiMin;
            // max > 0 then solution i is strongly dominated
            if (maxiMin < -THRESHOLD) {
                nonDomList.add(i);
            }
        }

        if (nonDomList.size() < 1) {
            int bestDomIdx = -1;
            double bestDom = Double.POSITIVE_INFINITY;
            for (int i = 0; i < maxiMinScores.length; ++i) {
                if (bestDom > maxiMinScores[i]) {
                    bestDom = maxiMinScores[i];
                    bestDomIdx = i;
                }
            }
            assert (bestDomIdx != -1);
            nonDomList.add(bestDomIdx);
        }

        return nonDomList;
    }

    private void updateUtopiaPoint(double[][] objectives) {
        // update utopia point
        for (int i = 0; i < objectives.length; ++i) {
            for (int iO = 0; iO < evaluations.length; ++iO) {
                double obj = objectives[i][iO];
                if (obj < objBestCoordinates[iO]) {
                    objBestCoordinates[iO] = obj;
                }
                if (obj > objWorstCoordinates[iO]) {
                    objWorstCoordinates[iO] = obj;
                }
            }
        }
    }

    public int pickALeader(HashSet<Integer> nonDomIndices, double[][] objectives) {
        assert (nonDomIndices.size() > 0);
        double minDist = Double.POSITIVE_INFINITY;
        double[] utopiaCoords = new double[]{0.0, 0.0};
        int i = 0;
        int leader = -1;
        for (double[] cur: objectives) {
            if (nonDomIndices.contains(i)) {
                double[] normCur = utils.Utils.normalize(cur, this.objBestCoordinates, this.objWorstCoordinates);
                double distToUtopia = utils.Utils.dist(normCur, utopiaCoords, 2.0);
                //double distToUtopia = utils.Utils.dist(cur, objBestCoordinates, 2);
                if (distToUtopia < minDist) {
                    leader = i;
                    minDist = distToUtopia;
                }
            }
            ++i;
        }
        assert (leader != -1);

        return leader;
    }

    public void measureFinalPop(double[][] data, int[] labelsTrue) throws Exception {
        int[] labelsPred;
        for (int i = 0; i < this.finalpopulation.length; ++i) {
            labelsPred = finalpopulation[i].getAssignments();
            HashMap<Integer, double[]> myCentroids = utils.Utils.centroids(data, labelsPred);
            /*for (double[] c: myCentroids.values()) {
                System.out.println(Arrays.toString(c));
            }
            System.out.println("---------");
            Instances innerCentroids = finalpopulation[i].getClusterCentroids();
            for (int j = 0; j < innerCentroids.size(); ++j) {
                System.out.println(Arrays.toString(innerCentroids.get(j).toDoubleArray()));
            }
            System.out.println();*/

            // step 4 - measure
            double ARI = adjustedRandIndex.measure(labelsTrue, labelsPred);
            double myDBWithMyCentroids = utils.Utils.dbIndexScore(myCentroids, labelsPred, data);
            double k = utils.Utils.distinctNumberOfItems(labelsPred);
            System.out.println("ARI score: " + utils.Utils.doublePrecision(ARI, 4));
            System.out.println("DB score:  " + utils.Utils.doublePrecision(myDBWithMyCentroids, 4));
            System.out.println("inner DB score: " + utils.Utils.doublePrecision(1.0D / fitness(this.finalpopulation[i]), 4));
            System.out.println("num of clusters: " + k +  " : " + finalpopulation[i].getClusterCentroids().size());
        }
    }

    private MyGenClustPlusPlus.MKMeans[] generateInitialPopulation(Instances data) throws Exception {
        /* pre-initial size is 3 * initial size = 90 */
        int maxK = 3 * this.m_initialPopulationSize / 10 + 1;
        int numberOfChromosomes = 5 * (maxK - 1) * 2;
        MyGenClustPlusPlus.MKMeans[] population = new MyGenClustPlusPlus.MKMeans[numberOfChromosomes];
        int chromosomeCount = 0;

        int i;
        int randomK;
        /* Initial Population with Probabilistic Selection */
        for(i = 2; i <= maxK; ++i) {
            /* five clusterings for each value of k */
            for(randomK = 0; randomK < 5; ++randomK) {
                MyGenClustPlusPlus.MKMeans j = new MyGenClustPlusPlus.MKMeans();
                int chromosome = -1;

                /* chromosome is built until number of clusters is at least two */
                do {
                    ++chromosome;
                    if(chromosome > 100) {
                        System.out.println("Unable to cluster this dataset using genetic algorithm, try imputing missing values first.");
                        j.setUnable(true);
                        this.m_builtClusterer = j;
                        this.m_numberOfClusters = this.m_builtClusterer.getClusterCentroids().size();
                        return null;
                    }

                    j.setSeed(this.m_rand.nextInt());
                    j.setNumClusters(i);
                    j.setPreserveInstancesOrder(true);
                    j.setMaxIterations(this.m_maxKMeansIterationsInitial);
                    j.setDontReplaceMissingValues(this.m_dontReplaceMissing);
                    j.setInitializationMethod(new SelectedTag(1, TAGS_SELECTION_MK));
                    j.buildClusterer(data, this.m_distFunc);
                } while(j.getNumClusters() < 2);

                population[chromosomeCount++] = j;
            }
        }

        for(i = 0; i < maxK - 1; ++i) {
            randomK = this.m_rand.nextInt((int)(Math.sqrt((double)data.size()) - 2.0D)) + 2;

            for(int var10 = 0; var10 < 5; ++var10) {
                MyGenClustPlusPlus.MKMeans var11 = new MyGenClustPlusPlus.MKMeans();
                var11.setSeed(this.m_rand.nextInt());
                var11.setNumClusters(randomK);
                var11.setDontReplaceMissingValues(this.m_dontReplaceMissing);
                var11.setPreserveInstancesOrder(true);
                var11.setInitializationMethod(new SelectedTag(1, TAGS_SELECTION_MK));

                /* chromosome is built until number of clusters is at least two */
                do {
                    var11.buildClusterer(data, this.m_distFunc);
                } while(var11.getNumClusters() < 2);

                population[chromosomeCount++] = var11;
            }
        }

        return population;
    }

    public int[] getLabels() throws Exception {
        return m_builtClusterer.getAssignments();
    }

    private double[] evaluate(int[] labelPred) {
        double[] result = new double[evaluations.length];
        for (int iE = 0; iE < evaluations.length; ++iE) {
            result[iE] = evaluator.evaluate(labelPred, evaluations[iE], myData, ncConstruct);
        }
        return result;
    }

    public double dbScore() {
        return 1.0D / fitness(m_bestChromosome);
    }

    private double fitness(MyGenClustPlusPlus.MKMeans chromosome) {
        /*int[] labelsPred = null;
        try {
            labelsPred = chromosome.getAssignments();
        } catch (Exception var19) {
            var19.printStackTrace();
        }

        HashMap<Integer, double[]> centroids = utils.Utils.centroids(myData, labelsPred);
        if (centroids.size() < 2) {
            return 0.0;
        }

        double dbScore = utils.Utils.dbIndexScore(centroids, labelsPred, myData);

        return 1.0D / dbScore;*/
        EuclideanDistance eu = new EuclideanDistance(this.m_data);
        Instances centroids = chromosome.getClusterCentroids();

        double[] Si = new double[centroids.numInstances()];
        int[] Ti = new int[centroids.numInstances()];
        int[] clustSize = new int[centroids.numInstances()];
        int numClust = centroids.numInstances();
        if(numClust == 1) {
            return 0.0D;
        } else {
            int[] assignments = null;

            try {
                assignments = chromosome.getAssignments();
            } catch (Exception var19) {
                var19.printStackTrace();
            }

            int DB;
            /* step 1 - computer cluster diameter */
            for(DB = 0; DB < assignments.length; ++DB) {
                try {
                    /* e is cluster id */
                    int e = chromosome.clusterInstance(this.m_data.get(DB));
                    ++clustSize[e];
                    /* Si is cluster diamater, not averaged yet */
                    Si[e] += Math.abs(eu.distance(this.m_data.get(DB), centroids.get(e)));
                } catch (Exception var18) {
                    var18.printStackTrace();
                    System.exit(0);
                }
            }

            /* still step 1 - compute average  */
            for(DB = 0; DB < centroids.numInstances(); ++DB) {
                Ti[DB] = clustSize[DB];
                if(clustSize[DB] == 0) {
                    return 0.0D;
                }

                Si[DB] /= (double)Ti[DB];
            }

            double var20 = 0.0D;

            /* step 2 - compute Rij for each possible pair of clusters */
            for(int i = 0; i < centroids.numInstances(); ++i) {
                if(clustSize[i] != 0) {
                    double max = -1.0D / 0.0;
                    int maxIndex = -2147483648;

                    /* find max Rij for cluster */
                    for(int j = 0; j < centroids.numInstances(); ++j) {
                        if(i != j) {
                            double Rij = Si[i] + Si[j] / Math.abs(eu.distance(centroids.get(i), centroids.get(j)));
                            if(Rij > max) {
                                max = Rij;
                            }
                        }
                    }

                    assert (max != -1.0D / 0.0);
                    var20 += max;
                }
            }

            var20 /= (double)numClust;
            return 1.0D / var20;
        }
    }

    private MyGenClustPlusPlus.MKMeans[] probabilisticSelection(MyGenClustPlusPlus.MKMeans[] population) {
        MyGenClustPlusPlus.MKMeans[] selectedPopulation = new MyGenClustPlusPlus.MKMeans[this.m_initialPopulationSize];
        int currentSelections = 0;
        double[] fitnessArray = new double[population.length];
        boolean[] usedChromosome = new boolean[population.length];
        double[] TkArray = new double[population.length / 5];
        int[] kArray = new int[population.length / 5];
        double sumTk = 0.0D;

        /* compute average fitness value for each value of k */
        for(int p = 0; p < population.length; p += 5) {
            kArray[p / 5] = population[p].getNumClusters();
            double Tk = 0.0D;

            for(int j = 0; j < 5; ++j) {
                double i = this.fitness(population[p + j]);
                fitnessArray[p + j] = i;
                Tk += i;
            }

            sumTk += Tk;
            TkArray[p / 5] = Tk;
        }

        /* choose probabilistically next chromosome to include in initial population */
        while(currentSelections < selectedPopulation.length) {
            double var21 = this.m_rand.nextDouble();
            double cumulativeProbability = 0.0D;

            /* iterate through each value of k,
             * roulette-wheel selection, the higher average fitness the higher chance to be selected */
            for(int var22 = 0; var22 < TkArray.length; ++var22) {
                cumulativeProbability += TkArray[var22] / sumTk;
                if(var21 <= cumulativeProbability) {
                    double max = -1.0D / 0.0;
                    int maxIndex = 2147483647;

                    /* 5 clusterings belong to each of the values k
                    *  choose the next one with the best fitness function */
                    for(int selected = 0; selected < 5; ++selected) {
                        if(fitnessArray[var22 + selected] > max && !usedChromosome[var22 + selected]) {
                            max = fitnessArray[var22 + selected];
                            maxIndex = var22 + selected;
                        }
                    }

                    MyGenClustPlusPlus.MKMeans var23;
                    if(maxIndex == 2147483647) {
                        /* if clusterings corresponding to given k are chosen more than five times
                         * then we create a new chromosome by running MK-Means/MK-Means++ once more
                         * with input k for the number of clusters. */
                        try {
                            MyGenClustPlusPlus.MKMeans ex = new MyGenClustPlusPlus.MKMeans();
                            ex.setSeed(this.m_rand.nextInt());
                            ex.setNumClusters(kArray[var22]);
                            ex.setPreserveInstancesOrder(true);
                            ex.setDontReplaceMissingValues(this.m_dontReplaceMissing);
                            ex.buildClusterer(this.m_data, this.m_distFunc);
                            var23 = ex;
                        } catch (Exception var20) {
                            var20.printStackTrace();
                            System.out.println("failed in probabilistic selection to generate new buildClusterer" +
                                    "when given k is chosen more than five times");
                            return null;
                        }
                    } else {
                        /* otherwise include in initial population and indicate that it's already chosen */
                        usedChromosome[maxIndex] = true;
                        var23 = population[maxIndex];
                    }

                    selectedPopulation[currentSelections++] = var23;
                    break;
                }
            }
        }

        return selectedPopulation;
    }

    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation result = new TechnicalInformation(Type.ARTICLE);
        result.setValue(Field.TITLE, "Combining K-Means and a Genetic Algorithm through a Novel Arrangement of Genetic Operators for High Quality Clustering");
        result.setValue(Field.AUTHOR, "Islam, M. Z., Estivill-Castro, V., Rahman, M. A. and Bossomaier, T.");
        result.setValue(Field.YEAR, "2018");
        result.setValue(Field.JOURNAL, "Expert Systems with Applications");
        result.setValue(Field.VOLUME, "91");
        result.setValue(Field.PAGES, "402-417");
        return result;
    }

    private MyGenClustPlusPlus.MKMeans[] crossover(MyGenClustPlusPlus.MKMeans[] selectedPopulation) throws Exception {
        Instances[] offspring = new Instances[selectedPopulation.length];
        int offspringCounter = 0;
        MyGenClustPlusPlus.FitnessContainer[] sorted = new MyGenClustPlusPlus.FitnessContainer[selectedPopulation.length];
        double fitnessSum = 0.0D;

        for(int offspringMK = 0; offspringMK < sorted.length; ++offspringMK) {
            double i = this.fitness(selectedPopulation[offspringMK]);
            fitnessSum += i;
            sorted[offspringMK] = new MyGenClustPlusPlus.FitnessContainer(i, selectedPopulation[offspringMK]);
        }

        Arrays.sort(sorted, Collections.reverseOrder());

        while(sorted.length > 0) {
            MyGenClustPlusPlus.MKMeans var27 = sorted[0].clustering;
            fitnessSum -= sorted[0].fitness;
            sorted[0] = null;
            MyGenClustPlusPlus.MKMeans var29 = sorted[sorted.length - 1].clustering;
            double toRemove = this.m_rand.nextDouble();
            double k = 0.0D;

            for(int val = 1; val < sorted.length; ++val) {
                k += sorted[val].fitness / fitnessSum;
                if(toRemove <= k) {
                    var29 = sorted[val].clustering;
                    fitnessSum -= sorted[val].fitness;
                    sorted[val] = null;
                    break;
                }
            }

            Instances var33 = var27.getClusterCentroids();
            Instances parentTwoCentroids = var29.getClusterCentroids();
            Instances target = new Instances(var33, 0);

            int refRandom;
            int refFirstHalf;
            double refSecondHalf;
            int targetFirstHalf;
            double targetSecondHalf;
            for(refRandom = 0; refRandom < var33.numInstances() && parentTwoCentroids.numInstances() > 0; ++refRandom) {
                refFirstHalf = 2147483647;
                refSecondHalf = 1.7976931348623157E308D;

                for(targetFirstHalf = 0; targetFirstHalf < parentTwoCentroids.numInstances(); ++targetFirstHalf) {
                    targetSecondHalf = Math.abs(this.m_distFunc.distance(var33.get(refRandom), parentTwoCentroids.get(targetFirstHalf)));
                    if(targetSecondHalf == 1.0D / 0.0) {
                        this.m_distFunc.distance(var33.get(refRandom), parentTwoCentroids.get(targetFirstHalf));
                    }

                    if(targetSecondHalf < refSecondHalf) {
                        refFirstHalf = targetFirstHalf;
                        refSecondHalf = targetSecondHalf;
                    }
                }

                if(refFirstHalf != 2147483647) {
                    target.add(parentTwoCentroids.get(refFirstHalf));
                    parentTwoCentroids.remove(refFirstHalf);
                }
            }

            if(parentTwoCentroids.numInstances() > 0) {
                for(refRandom = 0; refRandom < parentTwoCentroids.numInstances(); ++refRandom) {
                    refFirstHalf = 2147483647;
                    refSecondHalf = 1.7976931348623157E308D;

                    for(targetFirstHalf = 0; targetFirstHalf < target.numInstances(); ++targetFirstHalf) {
                        targetSecondHalf = Math.abs(this.m_distFunc.distance(parentTwoCentroids.get(refRandom), target.get(targetFirstHalf)));
                        if(targetSecondHalf < refSecondHalf) {
                            refSecondHalf = targetSecondHalf;
                            refFirstHalf = targetFirstHalf;
                        }
                    }

                    target.add(refFirstHalf + 1, parentTwoCentroids.get(refRandom));
                }
            }

            refRandom = var33.numInstances() > 1?this.m_rand.nextInt(var33.numInstances() - 1) + 1:0;
            Instances var35 = new Instances(var33, 0);
            Instances var36 = new Instances(var33, 0);

            int targetRandom;
            for(targetRandom = 0; targetRandom < var33.numInstances(); ++targetRandom) {
                if(targetRandom < refRandom) {
                    var35.add(var33.get(targetRandom));
                } else {
                    var36.add(var33.get(targetRandom));
                }
            }

            targetRandom = target.numInstances() > 1?this.m_rand.nextInt(target.numInstances() - 1) + 1:0;
            Instances var37 = new Instances(target, 0);
            Instances var38 = new Instances(target, 0);

            for(int tmp = 0; tmp < target.numInstances(); ++tmp) {
                if(tmp < targetRandom) {
                    var37.add(target.get(tmp));
                } else {
                    var38.add(target.get(tmp));
                }
            }

            var35.addAll(var38);
            var37.addAll(var36);
            offspring[offspringCounter++] = var35;
            offspring[offspringCounter++] = var37;
            MyGenClustPlusPlus.FitnessContainer[] var39 = new MyGenClustPlusPlus.FitnessContainer[sorted.length - 2];
            boolean hitSkip = false;

            for(int i1 = 0; i1 < var39.length; ++i1) {
                int ind = i1 + 1;
                if(hitSkip || sorted[ind] == null) {
                    ind = i1 + 2;
                    hitSkip = true;
                }

                var39[i1] = sorted[ind];
            }

            sorted = var39;
        }

        MyGenClustPlusPlus.MKMeans[] var28 = new MyGenClustPlusPlus.MKMeans[offspring.length];

        for(int var30 = 0; var30 < offspring.length; ++var30) {
            new ArrayList();

            for(int e = 0; e < offspring[var30].numInstances(); ++e) {
                for(int var32 = e + 1; var32 < offspring[var30].numInstances(); ++var32) {
                    if(Math.abs(this.m_distFunc.distance(offspring[var30].get(e), offspring[var30].get(var32))) <= this.m_duplicateThreshold) {
                        if(offspring[var30].numInstances() > 2) {
                            offspring[var30].remove(var32);
                        } else {
                            while(Math.abs(this.m_distFunc.distance(offspring[var30].get(e), offspring[var30].get(var32))) <= this.m_duplicateThreshold) {
                                int attribute = this.m_rand.nextInt(offspring[var30].numAttributes());

                                double var34;
                                do {
                                    if(offspring[var30].attribute(attribute).isNumeric()) {
                                        for(var34 = this.m_data.attributeStats(attribute).numericStats.min + this.m_rand.nextDouble() * (this.m_data.attributeStats(attribute).numericStats.max - this.m_data.attributeStats(attribute).numericStats.min + 1.0D); var34 == 1.0D / 0.0; var34 = this.m_data.attributeStats(attribute).numericStats.min + this.m_rand.nextDouble() * (this.m_data.attributeStats(attribute).numericStats.max - this.m_data.attributeStats(attribute).numericStats.min + 1.0D)) {
                                            ;
                                        }
                                    } else {
                                        var34 = (double)this.m_rand.nextInt(offspring[var30].attributeStats(attribute).nominalCounts.length);
                                    }
                                } while(var34 == offspring[var30].get(var32).value(attribute));

                                offspring[var30].get(var32).setValue(attribute, var34);
                            }
                        }
                    }
                }
            }

            try {
                MyGenClustPlusPlus.MKMeans var31 = new MyGenClustPlusPlus.MKMeans();
                var31.setSeed(this.m_rand.nextInt());
                var31.setInitializationMethod(new SelectedTag(4, TAGS_SELECTION_MK));
                var31.setInitial(offspring[var30]);
                var31.setDontReplaceMissingValues(this.m_dontReplaceMissing);
                var31.setPreserveInstancesOrder(true);
                var31.setMaxIterations(1);
                var31.buildClusterer(this.m_data, this.m_distFunc);
                var28[var30] = var31;
            } catch (Exception var26) {
                var26.printStackTrace();
            }
        }

        double[][] tmpObjs = new double[var28.length][];
        for (int i = 0; i < var28.length; ++i) {
            tmpObjs[i] = evaluate(var28[i].getAssignments());
        }
        updateUtopiaPoint(tmpObjs);

        return var28;
    }

    private MyGenClustPlusPlus.MKMeans[] probabilisticCloning(MyGenClustPlusPlus.MKMeans[] selectedPopulation) {
        MyGenClustPlusPlus.MKMeans[] newPopulation = new MyGenClustPlusPlus.MKMeans[selectedPopulation.length];
        MyGenClustPlusPlus.FitnessContainer[] fArray = new MyGenClustPlusPlus.FitnessContainer[selectedPopulation.length];
        double fitnessSum = 0.0D;

        int newCount;
        double i;
        for(newCount = 0; newCount < selectedPopulation.length; ++newCount) {
            i = this.fitness(selectedPopulation[newCount]);
            fitnessSum += i;
            fArray[newCount] = new MyGenClustPlusPlus.FitnessContainer(i, selectedPopulation[newCount]);
        }

        newCount = 0;

        while(true) {
            while(newCount < selectedPopulation.length) {
                i = this.m_rand.nextDouble();
                double cumulativeProbability = 0.0D;

                for(int i1 = 0; i1 < fArray.length; ++i1) {
                    cumulativeProbability += fArray[i1].fitness / fitnessSum;
                    if(i <= cumulativeProbability) {
                        newPopulation[newCount++] = fArray[i1].clustering;
                        break;
                    }
                }
            }

            for(int var12 = 0; var12 < newPopulation.length; ++var12) {
                newPopulation[var12] = this.mutate(newPopulation[var12].getClusterCentroids());
            }

            return newPopulation;
        }
    }

    private MyGenClustPlusPlus.MKMeans[] mutation(MyGenClustPlusPlus.MKMeans[] crossoverPopulation) throws Exception {
        MyGenClustPlusPlus.FitnessContainer[] fArray = new MyGenClustPlusPlus.FitnessContainer[crossoverPopulation.length];
        double fitnessAvg = 0.0D;
        double fitnessMax = 4.9E-324D;

        int i;
        double prob;
        for(i = 0; i < crossoverPopulation.length; ++i) {
            prob = this.fitness(crossoverPopulation[i]);
            fitnessAvg += prob;
            fArray[i] = new MyGenClustPlusPlus.FitnessContainer(prob, crossoverPopulation[i]);
            if(prob > fitnessMax) {
                fitnessMax = prob;
            }
        }

        fitnessAvg /= (double)crossoverPopulation.length;

        for(i = 0; i < fArray.length; ++i) {
            prob = 0.0D;
            if(fArray[i].fitness > fitnessAvg) {
                prob = (fitnessMax - fArray[i].fitness) / (2.0D * (fitnessMax - fitnessAvg));
            } else {
                prob = 0.5D;
            }

            if(this.m_rand.nextDouble() <= prob) {
                Instances centroids = fArray[i].clustering.getClusterCentroids();
                crossoverPopulation[i] = this.mutate(centroids);
            }
        }

        double[][] tmpObjs = new double[crossoverPopulation.length][];
        for (i = 0; i < crossoverPopulation.length; ++i) {
            tmpObjs[i] = evaluate(crossoverPopulation[i].getAssignments());
        }
        updateUtopiaPoint(tmpObjs);

        return crossoverPopulation;
    }

    private MyGenClustPlusPlus.MKMeans mutate(Instances centroids) {
        int ex;
        int k;
        for(ex = 0; ex < centroids.numInstances(); ++ex) {
            k = this.m_rand.nextInt(centroids.numAttributes());

            double attribute;
            do {
                if(centroids.attribute(k).isNumeric()) {
                    for(attribute = this.m_data.attributeStats(k).numericStats.min + this.m_rand.nextDouble() * (this.m_data.attributeStats(k).numericStats.max - this.m_data.attributeStats(k).numericStats.min + 1.0D); attribute == 1.0D / 0.0; attribute = this.m_data.attributeStats(k).numericStats.min + this.m_rand.nextDouble() * (this.m_data.attributeStats(k).numericStats.max - this.m_data.attributeStats(k).numericStats.min + 1.0D)) {
                        ;
                    }
                } else {
                    attribute = (double)this.m_rand.nextInt(centroids.attributeStats(k).nominalCounts.length);
                }
            } while(attribute == centroids.get(ex).value(k));

            centroids.get(ex).setValue(k, attribute);
        }

        for(ex = 0; ex < centroids.numInstances(); ++ex) {
            for(k = ex + 1; k < centroids.numInstances(); ++k) {
                if(Math.abs(this.m_distFunc.distance(centroids.get(ex), centroids.get(k))) <= this.m_duplicateThreshold) {
                    if(centroids.numInstances() > 2) {
                        centroids.remove(k);
                    } else {
                        while(Math.abs(this.m_distFunc.distance(centroids.get(ex), centroids.get(k))) <= this.m_duplicateThreshold) {
                            int var9 = this.m_rand.nextInt(centroids.numAttributes());

                            double val;
                            do {
                                if(centroids.attribute(var9).isNumeric()) {
                                    for(val = this.m_data.attributeStats(var9).numericStats.min + this.m_rand.nextDouble() * (this.m_data.attributeStats(var9).numericStats.max - this.m_data.attributeStats(var9).numericStats.min + 1.0D); val == 1.0D / 0.0; val = this.m_data.attributeStats(var9).numericStats.min + this.m_rand.nextDouble() * (this.m_data.attributeStats(var9).numericStats.max - this.m_data.attributeStats(var9).numericStats.min + 1.0D)) {
                                        ;
                                    }
                                } else {
                                    val = (double)this.m_rand.nextInt(centroids.attributeStats(var9).nominalCounts.length);
                                }
                            } while(val == centroids.get(k).value(var9));

                            centroids.get(k).setValue(var9, val);
                        }
                    }
                }
            }
        }

        try {
            MyGenClustPlusPlus.MKMeans var8 = new MyGenClustPlusPlus.MKMeans();
            var8.setSeed(this.m_rand.nextInt());
            var8.setInitializationMethod(new SelectedTag(4, TAGS_SELECTION_MK));
            var8.setInitial(centroids);
            var8.setMaxIterations(1);
            var8.setDontReplaceMissingValues(this.m_dontReplaceMissing);
            var8.setPreserveInstancesOrder(true);
            var8.buildClusterer(this.m_data, this.m_distFunc);
            return var8;
        } catch (Exception var7) {
            var7.printStackTrace();
            return null;
        }
    }

    private MyGenClustPlusPlus.MKMeans[] elitism(MyGenClustPlusPlus.MKMeans[] population) throws Exception {
        double worstFitness = 1.7976931348623157E308D;
        int worstIndex = 2147483647;
        double newBestFitness = 4.9E-324D;
        int newBestIndex = 2147483647;

        for(int finalRun = 0; finalRun < population.length; ++finalRun) {
            double f = this.fitness(population[finalRun]);
            if(f < worstFitness) {
                worstFitness = f;
                worstIndex = finalRun;
            }

            if(f > newBestFitness) {
                newBestFitness = f;
                newBestIndex = finalRun;
            }
        }

        if(this.m_bestFitness > worstFitness) {
            population[worstIndex] = new MyGenClustPlusPlus.MKMeans(this.m_bestChromosome);
        }

        if(newBestFitness > this.m_bestFitness) {
            if(newBestFitness == 1.0D / 0.0) {
                MyGenClustPlusPlus.MKMeans var11 = new MyGenClustPlusPlus.MKMeans();
                var11.setSeed(this.m_rand.nextInt());
                var11.setInitializationMethod(new SelectedTag(4, TAGS_SELECTION_MK));
                var11.setInitial(population[newBestIndex].getClusterCentroids());
                var11.setDontReplaceMissingValues(this.m_dontReplaceMissing);
                var11.setPreserveInstancesOrder(true);
                var11.setMaxIterations(this.m_maxKMeansIterationsFinal);
                var11.buildClusterer(this.m_data, this.m_distFunc);
                this.m_builtClusterer = var11;
                this.m_numberOfClusters = this.m_builtClusterer.getClusterCentroids().size();
            }

            this.m_bestChromosome = new MyGenClustPlusPlus.MKMeans(population[newBestIndex]);
            this.m_bestFitness = newBestFitness;
        }

        return population;
    }

    public int clusterInstance(Instance instance) throws Exception {
        return this.m_builtClusterer.clusterInstance(instance);
    }

    public String toString() {
        return this.m_builtClusterer == null?"No clusterer built yet!":this.m_builtClusterer.toString();
    }

    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();
        result.enable(Capability.NO_CLASS);
        result.enable(Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capability.MISSING_VALUES);
        return result;
    }

    public String globalInfo() {
        return "Class implementing algorithm described in \"Combining K-Means and a Genetic Algorithm through a Novel Arrangement of Genetic Operators for High Quality Clustering\".\n\nDifferences to the original algorithm are: \n1. No use of VICUS similarity measure - standard ManhattanDistanceclass is used instead.\n2. Uses the basic missing value handling from SimpleKMeans.\n3. If an operation generates a chromosome where all records are assigned to a single cluster, chromosome will be mutated until at least 2 clusters are found.\n4. The starting generation for the chromosome selection operation is now modifiable, where in the original paperit was set to 11. The default is now after generation 50 (with the default number of generations being 60).\n\nFor more information see:" + this.getTechnicalInformation().toString();
    }

    public static void main(String[] args) throws Exception {
        //runClusterer(new MyGenClustPlusPlus(), args);
    }

    public void setOptions(String[] options) throws Exception {
        this.m_dontReplaceMissing = Utils.getFlag("M", options);
        String optionString = Utils.getOption("I", options);
        if(optionString.length() != 0) {
            this.setNumGenerations(Integer.parseInt(optionString));
        }

        optionString = Utils.getOption("P", options);
        if(optionString.length() != 0) {
            this.setInitialPopulationSize(Integer.parseInt(optionString));
        }

        optionString = Utils.getOption("N", options);
        if(optionString.length() != 0) {
            this.setMaxKMeansIterationsInitial(Integer.parseInt(optionString));
        }

        optionString = Utils.getOption("Q", options);
        if(optionString.length() != 0) {
            this.setMaxKMeansIterationsQuick(Integer.parseInt(optionString));
        }

        optionString = Utils.getOption("F", options);
        if(optionString.length() != 0) {
            this.setMaxKMeansIterationsFinal(Integer.parseInt(optionString));
        }

        optionString = Utils.getOption("C", options);
        if(optionString.length() != 0) {
            this.setStartChromosomeSelectionGeneration(Integer.parseInt(optionString));
        }

        optionString = Utils.getOption("D", options);
        if(optionString.length() != 0) {
            this.setDuplicateThreshold(Double.parseDouble(optionString));
        }

        optionString = Utils.getOption("S", options);
        if(optionString.length() != 0) {
            this.setSeed(Integer.parseInt(optionString));
        }

    }

    public String[] getOptions() {
        Vector result = new Vector();
        result.add("-I");
        result.add("" + this.getNumGenerations());
        result.add("-P");
        result.add("" + this.getInitialPopulationSize());
        result.add("-N");
        result.add("" + this.getMaxKMeansIterationsInitial());
        result.add("-Q");
        result.add("" + this.getMaxKMeansIterationsQuick());
        result.add("-F");
        result.add("" + this.getMaxKMeansIterationsFinal());
        result.add("-D");
        result.add("" + this.getDuplicateThreshold());
        result.add("-C");
        result.add("" + this.getStartChromosomeSelectionGeneration());
        result.add("-S");
        result.add("" + this.getSeed());
        if(this.m_dontReplaceMissing) {
            result.add("-M");
        }

        return (String[])result.toArray(new String[result.size()]);
    }

    public void setNumGenerations(int n) throws Exception {
        if(n <= 0) {
            throw new Exception("Number of generations must be > 0");
        } else {
            this.m_numberOfGenerations = n;
        }
    }

    public void setInitialPopulationSize(int n) throws Exception {
        if(n <= 2) {
            throw new Exception("Initial population must be > 2");
        } else if(n % 2 != 0) {
            throw new Exception("Initial population must be divisible by 2");
        } else {
            this.m_initialPopulationSize = n;
        }
    }

    public void setMaxKMeansIterationsInitial(int n) throws Exception {
        if(n <= 0) {
            throw new Exception("Max initial k means iterations must be positive.");
        } else {
            this.m_maxKMeansIterationsInitial = n;
        }
    }

    public void setMaxKMeansIterationsQuick(int n) throws Exception {
        if(n <= 0) {
            throw new Exception("Max quick k means iterations must be positive.");
        } else {
            this.m_maxKMeansIterationsQuick = n;
        }
    }

    public void setMaxKMeansIterationsFinal(int n) throws Exception {
        if(n <= 0) {
            throw new Exception("Max final k means iterations must be positive.");
        } else {
            this.m_maxKMeansIterationsFinal = n;
        }
    }

    public void setDuplicateThreshold(double d) throws Exception {
        if(d >= 0.0D && d <= 1.0D) {
            this.m_duplicateThreshold = d;
        } else {
            throw new Exception("Duplicate threshold must be between 0 and 1");
        }
    }

    public void setDontReplaceMissing(boolean m_dontReplaceMissing) {
        this.m_dontReplaceMissing = m_dontReplaceMissing;
    }

    public void setStartChromosomeSelectionGeneration(int s) throws Exception {
        if(s < 1) {
            throw new Exception("Chromosome Selection start generation must be greater than 0");
        } else {
            this.m_startChromosomeSelectionGeneration = s;
        }
    }

    public int getNumGenerations() {
        return this.m_numberOfGenerations;
    }

    public int getStartChromosomeSelectionGeneration() {
        return this.m_startChromosomeSelectionGeneration;
    }

    public int getInitialPopulationSize() {
        return this.m_initialPopulationSize;
    }

    public int getMaxKMeansIterationsInitial() {
        return this.m_maxKMeansIterationsInitial;
    }

    public int getMaxKMeansIterationsQuick() {
        return this.m_maxKMeansIterationsQuick;
    }

    public int getMaxKMeansIterationsFinal() {
        return this.m_maxKMeansIterationsFinal;
    }

    public double getDuplicateThreshold() {
        return this.m_duplicateThreshold;
    }

    public boolean getDontReplaceMissing() {
        return this.m_dontReplaceMissing;
    }

    public String numGenerationsTipText() {
        return "set number of genetic algorithm generations";
    }

    public String initialPopulationSizeTipText() {
        return "set initial population size for genetic algorithm";
    }

    public String maxKMeansIterationsInitialTipText() {
        return "set the max iterations for initial k-means";
    }

    public String maxKMeansIterationsQuickTipText() {
        return "set the max iterations for quick k-means";
    }

    public String maxKMeansIterationsFinalTipText() {
        return "set the max iterations for final k-means";
    }

    public String duplicateThresholdTipText() {
        return "set the duplicate threshold";
    }

    public String dontReplaceMissingTipText() {
        return "Replace missing values globally with mean/mode.";
    }

    public String startChromosomeSelectionGenerationTipText() {
        return "Generation after which to start using the chromosome selection operation.";
    }

    class MKMeans extends SimpleKMeans {
        private static final long serialVersionUID = -2890080669539418269L;
        protected boolean m_supplied = false;
        protected boolean m_unable = false;

        public MKMeans() {
        }

        public MKMeans(MyGenClustPlusPlus.MKMeans mk) throws Exception {
            this.m_ReplaceMissingFilter = new ReplaceMissingValues();
            this.m_ReplaceMissingFilter.setInputFormat(mk.m_ReplaceMissingFilter.getCopyOfInputFormat());
            this.setPreserveInstancesOrder(true);
            this.m_ClusterCentroids = new Instances(mk.getClusterCentroids());
            this.m_NumClusters = this.m_ClusterCentroids.numInstances();
            this.m_Assignments = new int[mk.getAssignments().length];
            this.m_DistanceFunction = mk.m_DistanceFunction;
            this.m_dontReplaceMissing = mk.getDontReplaceMissingValues();
            System.arraycopy(mk.getAssignments(), 0, this.m_Assignments, 0, mk.getAssignments().length);
        }

        public void buildClusterer(Instances data) throws Exception {
            this.m_DistanceFunction = new ManhattanDistance(data);
            super.buildClusterer(data);
        }

        public void buildClusterer(Instances data, ManhattanDistance distance) throws Exception {
            this.m_DistanceFunction = distance;
            this.m_canopyClusters = null;
            this.getCapabilities().testWithFail(data);
            this.m_Iterations = 0;
            this.m_ReplaceMissingFilter = new ReplaceMissingValues();
            Instances instances = new Instances(data);
            instances.setClassIndex(-1);
            if(!this.m_dontReplaceMissing) {
                this.m_ReplaceMissingFilter.setInputFormat(instances);
            }

            this.m_ClusterNominalCounts = new double[this.m_NumClusters][instances.numAttributes()][];
            this.m_ClusterMissingCounts = new double[this.m_NumClusters][instances.numAttributes()];
            if(this.m_displayStdDevs) {
                this.m_FullStdDevs = instances.variances();
            }

            this.m_FullMeansOrMediansOrModes = this.moveCentroid(0, instances, true, false);
            this.m_FullMissingCounts = this.m_ClusterMissingCounts[0];
            this.m_FullNominalCounts = this.m_ClusterNominalCounts[0];
            double sumOfWeights = instances.sumOfWeights();

            for(int clusterAssignments = 0; clusterAssignments < instances.numAttributes(); ++clusterAssignments) {
                if(instances.attribute(clusterAssignments).isNumeric()) {
                    if(this.m_displayStdDevs) {
                        this.m_FullStdDevs[clusterAssignments] = Math.sqrt(this.m_FullStdDevs[clusterAssignments]);
                    }

                    if(this.m_FullMissingCounts[clusterAssignments] == sumOfWeights) {
                        this.m_FullMeansOrMediansOrModes[clusterAssignments] = 0.0D / 0.0;
                    }
                } else if(this.m_FullMissingCounts[clusterAssignments] > this.m_FullNominalCounts[clusterAssignments][Utils.maxIndex(this.m_FullNominalCounts[clusterAssignments])]) {
                    this.m_FullMeansOrMediansOrModes[clusterAssignments] = -1.0D;
                }
            }

            this.m_ClusterCentroids = new Instances(instances, this.m_NumClusters);
            int[] var18 = new int[instances.numInstances()];
            if(this.m_PreserveOrder) {
                this.m_Assignments = var18;
            }

            this.m_DistanceFunction.setInstances(instances);
            Random RandomO = new Random((long)this.getSeed());
            HashMap initC = new HashMap();
            DecisionTableHashKey hk = null;
            Instances initInstances = null;
            if(this.m_PreserveOrder) {
                initInstances = new Instances(instances);
            } else {
                initInstances = instances;
            }

            if(this.m_speedUpDistanceCompWithCanopies) {
                this.m_canopyClusters = new Canopy();
                this.m_canopyClusters.setNumClusters(this.m_NumClusters);
                this.m_canopyClusters.setSeed(this.getSeed());
                this.m_canopyClusters.setT2(this.getCanopyT2());
                this.m_canopyClusters.setT1(this.getCanopyT1());
                this.m_canopyClusters.setMaxNumCandidateCanopiesToHoldInMemory(this.getCanopyMaxNumCanopiesToHoldInMemory());
                this.m_canopyClusters.setPeriodicPruningRate(this.getCanopyPeriodicPruningRate());
                this.m_canopyClusters.setMinimumCanopyDensity(this.getCanopyMinimumCanopyDensity());
                this.m_canopyClusters.setDebug(this.getDebug());
                this.m_canopyClusters.buildClusterer(initInstances);
                this.m_centroidCanopyAssignments = new ArrayList();
                this.m_dataPointCanopyAssignments = new ArrayList();
            }

            int i;
            if(this.m_initializationMethod == 1) {
                this.kMeansPlusPlusInit(initInstances);
                this.m_initialStartPoints = new Instances(this.m_ClusterCentroids);
            } else if(this.m_initializationMethod == 2) {
                this.canopyInit(initInstances);
                this.m_initialStartPoints = new Instances(this.m_canopyClusters.getCanopies());
            } else if(this.m_initializationMethod == 3) {
                this.farthestFirstInit(initInstances);
                this.m_initialStartPoints = new Instances(this.m_ClusterCentroids);
            } else if(this.m_initializationMethod != 4 && !this.m_supplied) {
                for(i = initInstances.numInstances() - 1; i >= 0; --i) {
                    int instIndex = RandomO.nextInt(i + 1);
                    hk = new DecisionTableHashKey(initInstances.instance(instIndex), initInstances.numAttributes(), true);
                    if(!initC.containsKey(hk)) {
                        this.m_ClusterCentroids.add(initInstances.instance(instIndex));
                        initC.put(hk, null);
                    }

                    initInstances.swap(i, instIndex);
                    if(this.m_ClusterCentroids.numInstances() == this.m_NumClusters) {
                        break;
                    }
                }

                this.m_initialStartPoints = new Instances(this.m_ClusterCentroids);
            } else {
                /* hill-climber */
                this.m_ClusterCentroids = this.m_initialStartPoints;
                this.m_NumClusters = this.m_initialStartPoints.numInstances();
                if(!this.m_supplied) {
                    throw new Exception("Please supply a set of initial centroids.");
                }
            }

            if(this.m_speedUpDistanceCompWithCanopies) {
                for(i = 0; i < instances.numInstances(); ++i) {
                    this.m_dataPointCanopyAssignments.add(this.m_canopyClusters.assignCanopies(instances.instance(i)));
                }
            }

            this.m_NumClusters = this.m_ClusterCentroids.numInstances();
            initInstances = null;
            boolean converged = false;
            Instances[] tempI = new Instances[this.m_NumClusters];
            this.m_squaredErrors = new double[this.m_NumClusters];
            this.m_ClusterNominalCounts = new double[this.m_NumClusters][instances.numAttributes()][0];
            this.m_ClusterMissingCounts = new double[this.m_NumClusters][instances.numAttributes()];
            this.startExecutorPool();

            int j;
            Instance var19;
            while(!converged) {
                if(this.m_speedUpDistanceCompWithCanopies) {
                    this.m_centroidCanopyAssignments.clear();

                    for(int vals2 = 0; vals2 < this.m_ClusterCentroids.numInstances(); ++vals2) {
                        this.m_centroidCanopyAssignments.add(this.m_canopyClusters.assignCanopies(this.m_ClusterCentroids.instance(vals2)));
                    }
                }

                int emptyClusterCount = 0;
                ++this.m_Iterations;
                converged = true;
                if(this.m_executionSlots > 1 && instances.numInstances() >= 2 * this.m_executionSlots) {
                    converged = this.launchAssignToClusters(instances, var18);
                } else {
                    for(i = 0; i < instances.numInstances(); ++i) {
                        var19 = instances.instance(i);
                        j = this.clusterProcessedInstance(var19, false, false, this.m_speedUpDistanceCompWithCanopies?(long[])this.m_dataPointCanopyAssignments.get(i):null);
                        if(j != var18[i]) {
                            converged = false;
                        }

                        var18[i] = j;
                    }
                }

                this.m_ClusterCentroids = new Instances(instances, this.m_NumClusters);

                for(i = 0; i < this.m_NumClusters; ++i) {
                    tempI[i] = new Instances(instances, 0);
                }

                for(i = 0; i < instances.numInstances(); ++i) {
                    tempI[var18[i]].add(instances.instance(i));
                }

                if(this.m_initializationMethod == 4 && this.m_MaxIterations == 1) {
                    this.m_ClusterCentroids = this.m_initialStartPoints;
                } else if(this.m_executionSlots > 1 && instances.numInstances() >= 2 * this.m_executionSlots) {
                    this.launchMoveCentroids(tempI);
                } else {
                    for(i = 0; i < this.m_NumClusters; ++i) {
                        if(tempI[i].numInstances() == 0) {
                            ++emptyClusterCount;
                        } else {
                            this.moveCentroid(i, tempI[i], true, true);
                        }
                    }
                }

                if(this.m_Iterations == this.m_MaxIterations) {
                    converged = true;
                }

                if(!converged) {
                    this.m_ClusterNominalCounts = new double[this.m_NumClusters][instances.numAttributes()][0];
                }
            }

            for(i = 0; i < instances.numInstances(); ++i) {
                var19 = instances.instance(i);
                j = this.clusterProcessedInstance(var19, false, true, this.m_speedUpDistanceCompWithCanopies?(long[])this.m_dataPointCanopyAssignments.get(i):null);
                var18[i] = j;
            }

            this.m_Assignments = var18;
            if(!this.m_FastDistanceCalc) {
                for(i = 0; i < instances.numInstances(); ++i) {
                    this.clusterProcessedInstance(instances.instance(i), true, false, (long[])null);
                }
            }

            if(this.m_displayStdDevs) {
                this.m_ClusterStdDevs = new Instances(instances, this.m_NumClusters);
            }

            this.m_ClusterSizes = new double[this.m_NumClusters];

            for(i = 0; i < this.m_NumClusters; ++i) {
                if(this.m_displayStdDevs) {
                    double[] var20 = tempI[i].variances();

                    for(j = 0; j < instances.numAttributes(); ++j) {
                        if(instances.attribute(j).isNumeric()) {
                            var20[j] = Math.sqrt(var20[j]);
                        } else {
                            var20[j] = Utils.missingValue();
                        }
                    }

                    this.m_ClusterStdDevs.add(new DenseInstance(1.0D, var20));
                }

                this.m_ClusterSizes[i] = tempI[i].sumOfWeights();
            }

            this.m_executorPool.shutdown();
            this.m_NumClusters = this.m_ClusterCentroids.numInstances();
        }

        public void setDistanceFunction(DistanceFunction df) throws Exception {
            this.m_DistanceFunction = df;
        }

        private int clusterProcessedInstance(Instance instance, boolean updateErrors, boolean useFastDistCalc, long[] instanceCanopies) {
            double minDist = 2.147483647E9D;
            int bestCluster = 0;

            for(int i = 0; i < this.m_ClusterCentroids.numInstances(); ++i) {
                double dist;
                if(useFastDistCalc) {
                    if(this.m_speedUpDistanceCompWithCanopies && instanceCanopies != null && instanceCanopies.length > 0) {
                        try {
                            if(!Canopy.nonEmptyCanopySetIntersection((long[])this.m_centroidCanopyAssignments.get(i), instanceCanopies)) {
                                continue;
                            }
                        } catch (Exception var12) {
                            var12.printStackTrace();
                        }

                        dist = this.m_DistanceFunction.distance(instance, this.m_ClusterCentroids.instance(i), minDist);
                    } else {
                        dist = this.m_DistanceFunction.distance(instance, this.m_ClusterCentroids.instance(i), minDist);
                    }
                } else {
                    dist = this.m_DistanceFunction.distance(instance, this.m_ClusterCentroids.instance(i));
                }

                if(dist < minDist) {
                    minDist = dist;
                    bestCluster = i;
                }
            }

            if(updateErrors) {
                if(this.m_DistanceFunction instanceof EuclideanDistance) {
                    minDist *= minDist * instance.weight();
                }

                this.m_squaredErrors[bestCluster] += minDist;
            }

            return bestCluster;
        }

        public void setInitial(Instances initial) {
            this.m_supplied = true;
            this.m_NumClusters = initial.numInstances();
            this.m_initialStartPoints = initial;
        }

        public void setInitializationMethod(SelectedTag method) {
            if(method.getTags() == MyGenClustPlusPlus.TAGS_SELECTION_MK) {
                this.m_initializationMethod = method.getSelectedTag().getID();
            }

        }

        public int clusterInstance(Instance instance) throws Exception {
            Instance inst = null;
            if(!this.m_dontReplaceMissing) {
                this.m_ReplaceMissingFilter.input(instance);
                this.m_ReplaceMissingFilter.batchFinished();
                inst = this.m_ReplaceMissingFilter.output();
            } else {
                inst = instance;
            }

            return this.clusterProcessedInstance(inst, false, false, (long[])null);
        }

        public String toString() {
            if(this.m_ClusterCentroids == null) {
                return "No clusterer built yet!";
            } else {
                int maxWidth = 0;
                int maxAttWidth = 0;
                boolean containsNumeric = false;

                int plusMinus;
                int temp;
                for(plusMinus = 0; plusMinus < this.m_NumClusters; ++plusMinus) {
                    for(temp = 0; temp < this.m_ClusterCentroids.numAttributes(); ++temp) {
                        if(this.m_ClusterCentroids.attribute(temp).name().length() > maxAttWidth) {
                            maxAttWidth = this.m_ClusterCentroids.attribute(temp).name().length();
                        }

                        if(this.m_ClusterCentroids.attribute(temp).isNumeric()) {
                            containsNumeric = true;
                            double cSize = Math.log(Math.abs(this.m_ClusterCentroids.instance(plusMinus).value(temp))) / Math.log(10.0D);
                            if(cSize < 0.0D) {
                                cSize = 1.0D;
                            }

                            cSize += 6.0D;
                            if((int)cSize > maxWidth) {
                                maxWidth = (int)cSize;
                            }
                        }
                    }
                }

                String i;
                int var24;
                for(plusMinus = 0; plusMinus < this.m_ClusterCentroids.numAttributes(); ++plusMinus) {
                    if(this.m_ClusterCentroids.attribute(plusMinus).isNominal()) {
                        Attribute var20 = this.m_ClusterCentroids.attribute(plusMinus);

                        for(var24 = 0; var24 < this.m_ClusterCentroids.numInstances(); ++var24) {
                            i = var20.value((int)this.m_ClusterCentroids.instance(var24).value(plusMinus));
                            if(i.length() > maxWidth) {
                                maxWidth = i.length();
                            }
                        }

                        for(var24 = 0; var24 < var20.numValues(); ++var24) {
                            i = var20.value(var24) + " ";
                            if(i.length() > maxAttWidth) {
                                maxAttWidth = i.length();
                            }
                        }
                    }
                }

                if(this.m_displayStdDevs) {
                    for(plusMinus = 0; plusMinus < this.m_ClusterCentroids.numAttributes(); ++plusMinus) {
                        if(this.m_ClusterCentroids.attribute(plusMinus).isNominal()) {
                            temp = Utils.maxIndex(this.m_FullNominalCounts[plusMinus]);
                            byte var26 = 6;
                            i = "" + this.m_FullNominalCounts[plusMinus][temp];
                            if(i.length() + var26 > maxWidth) {
                                maxWidth = i.length() + 1;
                            }
                        }
                    }
                }

                double[] var21 = this.m_ClusterSizes;
                temp = var21.length;

                String strVal;
                for(var24 = 0; var24 < temp; ++var24) {
                    double var25 = var21[var24];
                    strVal = "(" + var25 + ")";
                    if(strVal.length() > maxWidth) {
                        maxWidth = strVal.length();
                    }
                }

                if(this.m_displayStdDevs && maxAttWidth < "missing".length()) {
                    maxAttWidth = "missing".length();
                }

                String var22 = "+/-";
                maxAttWidth += 2;
                if(this.m_displayStdDevs && containsNumeric) {
                    maxWidth += var22.length();
                }

                if(maxAttWidth < "Attribute".length() + 2) {
                    maxAttWidth = "Attribute".length() + 2;
                }

                if(maxWidth < "Full Data".length()) {
                    maxWidth = "Full Data".length() + 1;
                }

                if(maxWidth < "missing".length()) {
                    maxWidth = "missing".length() + 1;
                }

                StringBuffer var23 = new StringBuffer();
                if(!this.m_unable) {
                    var23.append("\nkMeans after final generation\n======\n");
                } else {
                    var23.append("\nSimpleKMeans Results\n======\n");
                }

                var23.append("\nNumber of iterations: " + this.m_Iterations);
                if(!this.m_FastDistanceCalc) {
                    var23.append("\n");
                    if(this.m_DistanceFunction instanceof EuclideanDistance) {
                        var23.append("Within cluster sum of squared errors: " + Utils.sum(this.m_squaredErrors));
                    } else {
                        var23.append("Sum of within cluster distances: " + Utils.sum(this.m_squaredErrors));
                    }
                }

                var23.append("\n\nInitial starting points");
                if(!this.m_unable) {
                    var23.append(" (final chromosome)");
                } else {
                    var23.append("");
                }

                var23.append(":\n");
                if(this.m_initializationMethod != 2) {
                    var23.append("\n");

                    for(var24 = 0; var24 < this.m_initialStartPoints.numInstances(); ++var24) {
                        var23.append("Cluster " + var24 + ": " + this.m_initialStartPoints.instance(var24)).append("\n");
                    }
                } else {
                    var23.append(this.m_canopyClusters.toString(false));
                }

                if(this.m_speedUpDistanceCompWithCanopies) {
                    var23.append("\nReduced number of distance calculations by using canopies.");
                    if(this.m_initializationMethod != 2) {
                        var23.append("\nCanopy T2 radius: " + String.format("%-10.3f", new Object[]{Double.valueOf(this.m_canopyClusters.getActualT2())}));
                        var23.append("\nCanopy T1 radius: " + String.format("%-10.3f", new Object[]{Double.valueOf(this.m_canopyClusters.getActualT1())})).append("\n");
                    }
                }

                if(!this.m_dontReplaceMissing) {
                    var23.append("\nMissing values globally replaced with mean/mode");
                }

                var23.append("\n\nFinal cluster centroids:\n");
                var23.append(this.pad("Cluster#", " ", maxAttWidth + maxWidth * 2 + 2 - "Cluster#".length(), true));
                var23.append("\n");
                var23.append(this.pad("Attribute", " ", maxAttWidth - "Attribute".length(), false));
                var23.append(this.pad("Full Data", " ", maxWidth + 1 - "Full Data".length(), true));

                for(var24 = 0; var24 < this.m_NumClusters; ++var24) {
                    i = "" + var24;
                    var23.append(this.pad(i, " ", maxWidth + 1 - i.length(), true));
                }

                var23.append("\n");
                String var29 = "(" + Utils.sum(this.m_ClusterSizes) + ")";
                var23.append(this.pad(var29, " ", maxAttWidth + maxWidth + 1 - var29.length(), true));

                int var27;
                for(var27 = 0; var27 < this.m_NumClusters; ++var27) {
                    var29 = "(" + this.m_ClusterSizes[var27] + ")";
                    var23.append(this.pad(var29, " ", maxWidth + 1 - var29.length(), true));
                }

                var23.append("\n");
                var23.append(this.pad("", "=", maxAttWidth + maxWidth * (this.m_ClusterCentroids.numInstances() + 1) + this.m_ClusterCentroids.numInstances() + 1, true));
                var23.append("\n");

                for(var27 = 0; var27 < this.m_ClusterCentroids.numAttributes(); ++var27) {
                    String attName = this.m_ClusterCentroids.attribute(var27).name();
                    var23.append(attName);

                    for(int var28 = 0; var28 < maxAttWidth - attName.length(); ++var28) {
                        var23.append(" ");
                    }

                    String valMeanMode;
                    if(this.m_ClusterCentroids.attribute(var27).isNominal()) {
                        if(this.m_FullMeansOrMediansOrModes[var27] == -1.0D) {
                            valMeanMode = this.pad("missing", " ", maxWidth + 1 - "missing".length(), true);
                        } else {
                            valMeanMode = this.pad(strVal = this.m_ClusterCentroids.attribute(var27).value((int)this.m_FullMeansOrMediansOrModes[var27]), " ", maxWidth + 1 - strVal.length(), true);
                        }
                    } else if(Double.isNaN(this.m_FullMeansOrMediansOrModes[var27])) {
                        valMeanMode = this.pad("missing", " ", maxWidth + 1 - "missing".length(), true);
                    } else {
                        valMeanMode = this.pad(strVal = Utils.doubleToString(this.m_FullMeansOrMediansOrModes[var27], maxWidth, 4).trim(), " ", maxWidth + 1 - strVal.length(), true);
                    }

                    var23.append(valMeanMode);

                    for(int stdDevVal = 0; stdDevVal < this.m_NumClusters; ++stdDevVal) {
                        if(this.m_ClusterCentroids.attribute(var27).isNominal()) {
                            if(this.m_ClusterCentroids.instance(stdDevVal).isMissing(var27)) {
                                valMeanMode = this.pad("missing", " ", maxWidth + 1 - "missing".length(), true);
                            } else {
                                valMeanMode = this.pad(strVal = this.m_ClusterCentroids.attribute(var27).value((int)this.m_ClusterCentroids.instance(stdDevVal).value(var27)), " ", maxWidth + 1 - strVal.length(), true);
                            }
                        } else if(this.m_ClusterCentroids.instance(stdDevVal).isMissing(var27)) {
                            valMeanMode = this.pad("missing", " ", maxWidth + 1 - "missing".length(), true);
                        } else {
                            valMeanMode = this.pad(strVal = Utils.doubleToString(this.m_ClusterCentroids.instance(stdDevVal).value(var27), maxWidth, 4).trim(), " ", maxWidth + 1 - strVal.length(), true);
                        }

                        var23.append(valMeanMode);
                    }

                    var23.append("\n");
                    if(this.m_displayStdDevs) {
                        String var30 = "";
                        if(this.m_ClusterCentroids.attribute(var27).isNominal()) {
                            Attribute var31 = this.m_ClusterCentroids.attribute(var27);

                            int k;
                            for(int count = 0; count < var31.numValues(); ++count) {
                                String val = "  " + var31.value(count);
                                var23.append(this.pad(val, " ", maxAttWidth + 1 - val.length(), false));
                                double percent = this.m_FullNominalCounts[var27][count];
                                k = (int)(this.m_FullNominalCounts[var27][count] / Utils.sum(this.m_ClusterSizes) * 100.0D);
                                String percentS1 = "" + k + "%)";
                                percentS1 = this.pad(percentS1, " ", 5 - percentS1.length(), true);
                                var30 = "" + percent + " (" + percentS1;
                                var30 = this.pad(var30, " ", maxWidth + 1 - var30.length(), true);
                                var23.append(var30);

                                for(int k1 = 0; k1 < this.m_NumClusters; ++k1) {
                                    k = (int)(this.m_ClusterNominalCounts[k1][var27][count] / this.m_ClusterSizes[k1] * 100.0D);
                                    percentS1 = "" + k + "%)";
                                    percentS1 = this.pad(percentS1, " ", 5 - percentS1.length(), true);
                                    var30 = "" + this.m_ClusterNominalCounts[k1][var27][count] + " (" + percentS1;
                                    var30 = this.pad(var30, " ", maxWidth + 1 - var30.length(), true);
                                    var23.append(var30);
                                }

                                var23.append("\n");
                            }

                            if(this.m_FullMissingCounts[var27] > 0.0D) {
                                var23.append(this.pad("  missing", " ", maxAttWidth + 1 - "  missing".length(), false));
                                double var32 = this.m_FullMissingCounts[var27];
                                int var33 = (int)(this.m_FullMissingCounts[var27] / Utils.sum(this.m_ClusterSizes) * 100.0D);
                                String percentS = "" + var33 + "%)";
                                percentS = this.pad(percentS, " ", 5 - percentS.length(), true);
                                var30 = "" + var32 + " (" + percentS;
                                var30 = this.pad(var30, " ", maxWidth + 1 - var30.length(), true);
                                var23.append(var30);

                                for(k = 0; k < this.m_NumClusters; ++k) {
                                    var33 = (int)(this.m_ClusterMissingCounts[k][var27] / this.m_ClusterSizes[k] * 100.0D);
                                    percentS = "" + var33 + "%)";
                                    percentS = this.pad(percentS, " ", 5 - percentS.length(), true);
                                    var30 = "" + this.m_ClusterMissingCounts[k][var27] + " (" + percentS;
                                    var30 = this.pad(var30, " ", maxWidth + 1 - var30.length(), true);
                                    var23.append(var30);
                                }

                                var23.append("\n");
                            }

                            var23.append("\n");
                        } else {
                            if(Double.isNaN(this.m_FullMeansOrMediansOrModes[var27])) {
                                var30 = this.pad("--", " ", maxAttWidth + maxWidth + 1 - 2, true);
                            } else {
                                var30 = this.pad(strVal = var22 + Utils.doubleToString(this.m_FullStdDevs[var27], maxWidth, 4).trim(), " ", maxWidth + maxAttWidth + 1 - strVal.length(), true);
                            }

                            var23.append(var30);

                            for(int j = 0; j < this.m_NumClusters; ++j) {
                                if(this.m_ClusterCentroids.instance(j).isMissing(var27)) {
                                    var30 = this.pad("--", " ", maxWidth + 1 - 2, true);
                                } else {
                                    var30 = this.pad(strVal = var22 + Utils.doubleToString(this.m_ClusterStdDevs.instance(j).value(var27), maxWidth, 4).trim(), " ", maxWidth + 1 - strVal.length(), true);
                                }

                                var23.append(var30);
                            }

                            var23.append("\n\n");
                        }
                    }
                }

                var23.append("\n\n");
                return var23.toString();
            }
        }

        private void setUnable(boolean u) {
            this.m_unable = u;
        }

        private String pad(String source, String padChar, int length, boolean leftPad) {
            StringBuffer temp = new StringBuffer();
            int i;
            if(leftPad) {
                for(i = 0; i < length; ++i) {
                    temp.append(padChar);
                }

                temp.append(source);
            } else {
                temp.append(source);

                for(i = 0; i < length; ++i) {
                    temp.append(padChar);
                }
            }

            return temp.toString();
        }
    }

    private class FitnessContainer implements Comparable<MyGenClustPlusPlus.FitnessContainer> {
        double fitness;
        MyGenClustPlusPlus.MKMeans clustering;

        FitnessContainer(double f, MyGenClustPlusPlus.MKMeans c) {
            this.fitness = f;
            this.clustering = c;
        }

        public int compareTo(MyGenClustPlusPlus.FitnessContainer other) {
            return Double.compare(this.fitness, other.fitness);
        }
    }
}

