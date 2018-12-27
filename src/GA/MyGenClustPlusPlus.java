package GA;

import java.util.*;

import clustering.Evaluator;
import clustering.KMeans;
import smile.validation.AdjustedRandIndex;
import utils.NCConstruct;

import utils.Silh;
import weka.clusterers.RandomizableClusterer;
import weka.core.Capabilities;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.Capabilities.Capability;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;

public class MyGenClustPlusPlus extends RandomizableClusterer implements TechnicalInformationHandler {
    public enum FITNESS {DBINDEX, SILHOUETTE, MULTIOBJECTIVE_SUM};

    private static final long serialVersionUID = -7247404718496233612L;
    private static final double MAXIMIN_THRESHOLD = -0.0001;
    private Instances m_data;
    private int m_numberOfClusters = 0;
    private int m_numberOfGenerations = 60;
    private int m_initialPopulationSize = 30;
    private int m_maxKMeansIterationsInitial = 60;
    private int m_maxKMeansIterationsQuick = 15;
    private int m_maxKMeansIterationsFinal = 50;
    private double m_duplicateThreshold = 0.0D;
    private int m_startChromosomeSelectionGeneration = 11;
    private KMeans m_bestChromosome;
    private double m_bestFitness;
    private int m_bestFitnessIndex;
    protected KMeans m_builtClusterer = null;
    private Random m_rand = null;
    private boolean m_dontReplaceMissing = false;
    public static final int SUPPLIED = 4;

    public void setNcConstruct(NCConstruct ncConstruct) {
        this.ncConstruct = ncConstruct;
    }

    private NCConstruct ncConstruct;
    private AdjustedRandIndex adjustedRandIndex = new AdjustedRandIndex();
    private Silh silhouette = new Silh();
    private Evaluator evaluator = new Evaluator();
    private Evaluator.Evaluation[] evaluations;
    private double[][] myData;
    private boolean normalizeObjectives;

    public void setFitnessType(FITNESS aFitnessType) {
        this.fitnessType = aFitnessType;
    }

    private FITNESS fitnessType;

    public void setDistance(double distance) {
        this.myDistance = distance;
    }

    private double myDistance;

    public void setMaximin(boolean maximin) {
        this.maximin = maximin;
    }

    private boolean maximin;

    public void setHillClimb(boolean hillClimb) {
        this.hillClimb = hillClimb;
    }

    private boolean hillClimb;

    public void setNormalizeObjectives(boolean normalizeObjectives) {
        this.normalizeObjectives = normalizeObjectives;
    }

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
        this.myData = utils.Utils.wekaInstancesToArray(data);

        /* Component 3: generate initial population */
        KMeans[] initialPopulation = this.generateInitialPopulation();
        /* evaluate and update utopia point */
        double[][] tmpObjectives = evaluate(initialPopulation);

        assert (initialPopulation != null);

        this.m_bestFitness = Double.NEGATIVE_INFINITY;
        this.m_bestFitnessIndex = -1;

        /* identify best chromosome */
        double[] fitnessArr = fitness(initialPopulation);

        this.m_bestFitnessIndex = pickBest(initialPopulation, fitnessArr, tmpObjectives);
        this.m_bestFitness = fitnessArr[this.m_bestFitnessIndex];
        this.m_bestChromosome = new KMeans(initialPopulation[this.m_bestFitnessIndex]);

        /* select initial population out of pre-initial population probabilistically
         * When a value of k is chosen, the next available solution in order of descending fitness
         * is added to initial population. */

        KMeans[] mainPopulation = this.probabilisticSelection(initialPopulation);
        KMeans[] prevPopulation = new KMeans[mainPopulation.length];

        int newBestIndex;
        KMeans finalRun;
        /* 60 generations */
        for(int generationIdx = 0; generationIdx <= this.m_numberOfGenerations; ++generationIdx) {
            KMeans[] resultingPopulation;
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
                        finalRun = new KMeans();
                        finalRun.setSeed(this.m_rand.nextInt());
                        finalRun.setInitializationMethod(KMeans.Initialization.HILL_CLIMBER);
                        finalRun.setInitial(resultingPopulation[newBestIndex].getCentroids());

                        finalRun.setMaxIterations(this.m_maxKMeansIterationsQuick);
                        finalRun.buildClusterer(this.myData);
                        if(finalRun.getCentroids().length <= 1) {
                            f = this.m_rand.nextInt((int)(Math.sqrt((double)data.size()) - 2.0D)) + 2;
                            KMeans t = new KMeans(f, this.myDistance);
                            t.setSeed(this.m_rand.nextInt());
                            t.buildClusterer(this.myData);
                            finalRun = new KMeans(t);
                        }
                    } while(finalRun.getCentroids().length <= 1);

                    resultingPopulation[newBestIndex] = finalRun;
                }
            }
            /* step 2 (4 of prob. cloning) - elitism */
            resultingPopulation = this.elitism(resultingPopulation);
            /* step 3 (5 of prob. cloning) - mutation */
            resultingPopulation = this.mutation(resultingPopulation);
            /* step 4 (6 of prob. cloning) - elitism */
            resultingPopulation = this.elitism(resultingPopulation);

            // remove clustering solutions with one cluster
            for (int i = 0; i < resultingPopulation.length; ++i) {
                while (resultingPopulation[i].getCentroids().length <= 1) {
                    f = this.m_rand.nextInt((int) (Math.sqrt((double) data.size()) - 2.0D)) + 2;
                    KMeans t = new KMeans(f, this.myDistance);
                    t.setSeed(this.m_rand.nextInt());
                    t.buildClusterer(this.myData);
                    resultingPopulation[i] = new KMeans(t);

                    resultingPopulation[i] = t;
                }
            }

            if(this.m_builtClusterer != null) {
                System.out.println("failed after crossover/probabilistic cloning");
                return;
            }

            if(generationIdx <= this.m_startChromosomeSelectionGeneration) {
                /* copy resulting population into main population instance */
                for(newBestIndex = 0; newBestIndex < resultingPopulation.length; ++newBestIndex) {
                    mainPopulation[newBestIndex] = new KMeans(resultingPopulation[newBestIndex]);
                }
            } else {
                /* start chromosome selection at this point
                 * merges all chromosomes between two populations of size s:
                 * the last (most recent) population and the resulting generation from all operations.
                 * This second type of elitism chooses the highest fitted s chromosomes from the merged populations.
                 * This restricts radical disruption by the genetic operators,
                 * but it still enables their exploratory nature to drive the genetic search. */
                double[] fitnessArray;

                KMeans[] kMeanses = new KMeans[2 * resultingPopulation.length];
                /* fitness corresponding to resulting population */
                for(int var21 = 0; var21 < resultingPopulation.length; ++var21) {
                    kMeanses[var21]=resultingPopulation[var21];
                }
                /* fitness corresponding to previous population */
                for(int var21 = 0; var21 < resultingPopulation.length; ++var21) {
                    kMeanses[var21+resultingPopulation.length] = prevPopulation[var21];
                }
                fitnessArray = fitness(kMeanses);

                List<FitnessContainer> nextPop = new ArrayList<>(kMeanses.length);
                for (int i = 0; i < kMeanses.length; ++i) {
                    nextPop.add(new FitnessContainer(fitnessArray[i], kMeanses[i]));
                }

                if (maximin) {
                    // select non-dominated set to next population
                    double[][] nextPopObjectives = new double[nextPop.size()][];
                    for (int i = 0; i < nextPop.size(); ++i) {
                        nextPopObjectives[i] = evaluate(nextPop.get(i).clustering.getLabels());
                    }
                    // update utopia point
                    //updateUtopiaDystopia(nextPopObjectives);
                    /*if (normalizeObjectives) {
                        utils.Utils.normalize(nextPopObjectives);
                    }*/
                    // MaxiMin strategy
                    double[] fitness = utils.Utils.determineParetoSet(utils.Utils.deepCopy(nextPopObjectives));
                    int mainPopCurIdx = 0;
                    for (int i = 0; i < nextPopObjectives.length; ++i) {
                        nextPop.get(i).setFitness(fitness[i]);
                        if (fitness[i] < MAXIMIN_THRESHOLD && mainPopCurIdx < mainPopulation.length) {
                            mainPopulation[mainPopCurIdx++] = new KMeans(nextPop.get(i).clustering);
                            nextPop.set(i, null);
                        }
                    }
                    nextPop.removeIf(fitnessContainer -> (fitnessContainer == null));

                    // randomly add weakly-dominated or dominated
                    // solutions from nextPop, until mainPopulation is filled up by population size
                    int roomToFill = mainPopulation.length - mainPopCurIdx;
                    // now randomly pick nextPopList
                    int i = 0;
                    /*while (i < roomToFill) {
                        int idxPick = m_rand.nextInt(nextPop.size());
                        assert (nextPop.get(idxPick) != null);
                        mainPopulation[mainPopCurIdx++] = new KMeans(nextPop.get(idxPick).clustering);
                        i++;
                        nextPop.remove(idxPick);
                    }*/
                    Collections.sort(nextPop, Collections.reverseOrder());
                    while (i < roomToFill) {
                        mainPopulation[mainPopCurIdx++] = new KMeans(nextPop.get(i).clustering);
                        ++i;
                    }
                    assert (mainPopCurIdx == mainPopulation.length);
                } else {
                    Collections.sort(nextPop, Collections.reverseOrder());
                    for (int i = 0; i < resultingPopulation.length; ++i) {
                        mainPopulation[i] = new KMeans(nextPop.get(i).clustering);
                    }
                }
            }

            /* update previous population */
            for(newBestIndex = 0; newBestIndex < mainPopulation.length; ++newBestIndex) {
                prevPopulation[newBestIndex] = new KMeans(mainPopulation[newBestIndex]);
            }
            //System.out.println("best objective coordinates: " + Arrays.toString(objBestCoordinates));
        }
        /* FINAL step - choose best performing chromosome to use as
         * the initial solution of a final full-length MK-Means to deliver the final buildClusterer solution */
        fitnessArr = fitness(mainPopulation);
        tmpObjectives = evaluate(mainPopulation);
        newBestIndex = pickBest(mainPopulation, fitnessArr, tmpObjectives);

        assert (newBestIndex >= 0);
        this.m_bestFitnessIndex = newBestIndex;
        this.m_bestFitness = fitnessArr[newBestIndex];
        this.m_bestChromosome = new KMeans(mainPopulation[newBestIndex]);
        if (this.hillClimb) {
            finalRun = new KMeans();
            finalRun.setSeed(this.m_rand.nextInt());
            finalRun.setInitializationMethod(KMeans.Initialization.HILL_CLIMBER);
            finalRun.setInitial(this.m_bestChromosome.getCentroids());
            finalRun.setMaxIterations(this.m_maxKMeansIterationsFinal);
            finalRun.buildClusterer(this.myData);
        } else {
            finalRun = mainPopulation[newBestIndex];
        }
        this.m_builtClusterer = finalRun;
        this.m_numberOfClusters = this.m_builtClusterer.getCentroids().length;
    }

    public int pickBest(KMeans[] population, double[] fitnessArr, double[][] mainPopObjectives) {
        double var17 = Double.NEGATIVE_INFINITY;
        int newBestIndex = -1;
        //this.finalpopulation = mainPopulation;
        // evaluate population
        for (int i = 0; i < population.length; ++i) {
            mainPopObjectives[i] = evaluate(population[i].getLabels());
        }

        // update utopia point
        //updateUtopiaDystopia(mainPopObjectives);

        /*System.out.println("-- FINAL POP START");
        measureFinalPop(mainPopulation, myData, trueLabels);
        System.out.println("-- FINAL POP END");*/
        if (maximin) {
            boolean chooseFromNonDom = true;
            double[][] cloned;
            if (chooseFromNonDom) {
                // MaxiMin strategy
                double[] fitness = utils.Utils.determineParetoSet(utils.Utils.deepCopy(mainPopObjectives));
                // indices of non-dominated solutions
                assert (fitness.length == mainPopObjectives.length);
                List<Integer> nonDomSetIndices = new ArrayList<>();
                for (int i = 0; i < mainPopObjectives.length; ++i) {
                    if (fitness[i] < MAXIMIN_THRESHOLD) {
                        nonDomSetIndices.add(i);
                    }
                }
                System.out.println(Arrays.toString(fitness));

                // for printing non-dominated solutions
                KMeans[] nonDomMKMeans = new KMeans[nonDomSetIndices.size()];
                for (int i = 0; i < nonDomSetIndices.size(); ++i) {
                    int index = nonDomSetIndices.get(i);
                    nonDomMKMeans[i] = new KMeans(population[index]);
                }

                System.out.println("size of pops: " + population.length + " : " + nonDomMKMeans.length);
                // if there is no non-dominated clustering solution
                if (nonDomSetIndices.size() == 0) {
                    /*for (int i = 0; i < fitness.length; ++i) {
                        if (fitness[i] >= MAXIMIN_THRESHOLD) {
                            nonDomSetIndices.add(i);
                            break;
                        }
                    }*/
                    cloned = utils.Utils.deepCopy(mainPopObjectives);
                } else {
                    //measureFinalPop(nonDomMKMeans, myData, trueLabels);
                    //System.out.println("--- FINAL NON-DOM END");

                    double[][] nonDomSetObjs = new double[nonDomSetIndices.size()][];
                    for (int i = 0; i < nonDomSetObjs.length; ++i) {
                        int index = nonDomSetIndices.get(i);
                        nonDomSetObjs[i] = mainPopObjectives[index];
                    }
                    // choose final cluster solution
                    cloned = utils.Utils.deepCopy(nonDomSetObjs);
                }
            } else {
                cloned = utils.Utils.deepCopy(mainPopObjectives);
            }

            if (this.normalizeObjectives) {
                utils.Utils.normalize(cloned, objBestCoordinates, objWorstCoordinates);
                newBestIndex = utils.Utils.pickClosestToUtopia(cloned, new double[]{0.0, 0.0});
            } else {
                newBestIndex = utils.Utils.pickClosestToUtopia(cloned, objBestCoordinates);
            }
        } else {
            for (int var20 = 0; var20 < population.length; ++var20) {
                double var23 = fitnessArr[var20];
                if (var23 > var17) {
                    var17 = var23;
                    newBestIndex = var20;
                }
            }
        }

        return newBestIndex;
    }

    public void measureFinalPop(KMeans[] pop, double[][] data, int[] labelsTrue) throws Exception {
        int[] labelsPred;
        for (int i = 0; i < pop.length; ++i) {
            labelsPred = pop[i].getLabels();
            double[] objCloned = evaluate(pop[i].getLabels().clone());
            //updateUtopiaDystopia(objCloned);
            utils.Utils.normalize(objCloned, objBestCoordinates, objWorstCoordinates);
            System.out.println(Arrays.toString(objCloned));

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
            System.out.println("ARI score for chromosome: " + utils.Utils.doublePrecision(ARI, 4));
            System.out.println("DB score for chromosome:  " + utils.Utils.doublePrecision(myDBWithMyCentroids, 4));
            System.out.println("num of clusters for chromosome: " + k +  " : " + pop[i].getCentroids().length);
            System.out.println("------------");
        }
    }

    private KMeans[] generateInitialPopulation() throws Exception {
        int multiplier = 5;
        /* pre-initial size is 3 * initial size = 90 */
        int maxK = 3 * this.m_initialPopulationSize / 10 + 1;
        int numberOfChromosomes = multiplier * (maxK - 1) * 2;
        KMeans[] population = new KMeans[numberOfChromosomes];
        int chromosomeCount = 0;

        int i;
        int randomK;
        /* Initial Population with Probabilistic Selection */
        for(i = 2; i <= maxK; ++i) {
            /* five clusterings for each value of k */
            for(randomK = 0; randomK < multiplier; ++randomK) {
                KMeans j = new KMeans(i, this.myDistance);
                int chromosome = -1;

                // chromosome is built until number of clusters is at least two
                do {
                    ++chromosome;
                    if(chromosome > 100) {
                        System.out.println("Unable to cluster this dataset using genetic algorithm, try imputing missing values first.");
                        this.m_builtClusterer = j;
                        this.m_numberOfClusters = this.m_builtClusterer.getCentroids().length;
                        return null;
                    }

                    j.setSeed(this.m_rand.nextInt());
                    j.setMaxIterations(this.m_maxKMeansIterationsInitial);
                    j.setInitializationMethod(KMeans.Initialization.KMEANS_PLUS_PLUS);
                    j.buildClusterer(this.myData);
                } while(utils.Utils.distinctNumberOfItems(j.getLabels()) < 2);

                population[chromosomeCount++] = j;
            }
        }

        for(i = 0; i < maxK - 1; ++i) {
            randomK = this.m_rand.nextInt((int)(Math.sqrt((double)this.myData.length) - 2.0D)) + 2;

            for(int var10 = 0; var10 < multiplier; ++var10) {
                KMeans var11 = new KMeans(randomK, this.myDistance);
                var11.setSeed(this.m_rand.nextInt());
                var11.setInitializationMethod(KMeans.Initialization.KMEANS_PLUS_PLUS);

                // chromosome is built until number of clusters is at least two
                do {
                    var11.buildClusterer(this.myData);
                } while(utils.Utils.distinctNumberOfItems(var11.getLabels()) < 2);

                population[chromosomeCount++] = var11;
            }
        }

        return population;
    }

    public int[] getLabels() throws Exception {
        return m_builtClusterer.getLabels();
    }

    /*private void updateUtopiaDystopia(double[][] objectives) {
        // update utopia point
        for (int i = 0; i < objectives.length; ++i) {
            updateUtopiaDystopia(objectives[i]);
        }
    }*/

    private void updateUtopiaDystopia(double[] objs) {
        for (int iO = 0; iO < evaluations.length; ++iO) {
            double obj = objs[iO];
            if (obj < objBestCoordinates[iO]) {
                objBestCoordinates[iO] = obj;
            }
            if (obj > objWorstCoordinates[iO]) {
                objWorstCoordinates[iO] = obj;
            }
        }
    }

    private double[] evaluate(int[] labelPred) {
        double[] result = new double[evaluations.length];
        for (int iE = 0; iE < evaluations.length; ++iE) {
            result[iE] = evaluator.evaluate(labelPred, evaluations[iE], myData, ncConstruct);
        }
        updateUtopiaDystopia(result);

        return result;
    }

    private double[][] evaluate(KMeans[] population) {
        double[][] objs = new double[population.length][];
        for (int i = 0; i < population.length; ++i) {
            objs[i] = evaluate(population[i].getLabels());
        }
        return objs;
    }

    private double modifiedDBIndex(KMeans chromosome) {
        double[][] centroids = chromosome.getCentroids();

        int numClust = centroids.length;
        double[] Si = new double[numClust];
        int[] Ti = new int[numClust];
        int[] clustSize = new int[numClust];
        if(numClust == 1) {
            return Double.POSITIVE_INFINITY;
        } else {
            int[] assignments = null;

            try {
                assignments = utils.Utils.adjustLabels(chromosome.getLabels());
            } catch (Exception var19) {
                var19.printStackTrace();
            }

            int DB;
            /* step 1 - computer cluster diameter */
            for(DB = 0; DB < assignments.length; ++DB) {
                try {
                    /* e is cluster id */
                    int e = assignments[DB];
                    ++clustSize[e];
                    /* Si is cluster diamater, not averaged yet */
                    Si[e] += Math.abs(utils.Utils.dist(this.m_data.get(DB).toDoubleArray(), centroids[e], 2.0));
                } catch (Exception var18) {
                    var18.printStackTrace();
                    System.exit(0);
                }
            }

            /* still step 1 - compute average  */
            for(DB = 0; DB < centroids.length; ++DB) {
                Ti[DB] = clustSize[DB];
                if(clustSize[DB] == 0) {
                    return 0.0D;
                }

                Si[DB] /= (double)Ti[DB];
            }

            double var20 = 0.0D;

            /* step 2 - compute Rij for each possible pair of clusters */
            for(int i = 0; i < centroids.length; ++i) {
                if(clustSize[i] != 0) {
                    double max = -1.0D / 0.0;
                    int maxIndex = -2147483648;

                    /* find max Rij for cluster */
                    for(int j = 0; j < centroids.length; ++j) {
                        if(i != j) {
                            double Rij = Si[i] + Si[j] / Math.abs(utils.Utils.dist(centroids[i], centroids[j], 2.0));
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
            return var20;
        }
    }

    private double myDBIndex(KMeans chromosome) {
        int[] labelsPred = null;
        try {
            labelsPred = chromosome.getLabels();
        } catch (Exception var19) {
            var19.printStackTrace();
        }

        HashMap<Integer, double[]> centroids = utils.Utils.centroids(myData, labelsPred);
        if (centroids.size() < 2) {
            return 0.0;
        }

        double dbScore = utils.Utils.dbIndexScore(centroids, labelsPred, myData);

        return dbScore;
    }

    private double fitness(KMeans chromosome) {
        if (fitnessType == FITNESS.MULTIOBJECTIVE_SUM) {
            double[] objs = evaluate(chromosome.getLabels());
            //updateUtopiaDystopia(objs);

            if (normalizeObjectives) {
                utils.Utils.normalize(objs, objBestCoordinates, objWorstCoordinates);
            }
            return 1.0D / utils.Utils.sum(objs, 1.0);
        } else if (this.fitnessType == FITNESS.SILHOUETTE) {
            return this.silhouette.compute(chromosome.getLabels(), this.myData);
        } else if (fitnessType == FITNESS.DBINDEX) {
            return 1.0D / modifiedDBIndex(chromosome);
        }
        return 1.0D / modifiedDBIndex(chromosome);
    }

    private double[] fitness(KMeans[] pop) {
        double[] res = new double[pop.length];
        for (int i = 0; i < res.length; ++i) {
            res[i] = fitness(pop[i]);
        }

        utils.Utils.normalize(res);
        boolean isAllEqual = true;
        for (double f: res) {
            if (f != 0.0) {
                isAllEqual = false;
                break;
            }
        }
        if (isAllEqual) {
            for (int i = 0; i < res.length; ++i) {
                res[i] = 1;
            }
        }

        return res;
    }

    private KMeans[] probabilisticSelection(KMeans[] population) {
        int multiplier = 5;
        KMeans[] selectedPopulation = new KMeans[this.m_initialPopulationSize];
        int currentSelections = 0;
        double[] fitnessArray = new double[population.length];
        boolean[] usedChromosome = new boolean[population.length];
        double[] TkArray = new double[population.length / multiplier];
        int[] kArray = new int[population.length / multiplier];
        double sumTk = 0.0D;

        /* compute average fitness value for each value of k */
        fitnessArray = fitness(population);
        // normalize to allow prob selection
        for (double f: fitnessArray) {
            assert (f >= 0.0);
        }

        for(int p = 0; p < population.length; p += multiplier) {
            kArray[p / multiplier] = population[p].numberOfClusters();
            double Tk = 0.0D;

            for(int j = 0; j < multiplier; ++j) {
                double i = fitnessArray[p + j];
                Tk += i;
            }

            sumTk += Tk;
            TkArray[p / multiplier] = Tk;
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
                    for(int selected = 0; selected < multiplier; ++selected) {
                        if(fitnessArray[var22 + selected] > max && !usedChromosome[var22 + selected]) {
                            max = fitnessArray[var22 + selected];
                            maxIndex = var22 + selected;
                        }
                    }

                    KMeans var23;
                    if(maxIndex == 2147483647) {
                        /* if clusterings corresponding to given k are chosen more than five times
                         * then we create a new chromosome by running MK-Means/MK-Means++ once more
                         * with input k for the number of clusters. */
                        try {
                            KMeans ex = new KMeans(kArray[var22], this.myDistance);
                            ex.setSeed(this.m_rand.nextInt());
                            ex.buildClusterer(this.myData);
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

    private Instances centroidsFromKmeans(KMeans kMeans) {
        double[][] centroidsArr = kMeans.getCentroids();
        Instances instances = new Instances(this.m_data, centroidsArr.length);
        for (double[] arr: centroidsArr) {
            instances.add(new DenseInstance(1.0, arr));
        }
        return instances;
    }

    private KMeans[] crossover(KMeans[] selectedPopulation) throws Exception {
        Instances[] offspring = new Instances[selectedPopulation.length];
        int offspringCounter = 0;
        MyGenClustPlusPlus.FitnessContainer[] sorted = new MyGenClustPlusPlus.FitnessContainer[selectedPopulation.length];
        double fitnessSum = 0.0D;
        double[] fitnessArr;
        fitnessArr = fitness(selectedPopulation);

        /* normalize fitness in any case, since normObjectives relates to objectives, not fitness */
        for (double f: fitnessArr) {
            assert (f >= 0.0);
        }

        for(int offspringMK = 0; offspringMK < sorted.length; ++offspringMK) {
            fitnessSum += fitnessArr[offspringMK];
            sorted[offspringMK] = new MyGenClustPlusPlus.FitnessContainer(fitnessArr[offspringMK], selectedPopulation[offspringMK]);
        }

        Arrays.sort(sorted, Collections.reverseOrder());

        while(sorted.length > 0) {
            KMeans var27 = sorted[0].clustering;
            fitnessSum -= sorted[0].fitness;
            sorted[0] = null;
            KMeans var29 = sorted[sorted.length - 1].clustering;
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
            //DenseInstance
            Instances var33 = centroidsFromKmeans(var27);
            Instances parentTwoCentroids = centroidsFromKmeans(var29);
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
                    targetSecondHalf = Math.abs(utils.Utils.dist(var33.get(refRandom).toDoubleArray(),
                            parentTwoCentroids.get(targetFirstHalf).toDoubleArray(), this.myDistance));
                    if(targetSecondHalf == 1.0D / 0.0) {
                        utils.Utils.dist(var33.get(refRandom).toDoubleArray(),
                                parentTwoCentroids.get(targetFirstHalf).toDoubleArray(), this.myDistance);
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
                        targetSecondHalf = Math.abs(utils.Utils.dist(parentTwoCentroids.get(refRandom).toDoubleArray(),
                                target.get(targetFirstHalf).toDoubleArray(), this.myDistance));
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

        KMeans[] var28 = new KMeans[offspring.length];

        for(int var30 = 0; var30 < offspring.length; ++var30) {
            for(int e = 0; e < offspring[var30].numInstances(); ++e) {
                for(int var32 = e + 1; var32 < offspring[var30].numInstances(); ++var32) {
                    if(Math.abs(utils.Utils.dist(offspring[var30].get(e).toDoubleArray()
                            , offspring[var30].get(var32).toDoubleArray(), this.myDistance)) <= this.m_duplicateThreshold) {
                        if(offspring[var30].numInstances() > 2) {
                            offspring[var30].remove(var32);
                        } else {
                            while(Math.abs(utils.Utils.dist(offspring[var30].get(e).toDoubleArray(),
                                    offspring[var30].get(var32).toDoubleArray(), this.myDistance)) <= this.m_duplicateThreshold) {
                                int attribute = this.m_rand.nextInt(offspring[var30].numAttributes());

                                double var34;
                                do {
                                    if(offspring[var30].attribute(attribute).isNumeric()) {
                                        for(var34 = this.m_data.attributeStats(attribute).numericStats.min + this.m_rand.nextDouble() * (this.m_data.attributeStats(attribute).numericStats.max - this.m_data.attributeStats(attribute).numericStats.min + 1.0D);
                                            var34 == 1.0D / 0.0;
                                            var34 = this.m_data.attributeStats(attribute).numericStats.min + this.m_rand.nextDouble() * (this.m_data.attributeStats(attribute).numericStats.max - this.m_data.attributeStats(attribute).numericStats.min + 1.0D)) {
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
                if (offspring[var30].size() > Math.sqrt(this.myData.length)) {
                    throw new Exception("too many clusters");
                }
                KMeans var31 = new KMeans();
                var31.setSeed(this.m_rand.nextInt());
                var31.setInitializationMethod(KMeans.Initialization.HILL_CLIMBER);
                var31.setInitial(utils.Utils.wekaInstancesToArray(offspring[var30]));
                var31.setMaxIterations(1);
                var31.buildClusterer(myData);
                var28[var30] = var31;
            } catch (Exception var26) {
                int randomK = this.m_rand.nextInt((int)(Math.sqrt((double)this.myData.length) - 2.0D)) + 2;
                KMeans kMeans = new KMeans(randomK, 2.0);
                kMeans.setSeed(this.m_rand.nextInt());
                kMeans.setMaxIterations(this.m_maxKMeansIterationsInitial);
                kMeans.setInitializationMethod(KMeans.Initialization.KMEANS_PLUS_PLUS);
                kMeans.buildClusterer(this.myData);
                var28[var30] = new KMeans(kMeans);
            }
        }

        /*double[][] tmpObjs = new double[var28.length][];
        for (int i = 0; i < var28.length; ++i) {
            tmpObjs[i] = evaluate(var28[i].getLabels());
        }
        updateUtopiaDystopia(utils.Utils.deepCopy(tmpObjs));*/

        return var28;
    }

    private KMeans[] probabilisticCloning(KMeans[] selectedPopulation) {
        KMeans[] newPopulation = new KMeans[selectedPopulation.length];
        MyGenClustPlusPlus.FitnessContainer[] fArray = new MyGenClustPlusPlus.FitnessContainer[selectedPopulation.length];
        double fitnessSum = 0.0D;

        int newCount;
        double i;
        double[] fitnessArr;
        fitnessArr = fitness(selectedPopulation);

        /* normalize to allow prob cloning */
        for (double f: fitnessArr) {
            assert (f >= 0.0);
        }

        for(newCount = 0; newCount < selectedPopulation.length; ++newCount) {
            fitnessSum += fitnessArr[newCount];
            fArray[newCount] = new MyGenClustPlusPlus.FitnessContainer(fitnessArr[newCount], selectedPopulation[newCount]);
        }

        newCount = 0;

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
            newPopulation[var12] = this.mutate(centroidsFromKmeans(newPopulation[var12]));
        }

        return newPopulation;
    }

    private KMeans[] mutation(KMeans[] crossoverPopulation) throws Exception {
        MyGenClustPlusPlus.FitnessContainer[] fArray = new MyGenClustPlusPlus.FitnessContainer[crossoverPopulation.length];
        double fitnessAvg = 0.0D;
        double fitnessMax = 4.9E-324D;

        int i;
        double prob;
        double[] fitnessArr;
        fitnessArr = fitness(crossoverPopulation);

        /* normalize to allow prob cloning */
        for (double f: fitnessArr) {
            assert (f >= 0.0);
        }

        for(i = 0; i < crossoverPopulation.length; ++i) {
            prob = fitnessArr[i];
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
                Instances centroids = centroidsFromKmeans(fArray[i].clustering);
                crossoverPopulation[i] = this.mutate(centroids);
            }
        }

        /*double[][] tmpObjs = new double[crossoverPopulation.length][];
        for (i = 0; i < crossoverPopulation.length; ++i) {
            tmpObjs[i] = evaluate(crossoverPopulation[i].getLabels());
        }
        updateUtopiaDystopia(tmpObjs);*/

        return crossoverPopulation;
    }

    private KMeans mutate(Instances centroids) {
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
                if(Math.abs(utils.Utils.dist(centroids.get(ex).toDoubleArray(),
                        centroids.get(k).toDoubleArray(), this.myDistance)) <= this.m_duplicateThreshold) {
                    if(centroids.numInstances() > 2) {
                        centroids.remove(k);
                    } else {
                        while(Math.abs(utils.Utils.dist(centroids.get(ex).toDoubleArray(),
                                centroids.get(k).toDoubleArray(), this.myDistance)) <= this.m_duplicateThreshold) {
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
            KMeans var8 = new KMeans();
            var8.setSeed(this.m_rand.nextInt());
            var8.setInitializationMethod(KMeans.Initialization.HILL_CLIMBER);
            var8.setInitial(utils.Utils.wekaInstancesToArray(centroids));
            var8.setMaxIterations(1);
            var8.buildClusterer(myData);
            return var8;
        } catch (Exception var7) {
            var7.printStackTrace();
            return null;
        }
    }

    private KMeans[] elitism(KMeans[] population) throws Exception {
        double worstFitness = 1.7976931348623157E308D;
        int worstIndex = 2147483647;
        double newBestFitness = 4.9E-324D;
        int newBestIndex = 2147483647;

        double[] fitnessArr = fitness(population);
        double[][] objs = evaluate(population);

        /* normalize to allow prob cloning */
        for (double f: fitnessArr) {
            assert (f >= 0.0);
        }

        for(int finalRun = 0; finalRun < population.length; ++finalRun) {
            double f = fitnessArr[finalRun];
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
            population[worstIndex] = new KMeans(this.m_bestChromosome);
        }

        if(newBestFitness > this.m_bestFitness) {
            if(newBestFitness == 1.0D / 0.0) {
                KMeans var11 = new KMeans();
                var11.setSeed(this.m_rand.nextInt());
                var11.setInitializationMethod(KMeans.Initialization.HILL_CLIMBER);
                var11.setInitial(population[newBestIndex].getCentroids());
                var11.setMaxIterations(this.m_maxKMeansIterationsFinal);
                var11.buildClusterer(this.myData);
                this.m_builtClusterer = var11;
                this.m_numberOfClusters = this.m_builtClusterer.getCentroids().length;
            }

            this.m_bestChromosome = new KMeans(population[newBestIndex]);
            this.m_bestFitness = newBestFitness;
        }

        return population;
    }

    public int clusterInstance(Instance instance) throws Exception {
        return this.m_builtClusterer.clusterInstance(instance.toDoubleArray());
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
        return "Class implementing algorithm described in \"Combining K-Means and a Genetic Algorithm through a Novel Arrangement of Genetic Operators for High Quality Clustering\".\n\nDifferences to the original algorithm are: \n1. No use of VICUS similarity measure - standard EuclideanDistanceclass is used instead.\n2. Uses the basic missing value handling from SimpleKMeans.\n3. If an operation generates a chromosome where all records are assigned to a single cluster, chromosome will be mutated until at least 2 clusters are found.\n4. The starting generation for the chromosome selection operation is now modifiable, where in the original paperit was set to 11. The default is now after generation 50 (with the default number of generations being 60).\n\nFor more information see:" + this.getTechnicalInformation().toString();
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

    private class FitnessContainer implements Comparable<MyGenClustPlusPlus.FitnessContainer> {
        public void setFitness(double fitness) {
            this.fitness = fitness;
        }

        double fitness;
        KMeans clustering;

        FitnessContainer(double f, KMeans c) {
            this.fitness = f;
            this.clustering = c;
        }

        public int compareTo(MyGenClustPlusPlus.FitnessContainer other) {
            return Double.compare(this.fitness, other.fitness);
        }
    }
}