package GA;

/**
 * Configurations for GenClustPlusPlus algorithm
 * MaxiMin algorithm (Li, 2007):
 *      'Better Spread and Convergence: Particle Swarm Multiobjective Optimization Using the Maximin Fitness Function'
 */
enum  GaConfiguration {
    // original GenClust++
    GA(),

    // GenClust++ with MaxiMin (optimizes Connectivity and Cohesion) fitness function is DB Index
    mgaC5(10,20, true, MyGenClustPlusPlus.FITNESS.DBINDEX),
    mgaC6(10,60, true, MyGenClustPlusPlus.FITNESS.DBINDEX),

    // Multi-objective GenClust++ with MaxiMin; fitness function is a weighted sum of euclidean distances to utopia point (same weights)
    mgaC13(10,20, true, MyGenClustPlusPlus.FITNESS.MULTIOBJECTIVE_SUM),
    mgaC14(10,60, true, MyGenClustPlusPlus.FITNESS.MULTIOBJECTIVE_SUM),
    // Multi-objective GenClust++ without MaxiMin; fitness function is a weighted sum of euclidean distances to utopia point
    mgaC22(10,20, false, MyGenClustPlusPlus.FITNESS.MULTIOBJECTIVE_SUM),

    // Multi-objective GenClust++ with MaxiMin; fitness function - a weighted sum of euclidean distances to utopia point (different weights)
    mgaC26(10,20, true, MyGenClustPlusPlus.FITNESS.MULTIOBJECTIVE_SUM, new double[]{0.4,0.6}),
    mgaC27(10,20, true, MyGenClustPlusPlus.FITNESS.MULTIOBJECTIVE_SUM, new double[]{0.8,0.2}),
    mgaC28(10,20, true, MyGenClustPlusPlus.FITNESS.MULTIOBJECTIVE_SUM, new double[]{0.1,0.9}),
    mgaC29(10,20, true, MyGenClustPlusPlus.FITNESS.MULTIOBJECTIVE_SUM, new double[]{0.05,0.95});

    int chrSelectionGen;
    int generations;
    boolean maximin;
    MyGenClustPlusPlus.FITNESS fitness;
    double[] w;

    GaConfiguration(int chrSelectionGen, int generations, boolean maximin, MyGenClustPlusPlus.FITNESS fitness) {
        this(chrSelectionGen,generations,maximin, fitness,new double[]{0.5, 0.5});
    }

    GaConfiguration(int chrSelectionGen, int generations, boolean maximin,
                    MyGenClustPlusPlus.FITNESS fitness, double[] aW) {
        this.chrSelectionGen = chrSelectionGen;
        this.generations = generations;
        this.maximin = maximin;
        this.fitness = fitness;
        this.w = aW.clone();
    }

    GaConfiguration() {
    }
}
