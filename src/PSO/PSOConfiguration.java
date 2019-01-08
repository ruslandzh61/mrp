package PSO;

/**
 * Configuratios for PSO-based algorithm
 */
public enum PSOConfiguration {
    // MaxiMin is used, weights of objectives are the same (0.5, 0.5)
    CONF2(200, true, true),
    // MaxiMin is not used, weights of objectives are different (0.05, 0.95)
    CONF5(200, true, false, new double[]{0.05, 0.95}),
    // MaxiMin is not used, weights of objectives are same (0.5, 0.5)
    CONF7(200, true, false),
    // MaxiMin is used, weights of objectives are different (0.05, 0.95)
    CONF8(200, true, true, new double[]{0.05, 0.95});

    PSOConfiguration(int aMaxIteration, boolean aEqualClusterNumDistribution, boolean aMaximin) {
        this.maxIteration = aMaxIteration;
        this.equalClusterNumDistribution = aEqualClusterNumDistribution;
        this.maximin = aMaximin;
    }

    PSOConfiguration(int aMaxIteration, boolean aEqualClusterNumDistribution, boolean aMaximin, double[] aWeights) {
        this.maxIteration = aMaxIteration;
        this.equalClusterNumDistribution = aEqualClusterNumDistribution;
        this.maximin = aMaximin;
        this.weights = aWeights.clone();
    }

    double c1 = 1.42;
    double c2  = 1.63;
    double maxW = 0.9;
    double minW = 0.4;
    int maxIteration = 200;
    int maxIterWithoutImprovement = 50;
    // maxK depends on size of dataset: Math.sqrt(N), N is a number of data points
    int maxK = 150;
    int pMax = 150;
    // for randomly picking leaders (global best) for each particle
    double numTopParticlesToPickForLeader = 0.2;
    boolean equalClusterNumDistribution = true;
    boolean maximin = true;
    double[] weights = {0.5, 0.5};
}