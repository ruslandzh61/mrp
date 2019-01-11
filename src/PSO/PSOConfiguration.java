package PSO;

/**
 * Configuratios for PSO-based algorithm
 */
public enum PSOConfiguration {
    CONF1(200, false, true, true),
    // MaxiMin is used, weights of objectives are the same (0.5, 0.5)
    CONF2(200, true, true, true),
    // MaxiMin is not used, weights of objectives are different
    CONF4(200, true, true, false, new double[]{0.15, 0.85}),
    CONF5(200, true, true, false, new double[]{0.05, 0.95}),
    // MaxiMin is not used, weights of objectives are same (0.5, 0.5)
    CONF7(200, true, true, false),
    // MaxiMin is used, weights of objectives are different
    CONF8(200, true, true, true, new double[]{0.05, 0.95});

    PSOConfiguration(int aMaxIteration, boolean aNormObjs, boolean aEqualClusterNumDistribution, boolean aMaximin) {
        this(aMaxIteration, aNormObjs, aEqualClusterNumDistribution, aMaximin, new double[]{0.5, 0.5});
    }

    PSOConfiguration(int aMaxIteration, boolean aNormObjs, boolean aEqualClusterNumDistribution, boolean aMaximin, double[] aWeights) {
        this.maxIteration = aMaxIteration;
        this.normObjs = aNormObjs;
        this.equalClusterNumDistribution = aEqualClusterNumDistribution;
        this.maximin = aMaximin;
        this.weights = aWeights.clone();
    }

    double c1 = 1.42;
    double c2  = 1.63;
    double maxW = 0.9;
    double minW = 0.4;
    int maxIteration = 200;
    int maxIterWithoutImprovement = 25;
    // maxK depends on size of dataset: Math.sqrt(N), N is a number of data points
    int maxK = 150;
    int pMax = 150;
    // for randomly picking leaders (global best) for each particle
    double numTopParticlesToPickForLeader = 0.2;
    boolean equalClusterNumDistribution = true;
    boolean maximin = true;
    boolean normObjs;
    double[] weights = {0.5, 0.5};
}