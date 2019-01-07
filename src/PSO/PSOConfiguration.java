package PSO;

/**
 * Created by rusland on 17.11.18.
 */
public enum PSOConfiguration {
    CONF2(200, true, true),
    CONF5(200, true, false, new double[]{0.05, 0.95}),
    CONF7(200, true, false),
    CONF8(200, true, true, new double[]{0.05, 0.95}),
    CONF10(200, true, true, new double[]{0.03, 0.97});

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

    public double c1 = 1.42;
    public double c2  = 1.63;
    public double maxW = 0.9;
    public double minW = 0.4;
    public int maxIteration = 200;
    public int maxIterWithoutImprovement = 50;
    public int maxK = 150;
    public int pMax = 150;
    public double numTopParticlesToPickForLeader = 0.2;
    public boolean equalClusterNumDistribution = true;
    public boolean maximin = true;
    public double[] weights = {0.5, 0.5};
}
