package PSO;

/**
 * Created by rusland on 17.11.18.
 */
public enum PSOConfiguration {
    CONF1(200, false, true), // better than CONF2 based on preliminary parameter tuning experiments
    CONF2(200, true, true);

    PSOConfiguration(double c1, double c2, double maxW, double minW, int maxIteration, int maxIterWithoutImprovement,
                     int maxK, int pMax, double numTopParticlesToPickForLeader, boolean normObjs) {
        this.c1 = c1;
        this.c2 = c2;
        this.maxW = maxW;
        this.minW = minW;
        this.maxIteration = maxIteration;
        this.maxIterWithoutImprovement = maxIterWithoutImprovement;
        this.maxK = maxK;
        this.pMax = pMax;
        this.numTopParticlesToPickForLeader = numTopParticlesToPickForLeader;
        this.normObjs = normObjs;
    }

    PSOConfiguration(int aMaxIteration, boolean aNormObjs, boolean aEqualClusterNumDistribution) {
        this.maxIteration = aMaxIteration;
        this.normObjs = aNormObjs;
        this.equalClusterNumDistribution = aEqualClusterNumDistribution;
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
    public boolean normObjs = false;
    public boolean equalClusterNumDistribution = true;
}
