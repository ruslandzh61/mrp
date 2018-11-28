package PSO;

/**
 * Created by rusland on 17.11.18.
 */
public class PSOConfiguration {
    public double c1 = 1.42;
    public double c2  = 1.63;
    public double maxW = 0.9;
    public double minW = 0.4;
    public int maxIteration = 100;
    public int maxIterWithoutImprovement = 50;
    public int maxK = 10;
    public int pMax = 150;
    public boolean pickLeaderRandomly = false;
}
