package PSO;

import java.util.Random;

/**
 * Created by rusland on 15.11.18.
 * calculates velocity of particle
 */
public class VelocityCalculator {
    private Random generator;
    private double w,c1,c2;
    private static int default_seed = 10;
    private int seed;

    public VelocityCalculator(double aC1, double aC2) {
        this.c1 = aC1;
        this.c2 = aC2;
        setSeed(default_seed);
    }

    public void setW(double aW) {
        this.w = aW;
    }

    public double[] calculate(Particle p) {
        double r1 = generator.nextDouble();
        double r2 = generator.nextDouble();
        double[] newVel = new double[p.getVelocity().length];

        for (int dimIdx = 0; dimIdx < newVel.length; ++dimIdx) {
            newVel[dimIdx] = (w * p.getVelocity()[dimIdx]) + (r1 * c1) *
                    (-1 - p.getDummySolAt(dimIdx)) +
                    (r2 * c2) * (1 - p.getDummySolAt(dimIdx));
        }
        return newVel;
    }

    public int getSeed() {
        return seed;
    }

    public void setSeed(int aSeed) {
        generator = new Random(aSeed);
        this.seed = aSeed;
    }
}
