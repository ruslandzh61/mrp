package clustering;

import utils.Utils;

import java.util.LinkedList;
import java.util.List;

/**
 * Represents Experimental results
 * Computes performance measures, mean and standard deviation
 */
public class Reporter {
    private Experiment[] experiments;

    private Experiment mean;
    private Experiment stdDev;
    private boolean computed;

    /**
    * @param numExperiments  - number of experiments
    * */
    public Reporter(int numExperiments) {
        this.experiments = new Experiment[numExperiments];
        this.computed = false;
    }

    /**
     * set experimental results for a specified run
     * @param experimentID index of experiment
     * @param experiment experimental result
     */
    public void set(int experimentID, Experiment experiment) {
        experiments[experimentID] = experiment.clone();
        experiments[experimentID].setTime(experiment.getTime());
        this.computed = false;
    }

    /**
     * @param idx index of run
     * @return experimental results for a specified run
     */
    public Experiment get(int idx) {
        return experiments[idx];
    }

    /**
     * compute performace measures
     */
    public void compute() {
        double[] aris = new double[experiments.length];
        double[] dbs = new double[experiments.length];
        double[] silhs = new double[experiments.length];
        double[] ks = new double[experiments.length];

        for (int i = 0; i < experiments.length; ++i) {
            aris[i] = experiments[i].getAri();
            dbs[i] = experiments[i].getDb();
            silhs[i] = experiments[i].getSilh();
            ks[i] = experiments[i].getK();
        }

        this.stdDev = new Experiment();
        this.stdDev.setAri(Utils.standardDeviation(aris));
        this.stdDev.setDb(Utils.standardDeviation(dbs));
        this.stdDev.setSilh(Utils.standardDeviation(silhs));
        this.stdDev.setK(Utils.standardDeviation(ks));

        this.mean = new Experiment();
        this.mean.setAri(Utils.sum(aris, 1.0)/experiments.length);
        this.mean.setDb(Utils.sum(dbs, 1.0)/experiments.length);
        this.mean.setSilh(Utils.sum(silhs, 1.0)/experiments.length);
        this.mean.setK(Utils.sum(ks, 1.0)/experiments.length);

        this.computed = true;
    }

    public int size() {
        return this.experiments.length;
    }

    /**
     * @return mean results of experiments
     */
    public Experiment getMean() {
        assert (this.computed == true);
        return mean;
    }

    /**
     * @return standard deviation results of experiments
     */
    public Experiment getStdDev() {
        assert (this.computed == true);
        return stdDev;
    }
}
