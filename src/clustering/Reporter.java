package clustering;

import utils.Utils;

import java.util.LinkedList;
import java.util.List;

/**
 * Created by rusland on 20.12.18.
 */
public class Reporter {
    private Experiment[] experiments;

    private Experiment mean;
    private Experiment stdDev;
    private boolean computed;

    /**
    * @r - number of experiments
    * */
    public Reporter(int numExperiments) {
        this.experiments = new Experiment[numExperiments];
        this.computed = false;
    }

    public void set(int experimentID, Experiment experiment) {
        experiments[experimentID] = experiment.clone();
        experiments[experimentID].setTime(experiment.getTime());
        this.computed = false;
    }

    public Experiment get(int idx) {
        return experiments[idx];
    }

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

    public Experiment getMean() {
        assert (this.computed == true);
        return mean;
    }

    public Experiment getStdDev() {
        assert (this.computed == true);
        return stdDev;
    }

    public Experiment[] getExperiments() {
        return experiments;
    }
}
