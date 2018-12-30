package clustering;

/**
 * Created by rusland on 20.12.18.
 */
public class Experiment {
    private int[] solution;
    private double ari, db, silh, k;

    public double getTime() {
        return time;
    }

    public void setTime(double time) {
        this.time = time;
    }

    private double time;

    public String getConfiguration() {
        return configuration;
    }

    public void setConfiguration(String configuration) {
        this.configuration = configuration;
    }

    private String configuration;

    public Experiment() {}

    public Experiment(int[] aSolution, double aAri, double aDb, double aSilh, double aK) {
        if (aSolution != null) {
            this.solution = aSolution.clone();
        } else {
            this.solution = null;
        }
        this.ari = aAri;
        this.db = aDb;
        this.silh = aSilh;
        this.k = aK;
    }

    public Experiment(Experiment e) {
        this(e.getSolution(), e.getAri(), e.getDb(), e.getSilh(), e.getK());
    }

    public Experiment clone() {
        return new Experiment(this);
    }

    public int[] getSolution() {
        return solution;
    }

    public double getAri() {
        return ari;
    }

    public double getDb() {
        return db;
    }

    public double getSilh() {
        return silh;
    }

    public double getK() {
        return k;
    }

    public void setSolution(int[] solution) {
        this.solution = solution.clone();
    }

    public void setAri(double ari) {
        this.ari = ari;
    }

    public void setDb(double db) {
        this.db = db;
    }

    public void setSilh(double silh) {
        this.silh = silh;
    }

    public void setK(double k) {
        this.k = k;
    }
}
