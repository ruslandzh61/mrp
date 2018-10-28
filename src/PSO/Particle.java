package PSO;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by rusland on 09.09.18.
 */
public class Particle {
    private double fitness;
    private double[] velocity;
    private Solution solution;

    public double[] getVelocity() {
        return velocity;
    }

    public void setVelocity(double[] velocity) {
        this.velocity = velocity;
    }

    public Solution getSolution() {
        return solution;
    }

    public void setSolution(Solution solution) {
        this.solution = solution;
    }

    public void setFitness(double fitness) {
        this.fitness = fitness;
    }

    public double getFitness() {
        return fitness;
    }

    /*private Solution solution;
    private List<Double> velocity;
    private List<Double> continuosPosition; // converted representation, not discrete
    private Solution pBest;

    public Particle(Problem problem) {
    }*/
}

