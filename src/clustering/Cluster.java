package clustering;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by rusland on 23.12.18.
 */
public class Cluster {
    private int id;
    private double[] centroid;
    private List<double[]> points;

    public Cluster(int id, double[] centroid) {
        this.id = id;
        this.centroid = centroid;
        points = new ArrayList<>();
    }

    public void add(double[] point) {
        points.add(point);
    }

    public List<double[]> getPoints() {
        return points;
    }

    public int id() {
        return id;
    }

    public int size() {
        return points.size();
    }
}