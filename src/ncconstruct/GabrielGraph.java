package ncconstruct;

import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Created by rusland on 09.10.18.
 */
public class GabrielGraph {
    private static final String NEWLINE = System.getProperty("line.separator");

    // Gabriel Graph is represented as an array of edge lists
    private List<Integer>[] adj;
    private int[][] density; // matrix
    private int V;

    /**
     * constructor accepts two dim array, where rows are data points and columns are dimensions
     * and build gabriel graph
     */
    public GabrielGraph(double[][] data) {
        this.V = data.length;
        System.out.println(V);
        adj = (LinkedList<Integer>[]) new LinkedList[V];
        for (int v = 0; v < V; v++) {
            adj[v] = new LinkedList<Integer>();
        }
        density = new int[V][V];
        build(data);
    }

    public int getV() {
        return this.V;
    }


    public double dist(double[] v,double[] w) {
        double sum = 0.0;
        for(int i=0;i<v.length;i++) {
            sum = sum + Math.pow((v[i]-w[i]),2.0);
        }
        return Math.sqrt(sum);
    }

    private void build(double[][] data) {
        int D = data[0].length; // # of dimensions
        for (int i = 0; i < V-1; ++i) {
            for (int j = i+1; j < V; ++j) {
                double[] center = new double[D]; // center point of disk with diamater point i<->point j
                for (int d = 0; d < D; ++d) {
                    center[d] = 0.5*(data[i][d] + data[j][d]);
                }
                double r = dist(center,data[i]); // radius of ball (half-distance between i and j)

                // check for each vertex that it's not in inside of i-j ball
                int p = 0;
                int denIJ = 0;
                while (p < V) {
                    if (p == i || p == j) {
                        ++p;
                        continue;
                    }
                    double dist = dist(data[p], center);
                    // if inside or on the edge
                    if ( dist <= r)
                        ++denIJ;
                    ++p;
                }
                if (denIJ == 0) {
                    addEdge(i,j);
                }
                density[i][j] = denIJ;
                density[j][i] = denIJ;

            }
        }
    }

    public void addEdge(int v, int w) {
        validateVertex(v);
        validateVertex(w);
        adj[v].add(w);
        adj[w].add(v);
    }

    public int degree(int v) {
        validateVertex(v);
        return adj[v].size();
    }

    public int density(int v, int w) {
        return density[v][w];
    }

    public String toString() {
        StringBuilder s = new StringBuilder();
        s.append(V + " vertices, " + NEWLINE);
        for (int v = 0; v < V; v++) {
            s.append(v + ": ");
            for (int w : adj[v]) {
                s.append(w + " ");
            }
            s.append(NEWLINE);
        }
        return s.toString();
    }

    public List<Integer> adj(int v) {
        validateVertex(v);
        return adj[v];
    }

    public void printDensityMatrix() {
        System.out.println("density matrix: ");
        for (int i = 0; i < V; ++i) {
            for (int j = 0; j < V; ++j) {
                System.out.print(density(i,j) + " ");
            }
            System.out.println();
        }
    }

    private void validateVertex(int v) {
        if (v < 0 || v >= V)
            throw new IllegalArgumentException("vertex " + v + " is not between 0 and " + (V-1));
    }

    public static void main(String[] args) {
        double[][] data = {{2,2}, {3,3}, {3,1}, {4,2}, {1.5,-1}, {2.5, -1.5}, {-2.5, 2}, {-1.5, 2}, {-2, 3}};
        GabrielGraph gg = new GabrielGraph(data);
        // gabriel graph where the following points are adjecent: 0<->1, 0->2, 1<->3, 2<->3
        System.out.println(gg);
        gg.printDensityMatrix();
        System.out.println("finished");
    }
}
