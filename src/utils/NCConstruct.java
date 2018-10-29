package utils;

import java.util.*;

/**
 * Created by rusland on 10.10.18.
 * NCConstruct implements neighborhood construction procedure
 */
public class NCConstruct {
    public enum Neighbor {
        CORE, DENSITY_CONNECTED, EXTENDED, FINAL,
    }
    class PointDistance implements Comparable<PointDistance> {
        private int point;
        private double dist;

        public PointDistance(int point, double dist) {
            this.point = point;
            this.dist = dist;
        }

        public int getPoint() {
            return point;
        }

        public double getDist() {
            return dist;
        }

        @Override
        public int compareTo(PointDistance other) {
            if(this.dist<other.dist)
                return -1;
            else if(other.dist<this.dist)
                return 1;
            return 0;
        }

        @Override
        public String toString() {
            return Integer.toString(this.point);
        }
    }

    /**edge between i and j in graph gg means there exists no point inside the disk with diameter ij
     * no edge between i and j in graph gg means there exists at least one point inside the disk with diameter ij */
    private GabrielGraph gg;
    private double[][] data;
    private PointDistance[][] orderedByDist; // indices ordered according to distance to a point
    /**CC contains indices of orderedByDist indicating closest indirect point,
     * all points before this point form core neighborhood */
    private int[] CC;
    /** BC store break point at which density decreases*/
    private int[] BC;
    private List<ArrayList<Integer>> extNeighbors, finalNeighbors;
    private int potentialNeighborsNum;

    /**
     * constructor takes data (2 dim array), build Gabriel Graph
    */
    public NCConstruct(double[][] aData) {
        this.data = Utils.deepCopy(aData);
        this.gg = new GabrielGraph(data);
        this.potentialNeighborsNum = gg.getV()-1;
        gg.printDensityMatrix();
        sortByDist();
        construct();
    }

    /**
     * id of cluster
     * returns a copied list of neighbors, big O(n) runtime */
    public ArrayList<Integer> neighbors(int id) {
        return new ArrayList<>(finalNeighbors.get(id));
    }

    private void construct() {
        extractCoreNeighbors();
        extractDensityConnectedNeighbors();
        extractExtendedNeighbors();
        extractFinalNeighbors();
    }

    /**
     * find points closer to given point i than closest indirect point
     */
    private void extractCoreNeighbors() {
        sortByDist();
        CC = new int[gg.getV()];
        for (int i = 0; i < gg.getV(); ++i) {
            int j = 0;
            // while point is direct and there are points
            while (j < potentialNeighborsNum &&
                    gg.density(i, orderedByDist[i][j].getPoint()) == 0) {
                ++j;
            }
            CC[i] = j;
            // j is indirect point to i
        }
    }
    private void extractDensityConnectedNeighbors() {
        // BC and CC stores indices to orderedByDist
        BC = new int[gg.getV()];
        // i is point
        for (int i = 0; i < gg.getV(); ++i) {
            // if all points are direct points
            if (CC[i] >= potentialNeighborsNum) {
                BC[i] = CC[i];
                continue;
            }

            int prevDen = gg.density(i, orderedByDist[i][CC[i]].getPoint());
            // j is index of CC
            int j = CC[i];
            while (j < potentialNeighborsNum) {
                // in the first iteration cur is next point after last point of nearest extNeighbors set,
                // can't be less than zero, that's why it's always included in density extNeighbors set
                int curDen = gg.density(i, orderedByDist[i][j].getPoint());
                // if it's break point at which density decreases
                if (curDen < prevDen) {
                    break;
                }
                prevDen = curDen;
                ++j;
            }
            BC[i] = j;
        }
    }
    private void extractExtendedNeighbors() {
        List<ArrayList<Integer>> densNeighbors = new ArrayList<>();
        extNeighbors = new ArrayList<>();
        // insert core and density-connected extNeighbors
        for (int i = 0; i < gg.getV(); ++i) {
            ArrayList<Integer> tmp = new ArrayList<>();
            ArrayList<Integer> densTmp = new ArrayList<>();

            for (int j = 0; j < BC[i]; ++j) {
                tmp.add(orderedByDist[i][j].getPoint());
                densTmp.add(orderedByDist[i][j].getPoint());
            }
            extNeighbors.add(tmp);
            densNeighbors.add(densTmp);
        }
        // add extended extNeighbors
        for (int i = 0; i < gg.getV(); ++i) {

            int j = BC[i]-1;
            if (j+1 >= this.potentialNeighborsNum-1) {
                continue;
            }
            do  {
                j++;

                // in the first iteration cur is break point and prev is last point of density extNeighbors set
                int curDen = gg.density(i, orderedByDist[i][j].getPoint());
                int prevDen = gg.density(i, orderedByDist[i][j-1].getPoint());
                if (curDen >= prevDen) {
                    extNeighbors.get(i).add(orderedByDist[i][j].getPoint());
                } else {
                    // check whether i and BC[i] have common extNeighbors, if they do they are indirectly connected
                    boolean intersected = false;

                    for (int iNeighbor: densNeighbors.get(i)) {
                        if (densNeighbors.get(orderedByDist[i][j].getPoint()).contains(iNeighbor)) {
                            intersected = true;
                            break;
                        }
                    }
                    // if intersected => add, else break
                    if (intersected) {
                        extNeighbors.get(i).add(orderedByDist[i][j].getPoint());
                    } else break;
                }
                // include points with increasing density into neighborhood and identify next break
                //j = nextBreakPoint(i, j);
            } while (j < this.potentialNeighborsNum-1);
        }
    }
    private void extractFinalNeighbors() {
        List<LinkedList<Integer>> coreNeighbors = new ArrayList<>();
        for (int i = 0; i < gg.getV(); ++i) {
            LinkedList<Integer> tmp = new LinkedList<>();
            for (int j = 0; j < CC[i]; ++j) {
                tmp.add(orderedByDist[i][j].getPoint());
            }
            coreNeighbors.add(tmp);
        }

        finalNeighbors = new ArrayList<>();
        // insert core and density-connected extNeighbors
        for (int i = 0; i < gg.getV(); ++i) {
            ArrayList<Integer> tmp = new ArrayList<>();
            tmp.addAll(extNeighbors.get(i));
            finalNeighbors.add(tmp);
        }
        int[] flag = new int[gg.getV()];
        do {
            for (int i = 0; i < gg.getV(); ++i) {
                int j = 0;
                flag[i] = 0;
                while (flag[i] != 1 && j < finalNeighbors.get(i).size()) {
                    if (finalNeighbors.get(finalNeighbors.get(i).get(j)).contains(i)) {
                        j++;
                        continue;
                    } else if (coreNeighbors.get(i).contains(finalNeighbors.get(i).get(j))) {
                        // by default assume no intersection occured
                        flag[i] = 1;
                        for (int coreN : coreNeighbors.get(i)) {
                            if (finalNeighbors.get(finalNeighbors.get(i).get(j)).contains(coreN)) {
                                flag[i] = 0;
                                break;
                            }
                        }
                        // if intersection didn't occur
                        if (flag[i] == 1) {
                            ArrayList<Integer> subL = new ArrayList<Integer>(finalNeighbors.get(i).subList(0, j));
                            finalNeighbors.set(i, subL);
                        }
                    } else {
                        flag[i] = 1;
                        for (int k = 0; k < j; ++k) {
                            if (finalNeighbors.get(finalNeighbors.get(i).get(j)).contains(finalNeighbors.get(i).get(k))) {
                                flag[i] = 0;
                                break;
                            }
                        }
                        // if intersection didn't occur
                        if (flag[i] == 1) {
                            ArrayList<Integer> subL = new ArrayList<Integer>(finalNeighbors.get(i).subList(0, j));
                            finalNeighbors.set(i, subL);
                        }
                    }
                    j++;
                }
            }
        } while (sum(flag) != 0);
    }

    private int sum(int[] arr) {
        int res = 0;
        for (int el: arr) {
            res += el;
        }
        return res;
    }


    /**
     * Given a point i as the base point,first,we list all remaining points in D in non-decreasing order of their
     * distance to point i,and we form the ordered set Ti
     */
    private void sortByDist() {
        //sort for each point according to distance to it
        orderedByDist = new PointDistance[gg.getV()][potentialNeighborsNum];
        for (int i = 0; i < gg.getV(); ++i) {
            int j = 0, p = 0;
            while (j < gg.getV()-1) {
                if (i == p) {
                    p++;
                    continue;
                }
                orderedByDist[i][j] = new PointDistance(p, Utils.dist(data[i], data[p]));
                ++p;
                ++j;
            }
        }
        for (int i = 0; i < gg.getV(); ++i) {
            Arrays.sort(orderedByDist[i]);
        }
    }

    public String toString() {
        StringBuilder s = new StringBuilder();
        for (int v = 0; v < gg.getV(); v++) {
            s.append(v + ": ");
            for (PointDistance w : orderedByDist[v]) {
                s.append(w + " ");
            }
            s.append(System.getProperty("line.separator"));
        }
        return s.toString();
    }

    public void printNeighbors(int p, Neighbor neighbor) {
        int pointOfInterest;
        if (neighbor == Neighbor.CORE) {
            pointOfInterest = CC[p];
            if (pointOfInterest == 0) System.out.println("no core");
            else System.out.print("core for " + p + ": ");
            for (int i = 0; i < pointOfInterest; ++i) {

                System.out.print(orderedByDist[p][i] + " ");
            }
            System.out.println();
        } else if (neighbor == Neighbor.DENSITY_CONNECTED) {
            pointOfInterest = BC[p];
            if (pointOfInterest == 0) System.out.println("no density");
            else System.out.print("density for " + p + ": ");
            for (int i = 0; i < pointOfInterest; ++i) {

                System.out.print(orderedByDist[p][i] + " ");
            }
            System.out.println();
        } else if (neighbor == Neighbor.EXTENDED) {
            System.out.print("extNeighbors for " + p + ": ");
            for (int i = 0; i < extNeighbors.get(p).size(); ++i) {
                System.out.print(extNeighbors.get(p).get(i) + " ");
            }
            System.out.println();
        } else if (neighbor == Neighbor.FINAL) {
            System.out.print("final for " + p + ": ");
            for (int i = 0; i < finalNeighbors.get(p).size(); ++i) {
                System.out.print(finalNeighbors.get(p).get(i) + " ");
            }
            System.out.println();
        }
        else {
            pointOfInterest = CC[p];
            if (pointOfInterest == 0) System.out.println("no extNeighbors");
            else System.out.print("extNeighbors for " + p + ": ");
            for (int i = 0; i < pointOfInterest; ++i) {

                System.out.print(orderedByDist[p][i] + " ");
            }
            System.out.println();
        }

    }

    public static void main(String[] args) {
        double[][] data = {{2,2}, {3,3}, {3,1}, {4,2}, {1.6,-0.5}, {3.01, -1.5}, {-4, 2}, {-2, 2}, {-3, 3},{7,7}};
        NCConstruct ncc = new NCConstruct(data);
        System.out.println(ncc); // prints ordered by distance points for each point
        for (int i = 0; i < data.length; ++i) {
            ncc.printNeighbors(i, Neighbor.CORE);
        }
        for (int i = 0; i < data.length; ++i) {
            ncc.printNeighbors(i, Neighbor.DENSITY_CONNECTED);
        }
        for (int i = 0; i < data.length; ++i) {
            ncc.printNeighbors(i, Neighbor.EXTENDED);
        }
        System.out.println();
        for (int i = 0; i < data.length; ++i) {
            ncc.printNeighbors(i, Neighbor.FINAL);
        }
    }
}
