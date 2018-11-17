package utils;

import java.util.*;

/**
 * Created by rusland on 10.11.18.
 * Pareto implements simple procedure for finding set of pareto-optimal front;
 * should be substituted with MaxiMin algorithm (Li, 2007):
 *      'Better Spread and Convergence: Particle Swarm Multiobjective Optimization Using the Maximin Fitness Function'
 *
 * some code is taken from
 * https://github.com/jcaucb/ParetoNondominated/blob/master/src/org/ucb/c5/pareto/Pareto.java
 */
public class Pareto {
    public static int PRECISION = 8;
    public static double THRESHOLD = Math.pow(0.1, PRECISION);

    public Set<Integer> extractParetoNondominated(Map<Integer, double[]> data) {
        //Create an array of the names in arbitrary order
        List<Integer> ranked = new ArrayList<>(data.keySet());

        //Rank on a first objective, if equal rank on the next one and so on,
        // return either more or less to prevent identical members in pareto set
        Collections.sort(ranked, (o1, o2) -> {
            double[] p1 = data.get(o1);
            double[] p2 = data.get(o2);
            int objNum = 0;
            double diff;
            do {
                diff = Utils.roundAvoid(p2[objNum], PRECISION) - Utils.roundAvoid(p1[objNum], PRECISION);
                if (diff > THRESHOLD) {
                    return 1;
                } else if (-diff > THRESHOLD) { // diff is negative
                    //System.out.println(o1 + " > " + o2 + " on " + objNum + ": " + diff);
                    return -1;
                }
                objNum++;
                //System.out.println(o1 + " <> " + o2);
            } while (p1.length>objNum); // Math.abs(diff) < THRESHOLD is already true at this point

            return -1;
        });

        /*Determine the non-dominated set:
            Starting with the highest-rank end of the List we just sorted,
            consider putting a member in the non-dominated set.  If any
            preexisting members of the non-dominated set dominate current member,
            then it does not belong in the set, and the algorithm should
            continue to the next member after current one.
        */
        Set<Integer> pareto = new HashSet<>();
        Outer: for(int id : ranked) {
            double[] p = data.get(id);
            for(Integer idOut : pareto) {
                double[] pTmp = data.get(idOut);
                if(testDominance(pTmp, p)) {
                    continue Outer;
                }
            }
            pareto.add(id);
        }

        //Return the Pareto Non-dominated set
        return pareto;
    }

    public boolean testDominance(double[] p1, double[] p2) {
        assert (p1.length==p2.length);
        int equalObjNum = 0;
        //If any of the individual scores of datum1 are higher than those of the domDatum, it is not dominated
        for(int i=0; i<p2.length; i++) {
            double p1Score = Utils.roundAvoid(p1[i], PRECISION);
            double p2Score = Utils.roundAvoid(p2[i], PRECISION);
            if(p2Score > p1Score) {
                return false;
            } else if (Math.abs(p2Score-p1Score) < THRESHOLD) {
                equalObjNum++;
            }
        }

        if (equalObjNum==p1.length) {
            return false;
        }
        return true;
    }

    public static void main(String[] args) {
        double[][] values = new double[6][];
        values[0] = new double[]{0.2, 0.17590848575069545};
        values[1] = new double[]{0.2, 0.22910852642658405};
        values[2] = new double[]{0.2, 0.19483310610576984};
        values[3] = new double[]{0.14285714285714285, 0.23748606942878325};
        values[4] = new double[]{0.14285714285714288, 0.24109386296189347};
        values[5] = new double[]{0.125, 0.28223727678569877};
        Map<Integer,double[]> map = new HashMap<>();
        for (int i = 0; i < values.length; ++i) {
            map.put(i, values[i]);
        }
        Set<Integer> set = new Pareto().extractParetoNondominated(map);
        System.out.println(set);
    }
}
