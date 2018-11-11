package utils;

import java.util.*;

/**
 * Created by rusland on 10.11.18.
 * code is taken from
 * https://github.com/jcaucb/ParetoNondominated/blob/master/src/org/ucb/c5/pareto/Pareto.java
 */
public class Pareto {
    public Set<Integer> extractParetoNondominated(Map<Integer, double[]> nameToDatum) {
        //Create an array of the names in arbitrary order
        List<Integer> ranked = new ArrayList<>(nameToDatum.keySet());

        //Rank on a single arbitrarily chosen field
        Collections.sort(ranked, (o1, o2) -> {
            double[] datum1 = nameToDatum.get(o1);
            double[] datum2 = nameToDatum.get(o2);

            //See which score is higher
            double diff = datum2[0] - datum1[0];
            if(diff > 0) {
                return 1;
            }
            return -1;
        });

        /*Determine the non-dominated set:

        Starting with the highest-rank end of the List we just sorted,
        consider putting that datum in the non-dominated set.  If any
        preexisting members of the non-dominated set dominate the proposed
        datum, then it does not belong in the set, and the algorithm should
        continue to the next value.
        */
        Set<Integer> out = new HashSet<>();
        Outer: for(int name : ranked) {
            double[] datum1 = nameToDatum.get(name);
            for(Integer domName : out) {
                double[] domDatum = nameToDatum.get(domName);
                if(testDominance(domDatum, datum1)) {
                    continue Outer;
                }
            }
            out.add(name);
        }

        //Return the Pareto Non-dominated set
        return out;
    }

    public boolean testDominance(double[] domDatum, double[] datum1) {
        //If any of the individual scores of datum1 are higher than those of the domDatum, it is not dominated
        for(int i=0; i<datum1.length; i++) {
            double score1 = datum1[i];
            double domScore = domDatum[i];
            if(score1 > domScore) {
                return false;
            }
        }
        return true;
    }

    public static void main(String[] args) {
        double[][] values = new double[4][];
        values[0] = new double[]{3,-3};
        values[1] = new double[]{2,-2};
        values[2] = new double[]{1,-1};
        values[3] = new double[]{4,-4};
        Map<Integer,double[]> map = new HashMap<>();
        for (int i = 0; i < values.length; ++i) {
            map.put(i, values[i]);
        }
        Set<Integer> set = new Pareto().extractParetoNondominated(map);
        System.out.println(set);
    }
}
