package utils;

import java.util.*;

/**
 * Created by rusland on 28.10.18.
 */
public class Utils {

    public static double dist(double[] v,double[] w) {
        double sum = 0.0;
        for(int i=0;i<v.length;i++) {
            sum = sum + Math.pow((v[i]-w[i]),2.0);
        }
        return Math.sqrt(sum);
    }

    public static double[][] deepCopy(double[][] a) {
        if (a == null) return null;
        if (a.length == 0) return new double[0][0];

        double[][] resArr = new double[a.length][a[0].length];
        for (int i = 0; i < a.length; ++i) {
            resArr[i] = a[i].clone();
        }
        return resArr;
    }

    public static double intersection(Set<Integer> ci, List<Integer> mj) {
        double intersect = 0;
        for (int p: mj) {
            if (ci.contains(p)) {
                ++intersect;
            }
        }
        return intersect;
    }

    public static double sum(double[] arr) {
        double res = 0;
        for (double el: arr) {
            res += el;
        }
        return res;
    }

    public static void main(String[] args) {
        Integer[] a = {0,1,2,3};
        Integer[] m = {0,1,2,3,4,5};
        Set<Integer> ci = new HashSet<>(Arrays.asList(a));
        System.out.println(intersection(ci, Arrays.asList(m)));
    }
}
