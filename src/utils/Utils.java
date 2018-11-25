package utils;

import com.opencsv.CSVParser;
import com.opencsv.CSVParserBuilder;
import com.opencsv.CSVReader;
import com.opencsv.CSVReaderBuilder;
import weka.core.EuclideanDistance;

import javax.swing.*;
import javax.swing.filechooser.FileNameExtensionFilter;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
/**
 * Created by rusland on 28.10.18.
 */
public class Utils {

    public static String pickAFile() {
        JFileChooser chooser = new JFileChooser();
        FileNameExtensionFilter filter = new FileNameExtensionFilter(
                "JPG & GIF Images", "jpg", "gif");
        chooser.setFileFilter(filter);
        int returnVal = chooser.showOpenDialog(null);
        if(returnVal == JFileChooser.APPROVE_OPTION) {
            System.out.println("You chose to open this file: " +
                    chooser.getSelectedFile().getName());
        }
        return chooser.getSelectedFile().getAbsolutePath();
    }

    /**
    * taken from https://www.geeksforgeeks.org/reading-csv-file-java-using-opencv/
    * */
    public static List<String[]> readDataFromCustomSeperator(String file, char sep) throws IOException
    {
        // Create object of filereader
        // class with csv file as parameter.
        FileReader filereader = new FileReader(file);

        // create csvParser object with
        // custom seperator semi-colon
        CSVParser parser = new CSVParserBuilder().withSeparator(sep).build();

        // create csvReader object with parameter
        // filereader and parser
        CSVReader csvReader = new CSVReaderBuilder(filereader)
                .withCSVParser(parser)
                .build();

        // Read all data at once
        List<String[]> allData = csvReader.readAll();
        /*
        // print Data
        for (String[] row : allData) {
            for (String cell : row) {
                System.out.print(cell + "\t");
            }
            System.out.println();
        }*/

        return allData;
    }

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

    public static double roundAvoid(double value, int places) {
        double scale = Math.pow(10, places);
        return Math.round(value * scale) / scale;
    }

    public static int distinctNumberOfItems(int[] array) {
        if (array.length <= 1) {
            return array.length;
        }

        Set<Integer> set = new HashSet<Integer>();
        for (int i : array) {
            set.add(i);
        }
        return set.size();
    }

    public static Set<Integer> distinctItems(int[] array) {
        if (array.length <= 1) {
            return null;
        }

        Set<Integer> set = new HashSet<Integer>();
        for (int i : array) {
            set.add(i);
        }
        return set;
    }



    public static double[][] extractAttributes(List<String[]> data, int[] excludedColumns) {
        double[][] result = new double[data.size()][data.get(0).length-excludedColumns.length];
        for (int i = 0; i < result.length; ++i) {
            int ak = 0;
            for (int j = 0; j < result[0].length; ++j) {
                if (Arrays.binarySearch(excludedColumns,j) < 0) {
                    result[i][ak++] = Double.parseDouble(data.get(i)[j]);
                }
            }
        }
        return result;
    }

    public static void whenWriteStringUsingBufferedWritter_thenCorrect(String str, String fileName)
            throws IOException {
        BufferedWriter writer = new BufferedWriter(new FileWriter(fileName));
        writer.write(str);

        writer.close();
    }

    public static void nominalForm(String file) throws IOException {
        List<String[]> data = readDataFromCustomSeperator(file, ',');
        String res = "";
        for (String[] record: data) {
            record[record.length-1] = "class"+record[record.length-1];
            String s = Arrays.toString(record);
            res = res.concat(s.substring(1,s.length()-1)+System.getProperty("line.separator"));
        }

        whenWriteStringUsingBufferedWritter_thenCorrect(res, "data/output.csv");
    }

    public static void replaceInFile(String p, String repl, String with) throws IOException {
        Path path = Paths.get(p);
        Charset charset = StandardCharsets.UTF_8;

        String content = new String(Files.readAllBytes(path));
        content = content.replaceAll(repl, with);
        Files.write(path, content.getBytes(charset));
    }

    public static void nominalFormToNumber(String file, char sep, int attrIdx) throws IOException {
        HashMap<String, Integer> map = new HashMap<>();

        List<String[]> data = readDataFromCustomSeperator(file, sep);
        if (attrIdx == -1)
            attrIdx = data.get(0).length-1;
        String res = "";
        int i = 0;
        for (String[] record: data) {
            String s;
            if (!map.containsKey(record[attrIdx])) {
                map.put(record[attrIdx], i++);
            }
            record[attrIdx] = map.get(record[attrIdx]).toString();
            s = Arrays.toString(record);
            res = res.concat(s.substring(1,s.length()-1)+System.getProperty("line.separator"));
        }

        whenWriteStringUsingBufferedWritter_thenCorrect(res, "data/output.csv");
    }

    public static int[] extractLabels(List<String[]> dataStr, int col) {
        int[] labels = new int[dataStr.size()];
        for (int i = 0; i < dataStr.size(); ++i) {
            try {
                labels[i] = Integer.parseInt(dataStr.get(i)[col]);
            } catch (NumberFormatException e) {
                System.out.println(dataStr.get(i)[0]);
            }
        }
        //System.out.println(Arrays.toString(labels));
        return labels;
    }

    public static void checkClusterLabels(int[] sol, int k) {
        for (int i = 0; i < sol.length; ++i) {
            assert (sol[i] < k);
        }
    }
    /**
     * @k - number of clusters
     * @clusters - dim: N x D
     * @label - label of len N
     * return centroids: dimensionality: k x D
     */
    public static HashMap<Integer, double[]> centroids(double[][] clusters, int[] labels) {
        int N = clusters.length;
        int D = clusters[0].length;
        assert (labels.length==N);

        HashMap<Integer,double[]> newc = new HashMap<>(); //new centroids
        for (int label: labels) {
            newc.put(label, new double[D]);
        }
        HashMap<Integer, Integer> counts = new HashMap<>(); // sizes of the clusters

        for (int i=0; i<N; i++){
            for (int j=0; j<D; j++){
                newc.get(labels[i])[j] += clusters[i][j]; // update that centroid by adding the member data record
            }
            if (counts.containsKey(labels[i])) {
                counts.put(labels[i],counts.get(labels[i])+1);
            } else {
                counts.put(labels[i],1);
            }
        }

        // finally get the average
        for (int i: counts.keySet()) {
            for (int j=0; j<D; j++){
                newc.get(i)[j] /= counts.get(i);
            }
        }
        return newc;
    }

    public static double dbIndexScore(HashMap<Integer, double[]> clusters, int[] labels, double[][] data) {
        int numberOfClusters = clusters.size();
        double david = 0.0;
        HashMap<Integer, HashSet<Integer>> labelToClusterPoints = new HashMap<>();
        Set<Integer> distLabels = Utils.distinctItems(labels);
        for (int label: distLabels) {
            labelToClusterPoints.put(label, new HashSet<>());
        }
        for (int i = 0; i < labels.length; ++i) {
            labelToClusterPoints.get(labels[i]).add(i);
        }

        if (numberOfClusters == 1) {
            throw new RuntimeException(
                    "Impossible to evaluate Davies-Bouldin index over a single cluster");
        } else {
            // counting distances within
            HashMap<Integer, Double> withinClusterDistance = new HashMap<>();

            for (int clusterID: labelToClusterPoints.keySet()) {
                HashSet<Integer> cluster = labelToClusterPoints.get(clusterID);
                withinClusterDistance.put(clusterID, 0.0);
                for (int p: cluster) {
                    double[] punto = data[p];
                    withinClusterDistance.put(clusterID,
                            withinClusterDistance.get(clusterID)+Utils.dist(punto, clusters.get(clusterID)));
                }
                withinClusterDistance.put(clusterID,
                        withinClusterDistance.get(clusterID)/cluster.size());
            }


            double result = 0.0;
            double max = Double.NEGATIVE_INFINITY;

            try {
                for (int i: distLabels) {
                    //if the cluster is null
                    if (clusters.get(i) != null) {

                        for (int j = 0; j < numberOfClusters; j++)
                            //if the cluster is null
                            if (i != j && clusters.get(j) != null) {
                                double val = (withinClusterDistance.get(i) + withinClusterDistance.get(j))
                                        / Utils.dist(clusters.get(i), clusters.get(j));
                                if (val > max)
                                    max = val;
                            }
                    }
                    result = result + max;
                }
            } catch (Exception e) {
                System.out.println("Excepcion al calcular DAVID BOULDIN");
                e.printStackTrace();
            }
            david = result / numberOfClusters;
        }

        return david;
    }

    public static void main(String[] args) throws Exception {
        //replaceInFile("data/winequality-red.csv", ";",",");
        //replaceInFile("data/output.csv", " ","");
        //Utils.nominalFormToNumber("data/ld.csv", ',', -1);
        //Utils.nominalForm("data/p-winequality-red.csv");

        //test centroids
        /*double[][] data = {{1,1},{5,5},{10,10},{11,11}};
        int[] labels = {0,0,2,2};
        HashMap<Integer, double[]> centroids = Utils.centroids(data,labels);
        for (int c: centroids.keySet()) {
            System.out.println(Arrays.toString(centroids.get(c)));
        }*/
    }
}
