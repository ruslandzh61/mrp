package utils;

import com.opencsv.CSVParser;
import com.opencsv.CSVParserBuilder;
import com.opencsv.CSVReader;
import com.opencsv.CSVReaderBuilder;
import weka.core.EuclideanDistance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.Remove;

import javax.swing.*;
import javax.swing.filechooser.FileNameExtensionFilter;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.math.BigDecimal;
import java.math.RoundingMode;
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

    public static double dist(double[] v,double[] w, double pow) {
        double sum = 0.0;
        for(int i=0;i<v.length;i++) {
            sum = sum + Math.pow((v[i]-w[i]),pow);
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

    public static void normalize(double[][] data) {
        double[] dataLow = new double[data[0].length];
        for (int i = 0; i < dataLow.length; ++i) {
            dataLow[i] = Double.POSITIVE_INFINITY;
        }
        double[] dataHigh = new double[data[0].length];
        for (int i = 0; i < dataLow.length; ++i) {
            dataHigh[i] = Double.NEGATIVE_INFINITY;
        }
        for (int i = 0; i < data.length; ++i) {
            for (int j = 0; j < data[0].length; ++j) {
                double tmp = data[i][j];
                if (tmp < dataLow[j]) {
                    dataLow[j] = tmp;
                }
                if (tmp > dataHigh[j]) {
                    dataHigh[j] = tmp;
                }
            }
        }
        for (int i = 0; i < data.length; ++i) {
            for (int j = 0; j < data[0].length; ++j) {
                data[i][j] = (data[i][j] - dataLow[j])
                        / (dataHigh[j] - dataLow[j]);
            }
        }
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

        /*if (numberOfClusters == 1) {
            throw new RuntimeException(
                    "Impossible to evaluate Davies-Bouldin index over a single cluster");
        }*/
        // counting distances within
        HashMap<Integer, Double> clustersDiameter = new HashMap<>();

        for (int clusterID: labelToClusterPoints.keySet()) {
            HashSet<Integer> cluster = labelToClusterPoints.get(clusterID);
            clustersDiameter.put(clusterID, 0.0);
            for (int p: cluster) {
                double[] punto = data[p];
                clustersDiameter.put(clusterID,
                        clustersDiameter.get(clusterID)+Utils.dist(punto, clusters.get(clusterID), 2));
            }
            clustersDiameter.put(clusterID,
                    clustersDiameter.get(clusterID)/cluster.size());
        }

        double result = 0.0;

        for (int i: distLabels) {
            //if the cluster is null
            if (clusters.get(i) != null) {
                double max = Double.NEGATIVE_INFINITY;
                for (int j = 0; j < numberOfClusters; j++) {
                    //if the cluster is null
                    if (i != j && clusters.get(j) != null) {
                        double val = (clustersDiameter.get(i) + clustersDiameter.get(j))
                                / Utils.dist(clusters.get(i), clusters.get(j), 2);
                        if (val > max)
                            max = val;
                    }
                }
                if (max != Double.NEGATIVE_INFINITY) {
                    result = result + max;
                }
            }
        }
        david = result / numberOfClusters;

        return david;
    }

    /*public static void measureFromFile(String filePath, char sep, double[][] data,) throws IOException {
        List<String[]> dataStr = Utils.readDataFromCustomSeperator(filePath, sep);
        String[] labelsStr = dataStr.get(0);
        int[] labelsTrue = new int[labelsStr.length];
        for (int i = 0; i < labelsTrue.length; ++i) {
            labelsTrue[i] = Integer.parseInt(labelsStr[i]);
        }
        int[] labelsPred;
        double meanDB = 0.0;
        for (int i = 1; i < dataStr.size(); ++i) {
            labelsStr = dataStr.get(i);
            labelsTrue = new int[labelsStr.length];
            for (int iA = 0; iA < labelsTrue.length; ++iA) {
                labelsTrue[iA] = Integer.parseInt(labelsStr[iA]);
            }
            meanDB += Utils.dbIndexScore()
        }
    }*/

    public static HashMap<Integer, double[]> centroidsFromWekaInstance(Instances instances) {
        HashMap<Integer, double[]> result = new HashMap<>();
        for (int i = 0; i < instances.numInstances(); ++i) {
            result.put(i,instances.get(i).toDoubleArray());
        }
        return result;
    }


    public static int[] adjustLabels(int[] labels) {
        HashMap<Integer, Integer> map = new HashMap<>();
        int[] result = new int[labels.length];
        int id = 0;
        for (int i = 0; i < labels.length; ++i) {
            int label = labels[i];
            if (!map.containsKey(label)) {
                map.put(label, id++);
            }
            result[i] = map.get(label);
        }
        return result;
    }

    public static Instances getData(String filePath, boolean removeFirst, boolean normalize) throws Exception {
        Remove filter;
        Instances data = ConverterUtils.DataSource.read(filePath);
        data.setClassIndex(data.numAttributes() - 1);
        if (removeFirst) {
            filter = new Remove();
            filter.setAttributeIndices("1");
            filter.setInputFormat(data);
            data = Filter.useFilter(data, filter);
            data.setClassIndex(data.numAttributes() - 1);
        }

        if (normalize) {
            Normalize normFilter = new Normalize();
            normFilter.setInputFormat(data);
            data = Filter.useFilter(data, normFilter);
            data.setClassIndex(data.numAttributes() - 1);
        }
        filter = new Remove();
        filter.setAttributeIndices("" + data.numAttributes());
        filter.setInputFormat(data);
        data = Filter.useFilter(data, filter);

        return data;
    }

    public static double standardDeviation(double arr[]) {
        double sum = 0.0;
        double standardDeviation = 0.0;

        for(double num : arr) {
            sum += num;
        }

        double mean = sum / arr.length;

        for(double num: arr) {
            standardDeviation += Math.pow(num - mean, 2);
        }

        return Math.sqrt(standardDeviation / arr.length);
    }

    public static void adjustAssignments(int[] labels) {
        HashMap<Integer, Integer> map = new HashMap();
        for (int i = 0; i < labels.length; ++i) {
            if (map.containsKey(labels[i])) {
                map.put(labels[i], map.get(labels[i])+1);
            } else {
                map.put(labels[i], 1);
            }
        }
        HashMap<Integer, Integer> map2 = new HashMap<>();
        int newLabel = 0;

        for (int cluser: map.keySet()) {
            map2.put(cluser, newLabel++);
            //System.out.println(cluser + " : " + map.get(cluser));
        }
        for (int i = 0; i < labels.length; ++i) {
            assert (map2.containsKey(labels[i]));
            labels[i] = map2.get(labels[i]);
        }
    }


    public static double[] normalize(double[] cur, double[] low, double[] high) {
        assert (low.length == cur.length);
        double[] normCur = cur.clone();
        for (int i = 0; i < normCur.length; ++i) {
            normCur[i] = (cur[i] - low[i]) / (high[i] - low[i]);
        }
        return normCur;
    }

    public static void removeNoise(int[] labels, double[][] data, int minSizeOfCluster, double pow) {
        Set<Integer> goodClusters = new HashSet<>();

        // count size of clusters
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < labels.length; ++i) {
            int clID = labels[i];
            if (map.containsKey(clID)) {
                map.put(clID, map.get(clID)+1);
            } else {
                map.put(clID, 1);
            }
        }

        // identify good clusters
        for (int clID: map.keySet()) {
            if (map.get(clID) >= minSizeOfCluster) {
                goodClusters.add(clID);
            }
        }

        //System.out.println(goodClusters.size());
        // remove bad clusters
        HashMap<Integer, double[]> centroids = Utils.centroids(data, labels);
        for (int i = 0; i < labels.length; ++i) {
            if (!goodClusters.contains(labels[i])) {
                double minDist = Double.POSITIVE_INFINITY;
                int targetC = -1;
                for (int c : centroids.keySet()) {
                    if (!goodClusters.contains(c)) {
                        continue;
                    }
                    double tmpDist = dist(centroids.get(c), data[i], pow);
                    if (minDist > tmpDist) {
                        minDist = tmpDist;
                        targetC = c;
                    }
                }
                labels[i] = targetC;
            }
        }
    }

    public static double doublePrecision(double toBeTruncated, int precision) {
        return BigDecimal.valueOf(toBeTruncated)
                .setScale(precision, RoundingMode.HALF_UP)
                .doubleValue();
    }

    public static void main(String[] args) throws Exception {
        //replaceInFile("data/winequality-red.csv", ";",",");
        //replaceInFile("data/p-flame.csv", " ","");
        //Utils.nominalFormToNumber("data/output.csv", ',', -1);
        //Utils.nominalForm("data/flame.csv");

        //test centroids
        /*double[][] data = {{1,1},{5,5},{10,10},{11,11}};
        int[] labels = {0,0,2,2};
        HashMap<Integer, double[]> centroids = Utils.centroids(data,labels);
        for (int c: centroids.keySet()) {
            System.out.println(Arrays.toString(centroids.get(c)));
        }*/
        /*int[] l = new int[] {0, 3, 2, 1};
        System.out.println(Arrays.toString(adjustLabels(l)));*/

        /*double[][] data = {{-1, 0.1}, {0.5, 0}, {1, 1}};
        normalize(data);
        for (int i = 0; i < data.length; ++i) {
            System.out.println(Arrays.toString(data[i]));
        }*/

        /*int[] labels = {1, 0, 5, 2, 2};
        //adjustAssignments(labels);
        System.out.println(Arrays.toString(labels));

        int distNumK = Utils.distinctNumberOfItems(labels);
        Integer[] distClusters = Utils.distinctItems(labels).toArray(new Integer[distNumK]);
        int idx = new Random().nextInt(distClusters.length);
        System.out.println(distClusters[idx]);*/
        /*double[][] data = {{1,1}, {1,2},{5,5},{5,5.5},{4,5},{1,0}};
        int[] label = {5,5,1,1,0,10};
        removeNoise(label, data, 2);
        System.out.println(Arrays.toString(label));*/

        //System.out.println(doublePrecision(10.3453453455, 5));
    }
}
