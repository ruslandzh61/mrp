package utils;

import clustering.Cluster;
import clustering.Dataset;
import clustering.Experiment;
import com.opencsv.CSVParser;
import com.opencsv.CSVParserBuilder;
import com.opencsv.CSVReader;
import com.opencsv.CSVReaderBuilder;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.Remove;

import javax.swing.*;
import javax.swing.filechooser.FileNameExtensionFilter;
import java.io.*;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

/**
 * Utility functions
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

    public static List<String[]> readFile(String file, char sep) throws IOException
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

        return csvReader.readAll();
    }

    public static double dist(double[] v,double[] w, double pow) {
        double sum = 0.0;
        for(int i = 0; i < v.length; i++) {
            sum += Math.pow(Math.abs(v[i]-w[i]),pow);
        }
        return Math.pow(sum, 1.0/pow);
    }

    public static double dist(double[] v,double[] w, double[] weights, double pow) {
        double sum = 0.0;
        for(int i = 0; i < v.length; i++) {
            sum += Math.pow(Math.abs(v[i]-w[i]),pow) * weights[i];
        }
        return Math.pow(sum, 1.0/pow);
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

    public static double sum(double[] arr, double pow) {
        double res = 0;
        for (double el: arr) {
            res += Math.pow(el, pow);
        }
        return res;
    }

    public static double sum(double[] arr, double[] w, double pow) {
        double res = 0;
        for (int i = 0; i < arr.length; ++i) {
            res += Math.pow(arr[i], pow) * w[i];
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


    public static void whenWriteStringUsingBufferedWritter_thenCorrect(String str, String fileName, boolean append)
            throws IOException {
        FileWriter fstream = new FileWriter(fileName, append);
        BufferedWriter writer = new BufferedWriter(fstream);
        writer.write(str);
        writer.close();
    }

    public static void nominalForm(String file, String output) throws IOException {
        List<String[]> data = readFile(file, ',');
        String res = "";
        for (String[] record: data) {
            record[record.length-1] = "class"+record[record.length-1];
            String s = Arrays.toString(record);
            res = res.concat(s.substring(1,s.length()-1)+System.getProperty("line.separator"));
        }

        whenWriteStringUsingBufferedWritter_thenCorrect(res, output, false);
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

        List<String[]> data = readFile(file, sep);
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

        whenWriteStringUsingBufferedWritter_thenCorrect(res, "data/output.csv", false);
    }

    public static int[] extractLabels(List<String[]> dataStr, int col) {
        assert (dataStr.size() > 0);
        int D = dataStr.get(0).length;
        assert (D > 0);
        assert (col >= 0 && col < D);
        for (String[] record: dataStr) {
            assert (record.length == D);
        }

        HashMap<String, Integer> mapNominalToNumeric = new HashMap<>();
        int[] labels = new int[dataStr.size()];
        int labelID = 1;
        for (int i = 0; i < dataStr.size(); ++i) {
            String label = dataStr.get(i)[col];
            if (!mapNominalToNumeric.containsKey(label)) {
                mapNominalToNumeric.put(label, labelID); //Integer.parseInt(dataStr.get(i)[col]);
                ++labelID;
            }
            labels[i] = mapNominalToNumeric.get(label);
        }

        return labels;
    }

    public static void checkClusterLabels(int[] sol, int k) {
        for (int i = 0; i < sol.length; ++i) {
            assert (sol[i] < k);
        }
    }

    public static HashMap<Integer, double[]> centroids(double[][] dataset, int[] labels) {
        int N = dataset.length;
        int D = dataset[0].length;
        assert (labels.length==N);

        HashMap<Integer,double[]> newc = new HashMap<>(); //new centroids
        for (int label: labels) {
            newc.put(label, new double[D]);
        }
        HashMap<Integer, Integer> counts = new HashMap<>(); // sizes of the clusters

        for (int i=0; i<N; i++){
            for (int j=0; j<D; j++){
                newc.get(labels[i])[j] += dataset[i][j]; // update that centroid by adding the member data record
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

    public static void normalize(double[][] data, double[] dataLow, double[] dataHigh) {
        assert (dataLow.length == dataHigh.length);
        assert (data[0].length == dataHigh.length);

        for (int i = 0; i < data.length; ++i) {
            for (int j = 0; j < data[0].length; ++j) {
                assert (dataLow[j] <= data[i][j]);
                assert (dataHigh[j] >= data[i][j]);
                if (dataHigh[j] == dataLow[j]) {
                    data[i][j] = 0.0;
                } else {
                    data[i][j] = (data[i][j] - dataLow[j])
                            / (dataHigh[j] - dataLow[j]);
                }
            }
        }
    }

    public static void normalize(double[] data) {
        double low = Double.POSITIVE_INFINITY;
        double high =Double.NEGATIVE_INFINITY;
        for (int i = 0; i < data.length; ++i) {
            if (data[i] > high) {
                high = data[i];
            }
            if (data[i] < low) {
                low = data[i];
            }
        }

        if (high == low) {
            for (int i = 0; i < data.length; ++i) {
                data[i] = 0.0;
            }
        } else {
            for (int i = 0; i < data.length; ++i) {
                data[i] = (data[i] - low) / (high - low);
            }
        }
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

        normalize(data, dataLow, dataHigh);
    }

    /**
     * Calculates Davies-Bauldin (DB) Index
     * @param clusters - clusters represented as HashpMap
     * @param labels - data points assignments produced by clustering algorithm
     * @param data - dataset without labels
     * @return DB Index score in range [0..1]
     */
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
        List<String[]> dataStr = Utils.readFile(filePath, sep);
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

    public static double[][] wekaInstancesToArray(Instances instances) {
        double[][] result = new double[instances.size()][];
        for (int i = 0; i < result.length; ++i) {
            result[i] = instances.get(i).toDoubleArray().clone();
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

    public static Instances getData(Dataset dataset) throws Exception {
        Remove filter;
        Instances data = ConverterUtils.DataSource.read(dataset.getPath());
        data.setClassIndex(data.numAttributes() - 1);

        /* remove first attribute */
        if (dataset.isRemoveFirst()) {
            filter = new Remove();
            filter.setAttributeIndices("1");
            filter.setInputFormat(data);
            data = Filter.useFilter(data, filter);
            data.setClassIndex(data.numAttributes() - 1);
        }

        /* normalize data if specified */
        if (dataset.isNormalize()) {
            Normalize normFilter = new Normalize();
            normFilter.setInputFormat(data);
            data = Filter.useFilter(data, normFilter);
            data.setClassIndex(data.numAttributes() - 1);
        }
        /* remove class attribute */
        filter = new Remove();
        filter.setAttributeIndices("" + data.numAttributes());
        filter.setInputFormat(data);
        data = Filter.useFilter(data, filter);

        return data;
    }

    /**
     * computes standard deviation of class
     * @param arr - array of values
     * @return value of standard deviation
     */
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

    /**
     * set ids of labels in range [1..K], K number of labels
     * @param labels - array of data point assignments produced by clustering algorithm
     */
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

    /**
     * determine pareto set using MaxiMin Strategy
     * @param objsList - vector of objectives for each solution
     * @return - MaxiMin function value for each solution
     */
    public static double[] determineParetoSet(double[][] objsList) {
        double[] maxiMins = new double[objsList.length];
        for (int i = 0; i < objsList.length; ++i) {
            double maxiMin = Double.NEGATIVE_INFINITY;
            double[] objI = objsList[i];
            double[] objJ;
            for (int j = 0; j < objsList.length; ++j) {
                if (i == j) continue;
                objJ = objsList[j];
                assert (objI.length == objJ.length);

                double min = Double.POSITIVE_INFINITY;
                for (int m = 0; m < objI.length; ++m) {
                    double curDiff = objI[m] - objJ[m];
                    if (curDiff < min) {
                        min = curDiff;
                    }
                }
                if (min > maxiMin) {
                    maxiMin = min;
                }
            }
            maxiMins[i] = maxiMin;
        }
        return maxiMins;
    }

    /**
     * pick solution closest to utopia point
     * @param objs - values of objectives for each solution
     * @param utopiaCoords - values of ideal objectives
     * @param weights - weights for each dimension distance
     * @param distMeasure - distance measure, e.g. Manhattan, Euclidean
     * @return index of closest solution
     */
    public static int pickClosestToUtopia(double[][] objs, double[] utopiaCoords, double[] weights, double distMeasure) {
        double minDist = Double.POSITIVE_INFINITY;
        int leader = -1;
        int i = 0;
        for (double[] cur: objs) {
            double distToUtopia = Utils.dist(cur, utopiaCoords, weights, distMeasure);
            if (distToUtopia < minDist) {
                leader = i;
                minDist = distToUtopia;
            }
            ++i;
        }
        assert (leader > -1);
        return leader;
    }

    /**
     * normalize vector
     * @param cur - vector to normalize
     * @param low - low bound
     * @param high - high bound
     */
    public static void normalize(double[] cur, double[] low, double[] high) {
        assert (low.length == cur.length);
        for (int i = 0; i < cur.length; ++i) {
            assert (low[i]<=cur[i]);
            assert (high[i]>=cur[i]);
        }

        for (int i = 0; i < cur.length; ++i) {
            if (low[i] == high[i]) {
                cur[i] = 0.0;
            } else {
                cur[i] = (cur[i] - low[i]) / (high[i] - low[i]);
            }
        }
    }

    /**
     * removes small clusters
     * @param labels - data point assignments
     * @param data - dataset without labels
     * @param minSizeOfCluster - minimum of size cluster
     * @param distMeasure
     */
    public static void removeNoise(int[] labels, double[][] data, int minSizeOfCluster, double distMeasure) {
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
                    double tmpDist = dist(centroids.get(c), data[i], distMeasure);
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

    /**
     * reduce size of dataset
     * @param dataset - dataset
     * @param inOrder - whether dataset is ordered by label value
     * @param portion - portion of dataset to use to generate new reduced in size dataset
     * @param randomly - pick data points randomly
     * @throws IOException
     */
    public static void reduceDataset(Dataset dataset, boolean inOrder, int portion, boolean randomly) throws IOException {
        // given that clusters are in order and cluster ids are in order
        List<String[]> fileArr = readFile(dataset.getPath(), ',');
        if (!inOrder) {
            HashMap<String, Integer> map = new HashMap<>();
            int i = 1;
            int attrIdx = fileArr.get(0).length - 1;
            for (String[] record : fileArr) {
                if (!map.containsKey(record[attrIdx])) {
                    map.put(record[attrIdx], i++);
                }
                record[attrIdx] = map.get(record[attrIdx]).toString();
            }
            //int[] labels = extractLabels(fileArr, fileArr.get(0).length-1);
            Collections.sort(fileArr, (o1, o2) -> Integer.compare(Integer.parseInt(o1[attrIdx]), Integer.parseInt(o2[attrIdx])));
        }

        int[] labels = extractLabels(fileArr, fileArr.get(0).length-1);

        List<Integer> labelsList = new ArrayList<>(labels.length);
        for (int i = 0; i < labels.length; ++i) {
            labelsList.add(labels[i]);
        }
        List<String[]> newData = new ArrayList<>(labels.length/2+1);
        Random rnd = new Random(1);
        for (int i = 1; i <= dataset.getK(); ++i) {
            HashSet<Integer> randomIdx = new HashSet<>();
            int first = labelsList.indexOf(i);
            int last = labelsList.lastIndexOf(i);
            if (randomly) {
                while (randomIdx.size() <= (last - first) / portion) {
                    randomIdx.add(rnd.nextInt(last - first + 1) + first);
                }
                for (int idx: randomIdx) {
                    newData.add(fileArr.get(idx));
                }
            } else {
                int mid = (last - first) / portion + first + 1;
                for (int j = first; j <= mid; ++j) {
                    newData.add(fileArr.get(j));
                }
            }
        }
        String res = "";
        for (String[] record: newData) {
            String s = Arrays.toString(record);
            res = res.concat(s.substring(1,s.length()-1)+System.getProperty("line.separator"));
        }

        whenWriteStringUsingBufferedWritter_thenCorrect(res, dataset.getPath().replace(".", "r" + portion + "."), false);
    }

    /**
     * read cluster solutions from file
     * @param filePath - folder path relative to the project root
     * @param r - number of runs
     * @param includesRuns - indicates whether txt file stores number of runs next name of corresponding dataset
     * @param includesTrueLabels - indicates whether true labels are stored in the file on the line after dataset name
     * @param includesTime - indicates whether file stores algorithm running time for each run
     * @param datasets - datasets
     * @return solutions in hashmap presentation, in which key is dataset name and value is array of solutions corresponding to dataset
     * @throws IOException
     */
    public static HashMap<String, int[][]> readSolutionFromFile(String filePath, int r, boolean includesRuns,
                                            boolean includesTrueLabels, boolean includesTime, Dataset[] datasets) throws IOException {
        HashMap<String, int[][]>  res = new HashMap<>();
        int runs = r;
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String datasetStr;
            int datasetIdx = 0;
            while ((datasetStr = br.readLine()) != null) {
                if (includesRuns) {
                    String[] datasetInfoLine = datasetStr.split(" ");
                    runs = Integer.parseInt(datasetInfoLine[1]);
                    datasetStr = datasetInfoLine[0];
                }
                assert (datasetStr.equals(datasets[datasetIdx]));

                if (includesTrueLabels) {
                    String labelsTrue = br.readLine();
                }

                int[][] solutions = new int[runs][];
                String time;
                for (int i = 0; i < runs; ++i) {
                    if (includesTime) {
                        time = br.readLine();
                    }
                    String solLine = br.readLine();
                    String[] expSols = solLine.substring(1, solLine.length()-1).split(", ");
                    solutions[i] = new int[expSols.length];
                    for (int j = 0; j < expSols.length; ++j) {
                        solutions[i][j] = Integer.parseInt(expSols[j]);
                    }
                }
                res.put(datasetStr, solutions);

                ++datasetIdx;
            }
        }
        return res;
    }

    private void printLabels(Dataset dataset) throws IOException {
        List<String[]> dataStr = Utils.readFile(dataset.getPath(), ',');
        if (dataset.getHeader() >= 0 && dataset.getHeader() < dataStr.size()) {
            dataStr.remove(dataset.getHeader());
        }

        // extract true labels
        int D = dataStr.get(0).length;
        int labelCol = D - 1;
        int[] labelsTrue = Utils.extractLabels(dataStr, labelCol);
        System.out.println(Arrays.toString(labelsTrue));
    }

    /**
     * Calculates sum of squared errors (SSE)
     * @param centroidList - centroids in double-array representation
     * @param labelList - data points assignments
     * @param data - dataset without labels
     * @return - SSE
     */
    public static double sse(double[][] centroidList, int[] labelList, double[][] data) {
        HashMap<Integer, double[]> mapC = new HashMap<>();
        for (int i = 0; i < centroidList.length; ++i) {
            mapC.put(i, centroidList[i]);
        }
        List<Cluster> clusters = transform(mapC, labelList, data);
        double sum = 0;
        for (Cluster cluster : clusters) {
            for (double[] point: cluster.getPoints()) {
                double dist = Utils.dist(cluster.getCentroid(), point, 2.0);
                sum += dist;
            }
        }
        return sum;
    }

    public static List<Cluster> transform(HashMap<Integer, double[]> aClusters, int[] aLabels, double[][] aData) {
        List<Cluster> clusters = new ArrayList<>(aClusters.size());
        HashMap<Integer, Integer> mapIDToArr = new HashMap<>();
        int idx = 0;
        for (int id: aClusters.keySet()) {
            clusters.add(new Cluster(id, aClusters.get(id)));
            mapIDToArr.put(id, idx);
            ++idx;
        }

        for (int i = 0; i < aData.length; ++i) {
            int label = aLabels[i];
            int indexToPut = mapIDToArr.get(label);
            clusters.get(indexToPut).add(aData[i]);
        }
        clusters.removeIf(cluster -> cluster.size() < 1);

        return clusters;
    }

    public static void experimentsToCsv(String path, Experiment[] es) throws IOException {
        StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append("ari,db,silh,k,kdiff\n");
        for (Experiment e : es) {
            stringBuilder.append(e.getAri() + ",");
            stringBuilder.append(e.getDb() + ",");
            stringBuilder.append(e.getSilh() + ",");
            stringBuilder.append(e.getK() + ",");
            stringBuilder.append(e.getKDiff() + "\n");
        }
        System.out.println(stringBuilder.toString());
        whenWriteStringUsingBufferedWritter_thenCorrect(stringBuilder.toString(), path, false);
    }

    public static void main(String[] args) throws IOException {
        Experiment e1 = new Experiment(null,1.0, 2.0,3.0,4,0.5);
        Experiment e2 = new Experiment(null,2.0, 1.0,3.0,4,0.2);
        Experiment[] es = {e1, e2};
        String path = "test.csv";
        experimentsToCsv(path, es);
    }
}
