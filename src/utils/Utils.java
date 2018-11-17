package utils;

import com.opencsv.CSVParser;
import com.opencsv.CSVParserBuilder;
import com.opencsv.CSVReader;
import com.opencsv.CSVReaderBuilder;

import javax.swing.*;
import javax.swing.filechooser.FileNameExtensionFilter;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
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
    public static List<String[]> readDataFromCustomSeperator(String file, int skipLine,char sep) throws IOException
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

    public static int[] extractLabels(List<String[]> dataStr, int col) {
        int[] labels = new int[dataStr.size()];
        for (int i = 0; i < dataStr.size(); ++i) {
            labels[i] = Integer.parseInt(dataStr.get(i)[col]);
        }
        System.out.println(Arrays.toString(labels));
        return labels;
    }

    public static void checkClusterLabels(int[] sol, int k) {
        for (int i = 0; i < sol.length; ++i) {
            assert (sol[i] < k);
        }
    }
}
