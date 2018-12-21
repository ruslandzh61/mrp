package test;

import utils.Utils;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Created by rusland on 20.12.18.
 */
public class UtilsTesting {

    private static void testReadFile() throws IOException {
        /* if there is a header, remove it */
        int header = 0;
        String path = "data/weka/glass.csv";
        List<String[]> dataStr = Utils.readFile(path, ',');
        for (String[] record: dataStr) {
            System.out.println(Arrays.toString(record));
        }
    }

    private static void testReadExtract() throws IOException {
        int header = 0;
        String path = "data/weka/glass.csv";
        List<String[]> dataStr = Utils.readFile(path, ',');
        if (header >= 0 && header < dataStr.size()) {
            dataStr.remove(header);
        }
        int D = dataStr.get(0).length;
        int labelCol = D - 1;
        int[] labels = Utils.extractLabels(dataStr, labelCol);
        System.out.println(Arrays.toString(labels));
        double[][] dataAttr = Utils.extractAttributes(dataStr, new int[]{labelCol});
        for (double[] record: dataAttr) {
            System.out.println(Arrays.toString(record));
        }
    }

    private static void testExtractLabels() {
        List<String[]> dataStr = new ArrayList<>();
        String[] record1 = {"1.1", "2.4", "class5"};
        String[] record2 = {"2", "2.8", "class5"};
        String[] record3 = {"-1", "2.2", "class1"};
        String[] record4 = {"2", "2.8", "class5"};
        dataStr.add(record1);
        dataStr.add(record2);
        dataStr.add(record3);
        dataStr.add(record4);
        int[] labels = Utils.extractLabels(dataStr, dataStr.get(0).length-1);
        System.out.println(Arrays.toString(labels));
    }

    public static void main(String[] args) throws Exception {
        testReadExtract();
    }
}
