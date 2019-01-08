package utils;

import clustering.Dataset;
import clustering.Experiment;
import org.apache.poi.hssf.usermodel.HSSFWorkbook;
import org.apache.poi.hssf.usermodel.HSSFWorkbookFactory;
import org.apache.poi.ss.usermodel.*;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Creates Excel files
 */
public class ExcelRW {
    private Workbook workbook;

    private static String[] columns = {"ari", "db", "silh", "k"};

    public static void write(String path, String[] datasetNames, String[][] table) throws IOException {
        File file = new File(path);
        FileOutputStream fileOut;
        Workbook workbook;
        if (file.exists()) {
            try {
                workbook = HSSFWorkbookFactory.create(file);
            } catch (Exception e) {
                file = new File(path.replace(".xls", "New.xls"));
                workbook = HSSFWorkbookFactory.create(file);
            }
            fileOut = new FileOutputStream(file);
        } else {
            workbook = new HSSFWorkbook();
            fileOut = new FileOutputStream(path);
        }

        Sheet sheet = workbook.createSheet();
        columns = new String[]{"Dataset","ARI", "DB", "Silhouette", "k"};
        Row headerRow = sheet.createRow(0);
        for (int i = 0; i < columns.length; ++i) {
            Cell cell = headerRow.createCell(i);
            cell.setCellValue(columns[i]);
        }

        int rowIdx = 1;
        for (String[] r: table) {
            Row row = sheet.createRow(rowIdx);
            row.createCell(0).setCellValue(datasetNames[rowIdx-1]);
            for (int i = 0; i < r.length; ++i) {
                row.createCell(i+1).setCellValue(r[i]);
            }
            ++rowIdx;
        }

        // Resize all columns to fit the content size
        for(int i = 0; i < columns.length; i++) {
            sheet.autoSizeColumn(i);
        }

        // Write the output to a file
        workbook.write(fileOut);
        fileOut.close();
    }

    public static void write(String path, Experiment[] experiments, String datasetName) throws Exception {
        File file = new File(path);
        FileOutputStream fileOut;
        Workbook workbook;
        if (file.exists()) {
            try {
                workbook = HSSFWorkbookFactory.create(file);
            } catch (Exception e) {
                file = new File(path.replace(".xls", "New.xls"));
                workbook = HSSFWorkbookFactory.create(file);
            }
            fileOut = new FileOutputStream(file);
        } else {
            workbook = new HSSFWorkbook();
            fileOut = new FileOutputStream(path);
        }
        // if sheet with this name exists
        int sheetIdx = workbook.getSheetIndex(datasetName);
        if (sheetIdx >= 0) {
            workbook.removeSheetAt(sheetIdx);
        }
        Sheet sheet = workbook.createSheet(datasetName);
        if (experiments[0].getConfiguration() != null) {
            columns = new String[]{"config", "ari", "db", "silh", "k"};
        }
        Row headerRow = sheet.createRow(0);
        for (int i = 0; i < columns.length; ++i) {
            Cell cell = headerRow.createCell(i);
            cell.setCellValue(columns[i]);
        }
        int rowIdx = 1;
        for (Experiment e: experiments) {
            Row row = sheet.createRow(rowIdx++);
            if (e.getConfiguration() == null) {
                row.createCell(0).setCellValue(e.getAri());
                row.createCell(1).setCellValue(e.getDb());
                row.createCell(2).setCellValue(e.getSilh());
                row.createCell(3).setCellValue(e.getK());
            } else {
                row.createCell(0).setCellValue(e.getConfiguration());
                row.createCell(1).setCellValue(e.getAri());
                row.createCell(2).setCellValue(e.getDb());
                row.createCell(3).setCellValue(e.getSilh());
                row.createCell(4).setCellValue(e.getK());
            }
        }

        // Resize all columns to fit the content size
        for(int i = 0; i < columns.length; i++) {
            sheet.autoSizeColumn(i);
        }

        // Write the output to a file
        workbook.write(fileOut);
        fileOut.close();
    }

    public static void main(String[] args) throws Exception {
        // tests
        String[] datasets = {Dataset.GLASS.name(), Dataset.WDBC.name(), Dataset.FLAME.name(), Dataset.COMPOUND.name(),
                Dataset.PATHBASED.name(), Dataset.S1.name(), Dataset.S3.name(), Dataset.DIM064.name(), Dataset.DIM256.name()};
        String[][] arrStr = {{"45+-4", "40+-6", "45+-4", "7"},{"55+-6", "40+-3", "35+-4","5"},{"30+-4", "45+-9", "35+-7", "5"}};
        ExcelRW.write("test.xls", datasets, arrStr);

        /*Experiment[] experiments1 = {
                new Experiment(new int[]{1,1,2,4}, 3.2, 2.3, 0.5, 6),
                new Experiment(new int[]{2,1,4,2}, 3.4, 3, 0.6, 4)
        };
        Experiment[] experiments2 = {
                new Experiment(new int[]{1,2,3}, 4, 2, 0.55, 2),
                new Experiment(new int[]{2,4,2}, 4, 3, 0.65, 3)
        };
        Dataset[] datasets = {Dataset.GLASS, Dataset.FLAME};
        Experiment[][] datasetExperiments = new Experiment[datasets.length][];
        datasetExperiments[0] = experiments1;
        datasetExperiments[1] = experiments2;
        String resultFilePath = "results/test1.xls";
        String solutionsFilePath = "results/test1.txt";
        StringBuilder solutionsLog = new StringBuilder();

        List<int[]> labelsTrueList = new ArrayList<>(datasets.length);
        labelsTrueList.add(new int[]{1,2});
        labelsTrueList.add(new int[]{2,3,3});
        for (int i = 0; i < datasets.length; ++i) {
            solutionsLog.append(datasets[i].name() + System.lineSeparator() +
                    Arrays.toString(labelsTrueList.get(i)) + System.lineSeparator());
            for (int j = 0; j < datasetExperiments[i].length; ++j) {
                solutionsLog.append(Arrays.toString(datasetExperiments[i][j].getSolution()) + System.lineSeparator());
            }
        }
        Utils.whenWriteStringUsingBufferedWritter_thenCorrect(solutionsLog.toString(), solutionsFilePath, true);
        for (int i = 0; i < datasets.length; ++i) {
            ExcelRW.write(resultFilePath, datasetExperiments[i], datasets[i].name());
        }*/
    }
}
