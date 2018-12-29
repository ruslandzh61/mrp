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
 * Created by rusland on 25.12.18.
 */
public class ExcelRW {
    private Workbook workbook;

    private static String[] columns = {"ari", "db", "silh", "k"};

    public static void write(String path, Experiment[] experiments, Dataset dataset) throws Exception {
        File file = new File(path);
        Workbook workbook;
        if (file.exists()) {
            workbook = HSSFWorkbookFactory.create(file);
        } else {
            workbook = new HSSFWorkbook();
        }
        // if sheet with this name exists
        int sheetIdx = workbook.getSheetIndex(dataset.name());
        if (sheetIdx >= 0) {
            workbook.removeSheetAt(sheetIdx);
        }
        Sheet sheet = workbook.createSheet(dataset.name());
        Row headerRow = sheet.createRow(0);
        for (int i = 0; i < columns.length; ++i) {
            Cell cell = headerRow.createCell(i);
            cell.setCellValue(columns[i]);
        }
        int rowIdx = 1;
        for (Experiment e: experiments) {
            Row row = sheet.createRow(rowIdx++);
            row.createCell(0).setCellValue(e.getAri());
            row.createCell(1).setCellValue(e.getDb());
            row.createCell(2).setCellValue(e.getSilh());
            row.createCell(3).setCellValue(e.getK());
        }

        // Resize all columns to fit the content size
        for(int i = 0; i < columns.length; i++) {
            sheet.autoSizeColumn(i);
        }

        // Write the output to a file
        FileOutputStream fileOut = new FileOutputStream(path);
        workbook.write(fileOut);
        fileOut.close();
    }

    public static void main(String[] args) throws Exception {
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
            ExcelRW.write(resultFilePath, datasetExperiments[i], datasets[i]);
        }*/
    }
}
