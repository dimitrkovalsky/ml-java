package com.keyprints;

import org.apache.commons.io.FileUtils;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.writable.Writable;
import org.datavec.image.loader.ImageLoader;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.List;

public class FacialKeyPrinter {
    private static String IMAGE_FOLDER = "C:\\Users\\Dmytro_Kovalskyi\\Documents\\jupyter projects\\dl fo cv\\week 2\\data\\images";
    private static String KEYPRINTS_CSV = "C:\\Users\\Dmytro_Kovalskyi\\Documents\\jupyter projects\\dl fo cv\\week 2\\data\\gt.csv";
    private static int height = 100;
    private static int width = 100;

    public static void main(String[] args) throws Exception {

        MultiLayerNetwork graph = KerasModelImport.importKerasSequentialModelAndWeights("C:\\gitlab\\ml\\ml-java\\models\\model49.h5");
        System.out.println(graph.summary());
        String imagePath = "C:\\gitlab\\ml\\ml-java\\datasets\\face2.jpeg";
        INDArray matrix = new NativeImageLoader(height, width, 1).asMatrix(new File(imagePath));
        INDArray dup = matrix.dup();
        normalizeImage(matrix);
        System.out.println(matrix);
        INDArray output = graph.output(matrix);

        double[] landmarks = output.toDoubleVector();
        List<Point> points = new ArrayList<>();
        for (int i = 0; i < landmarks.length; i += 2) {
            double valueX = denormalize(landmarks[i]);
            double valueY = denormalize(landmarks[i + 1]);
            System.out.println(valueX);
            System.out.println(valueY);
            points.add(new Point((int) valueX, (int) valueY));
        }

        showImage(imagePath, points, dup);
    }

    private static void showImage(String imagePath, List<Point> points, INDArray dup) throws IOException {
        BufferedImage bufferedImage = resize(ImageIO.read(new File((imagePath))), 100, 100);
        Graphics2D graphics = bufferedImage.createGraphics();
        graphics.setColor(Color.CYAN);
        for (Point point : points) {
            graphics.drawOval(point.x, point.y, 3, 3);
        }
        System.out.println(bufferedImage);
        JFrame frame = new JFrame();
        frame.getContentPane().setLayout(new FlowLayout());
        frame.getContentPane().add(new JLabel(new ImageIcon(bufferedImage)));
        frame.pack();
        frame.setVisible(true);
    }

    public static BufferedImage resize(BufferedImage img, int newW, int newH) {
        int w = img.getWidth();
        int h = img.getHeight();
        BufferedImage dimg = new BufferedImage(newW, newH, img.getType());
        Graphics2D g = dimg.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION,
                RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g.drawImage(img, 0, 0, newW, newH, 0, 0, w, h, null);
        g.dispose();
        return dimg;
    }

    private static double denormalize(double value) {

        double result = (value + 0.5) * width;
        if (result > width)
            result = width;
        if (result < 0)
            result = 0;
        return result;
    }

    private static void normalizeImage(final INDArray image) {
        ImagePreProcessingScaler preProcessor = new ImagePreProcessingScaler(-0.5, 0.5);
        preProcessor.transform(image);
    }

}
