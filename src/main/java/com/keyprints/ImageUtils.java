package com.keyprints;

import javafx.scene.image.Image;
import javafx.scene.image.PixelReader;
import javafx.scene.image.PixelWriter;
import javafx.scene.image.WritableImage;
import javafx.scene.paint.Color;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Random;

public final class ImageUtils {
    public static final Random RANDOM = new Random();

    private ImageUtils() {
        // This is intentionally empty
    }

    public static Color randomColor() {
        return Color.color(RANDOM.nextDouble(), RANDOM.nextDouble(), RANDOM.nextDouble());
    }

    public static WritableImage emptyImage(Color color, int width, int height) {
        WritableImage writableImage
                = new WritableImage(width, height);
        PixelWriter pixelWriter = writableImage.getPixelWriter();

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                pixelWriter.setColor(x, y, color);
            }
        }

        return writableImage;
    }

    public static WritableImage drawImage(INDArray data, int width, int height) {
        WritableImage writableImage
                = new WritableImage(width, height);
        PixelWriter pixelWriter = writableImage.getPixelWriter();

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int index = y * width + x;
                int r = (int) trimToRange0to1(data.getDouble(index, 0));
//                double g = trimToRange0to1(data.getDouble(index, 1));
//                double b = trimToRange0to1(data.getDouble(index, 2));
//                Color color = Color.color(r, g, b);
                pixelWriter.setArgb(x, y, r);
            }
        }

        return writableImage;
    }

    private static double trimToRange0to1(double value) {
        return trimToRange(value, 0.0, 1.0);
    }

    private static double trimToRange(double value, double min, double max) {
        return Math.max(Math.min(value, max), min);
    }

    public static DataSet convertToDataSet(Image inputImage, Image expectedImage) {
        if (inputImage.getWidth() != expectedImage.getWidth() ||
                inputImage.getHeight() != expectedImage.getHeight()) {
            throw new RuntimeException("Input and expected images have different size");
        }

        PixelReader inputPR = inputImage.getPixelReader();
        PixelReader expectedPR = expectedImage.getPixelReader();

        int width = (int) inputImage.getWidth();
        int height = (int) inputImage.getHeight();

        int pixelsNo = width * height;

        INDArray features = Nd4j.zeros(pixelsNo, 5); //x,y,r,g,b
        INDArray labels = Nd4j.zeros(pixelsNo, 3);//r,g,b

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int index = y * width + x;
                Color inputColor = inputPR.getColor(x, y);
                Color expectedColor = expectedPR.getColor(x, y);
                double fX = scale(x, width);
                double fY = scale(y, height);
                double fCr = scaleColor(inputColor.getRed());
                double fCg = scaleColor(inputColor.getGreen());
                double fCb = scaleColor(inputColor.getBlue());

                double lCr = scaleColor(expectedColor.getRed());
                double lCg = scaleColor(expectedColor.getGreen());
                double lCb = scaleColor(expectedColor.getBlue());


                features.put(index, 0, fX);
                features.put(index, 1, fY);
                features.put(index, 2, fCr);
                features.put(index, 3, fCg);
                features.put(index, 4, fCb);

                labels.put(index, 0, lCr);
                labels.put(index, 1, lCg);
                labels.put(index, 2, lCb);
            }
        }

        return new DataSet(features, labels);
    }

    private static double scaleColor(double value) {
        return value;
    }

    private static double scale(int value, int rangeSize) {
        return scale(value, rangeSize, 1.0d);
    }

    private static double scale(int value, int rangeSize, double targetRange) {
        return (targetRange / (double) rangeSize) * ((double) value) - targetRange * 0.5;
    }
}
