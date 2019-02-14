package com.classification;

import lombok.extern.slf4j.Slf4j;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.model.VGG16;
import org.deeplearning4j.zoo.util.imagenet.ImageNetLabels;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;

@Slf4j
public class ImageClassifier {
    private static final int HEIGHT = 224;
    private static final int WIDTH = 224;
    private static final int CHANNELS = 3;
    private ComputationGraph vgg16;
    private NativeImageLoader nativeImageLoader;

    public static void main(String[] args) throws IOException {
        new ImageClassifier().classify("C:\\Users\\Dmytro_Kovalskyi\\Downloads\\102318-dogs-color-determine-disesases-life.jpg");
    }

    public ImageClassifier() {
        try {
            vgg16 = (ComputationGraph) VGG16.builder().build().initPretrained(PretrainedType.IMAGENET);
        } catch (Exception e) {
            e.printStackTrace();
        }
        nativeImageLoader = new NativeImageLoader(HEIGHT, WIDTH, CHANNELS);
    }

    public String classify(String filePath) throws IOException {
        File file = new File(filePath);
        if (!file.exists()) {
            log.error("Image {} does not exists", filePath);
            return null;
        }

        return classify(new FileInputStream(file));
    }

    private String classify(InputStream inputStream) throws IOException {
        INDArray image = loadImage(inputStream);
        normalizeImage(image);
        INDArray output = processImage(image);
        return decodePredictions(output);
    }

    private INDArray processImage(final INDArray image) {
        INDArray[] output = vgg16.output(false, image);
        return output[0];
    }

    private INDArray loadImage(final InputStream inputStream) {
        INDArray image = null;
        try {
            image = nativeImageLoader.asMatrix(inputStream);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return image;
    }


    private void normalizeImage(final INDArray image) {
        DataNormalization scaler = new VGG16ImagePreProcessor();
        scaler.transform(image);
    }

    private String decodePredictions(INDArray encodedPredictions) throws IOException {
        String predictions = new ImageNetLabels().decodePredictions(encodedPredictions);

        return predictions;
    }
}