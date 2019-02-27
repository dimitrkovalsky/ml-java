package com.imported;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;

public class EmotionModel {
    private static int height = 64;
    private static int width = 64;
    private String[] labels = {"angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"};

    public static void main(String[] args) throws InvalidKerasConfigurationException, IOException, UnsupportedKerasConfigurationException {
        ComputationGraph computationGraph = KerasModelImport.importKerasModelAndWeights("C:\\gitlab\\ml\\ml-java\\models\\fer2013_mini_XCEPTION.102-0.66.hdf5");
        System.out.println(computationGraph.summary());
        INDArray matrix = new NativeImageLoader(height, width, 1).asMatrix(new File("C:\\gitlab\\ml\\ml-java\\datasets\\face1.jpg"));
        ImagePreProcessingScaler preProcessor = new ImagePreProcessingScaler(0, 1);
        preProcessor.transform(matrix);
        System.out.println(Arrays.toString(computationGraph.output(matrix)));
    }
}
