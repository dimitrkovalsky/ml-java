package com.expessions;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FileUtils;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.writable.IntWritable;
import org.datavec.api.writable.Writable;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.ColorConversionTransform;
import org.datavec.image.transform.FlipImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.RotateImageTransform;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.jetbrains.annotations.NotNull;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.preprocessor.ImageFlatteningDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.*;

import static com.classification.AnimalsClassification.*;

@Slf4j
public class EmotionClassifier {
    protected static int height = 100;
    protected static int width = 100;
    protected static int channels = 1;
    protected static int batchSize = 32;
    protected static long seed = 42;
    protected static Random rng = new Random(seed);
    protected static int epochs = 10;
    protected static int cycles = 2000;
    private static int numLabels = 8;

    public static void main(String[] args) throws IOException, InterruptedException {
        MultiLayerNetwork network = initConvModel("emotion-rnn.data");
        System.out.println(network.summary());
        //    preProcessImages();
        trainModel(network);
        evaluateModel(network);
    }

    @NotNull
    private static void trainModel(MultiLayerNetwork network) throws IOException {
        log.info("Train model....");
        ImageTransform flipTransform1 = new FlipImageTransform(new Random());
        ImageTransform flipTransform2 = new FlipImageTransform(new Random(123));
        ImageTransform resizeTransform2 = new RotateImageTransform(30);
        ImageTransform resizeTransform3 = new ColorConversionTransform(1);

        List<ImageTransform> pipeline = Arrays.asList(new ImageTransform[] {flipTransform1, flipTransform2, resizeTransform2, resizeTransform3});
        File trainDirectory = new File("C:\\gitlab\\facial_expressions\\processed");
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);
        FileSplit trainSplit = new FileSplit(trainDirectory, new String[] {"jpg"});
        recordReader.initialize(trainSplit, null);
        RecordReaderDataSetIterator trainDataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
        trainDataIter.setPreProcessor(new ImageFlatteningDataSetPreProcessor());
        trainDataIter.setPreProcessor(new ImagePreProcessingScaler(0, 1));
        network.fit(trainDataIter, epochs);
//        }
    }

    private static void evaluateModel(MultiLayerNetwork network) throws IOException {
        log.info("Evaluate model....");
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker, null);

        File testDirectory = new File("C:\\gitlab\\facial_expressions\\processed");
        FileSplit testSplit = new FileSplit(testDirectory, new String[] {"jpg"});
        recordReader.initialize(testSplit, null);
        RecordReaderDataSetIterator testDataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
        testDataIter.setPreProcessor(new ImageFlatteningDataSetPreProcessor());
        testDataIter.setPreProcessor(new ImagePreProcessingScaler(0, 1));
        Evaluation eval = network.evaluate(testDataIter);
        log.info(eval.stats(false));
    }

    private static MultiLayerNetwork initConvModel(String modelPath) throws IOException {
        File modelFile = new File(modelPath);
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        if (modelFile.exists()) {
            MultiLayerNetwork multiLayerNetwork = ModelSerializer.restoreMultiLayerNetwork(modelFile);
            multiLayerNetwork.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(10));
            return multiLayerNetwork;
        }

        double nonZeroBias = 1;
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
//                .iterations(iterations)
                .l2(0.005) // tried 0.0001, 0.0005
                .activation(Activation.RELU)
//                .learningRate(0.05) // tried 0.001, 0.005, 0.01
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.ADAM)
                .list()
                .layer(0, convInit("cnn1", channels, 32, new int[] {5, 5}, new int[] {1, 1}, new int[] {0, 0}, 0))
                .layer(1, maxPool("maxpool1", new int[] {2, 2}))
                .layer(2, conv3x3("cnn2", 64, 0))
                .layer(3, conv3x3("cnn3", 64, 1))
                .layer(4, maxPool("maxpool2", new int[] {2, 2}))
                .layer(5, new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(512).dropOut(0.5).build())
                .layer(6, new OutputLayer.Builder(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY)
                        .nOut(numLabels)
                        .activation(Activation.SOFTMAX)
                        .build())
                .backprop(true).pretrain(false)
                .setInputType(InputType.convolutional(height, width, channels))
                .build();
        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(10));
        return network;
    }

    @NotNull
    private static Map<String, Integer> preProcessImages() throws IOException, InterruptedException {
        CSVRecordReader csvRecordReader = new CSVRecordReader(1);
//        csvRecordReader.initialize(new FileSplit(new File("C:\\gitlab\\facial_expressions\\data\\500_picts_satz.csv")));
        csvRecordReader.initialize(new FileSplit(new File("C:\\gitlab\\facial_expressions\\data\\legend.csv")));
        Map<String, Integer> counters = new HashMap<>();
        while (csvRecordReader.hasNext()) {
            List<Writable> next = csvRecordReader.next();
            String emotion = next.get(2).toString().toLowerCase();
            if (counters.containsKey(emotion)) {
                counters.put(emotion, counters.get(emotion) + 1);
            } else {
                counters.put(emotion, 1);
            }
            File baseImage = new File("C:\\gitlab\\facial_expressions\\images\\" + next.get(1).toString());
            File destinationImage = new File("C:\\gitlab\\facial_expressions\\processed\\" + emotion + "\\" + next.get(1).toString());

            FileUtils.copyFile(baseImage, destinationImage);
        }
        return counters;
    }
}
