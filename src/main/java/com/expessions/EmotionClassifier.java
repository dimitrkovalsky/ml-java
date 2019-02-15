package com.expessions;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FileUtils;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.api.writable.IntWritable;
import org.datavec.api.writable.Writable;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.*;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
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
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
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
    public static final String MODEL_PATH = "emotion-rnn.data";
    protected static int height = 48;
    protected static int width = 48;
    protected static int channels = 3;
    protected static int batchSize = 32;
    protected static long seed = 42;
    protected static Random rng = new Random(seed);
    protected static int epochs = 10; //epochs * 21 iterations per 1 transformation
    protected static int cycles = 2000;
    private static int numLabels = 8;
    protected static double splitTrainTest = 0.7;

    public static void main(String[] args) throws IOException, InterruptedException {
        ComputationGraph network = initConvModel(MODEL_PATH);
        System.out.println(network.summary());
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        File mainPath = new File("C:\\gitlab\\facial_expressions\\processed");
        FileSplit fileSplit = new FileSplit(mainPath, NativeImageLoader.ALLOWED_FORMATS, rng);
        int numExamples = Math.toIntExact(fileSplit.length());
        numLabels = fileSplit.getRootDir().listFiles(File::isDirectory).length;
        BalancedPathFilter pathFilter = new BalancedPathFilter(new Random(42), labelMaker, numExamples, numLabels, batchSize/2);

        InputSplit[] inputSplit = fileSplit.sample(pathFilter, splitTrainTest, 1 - splitTrainTest);
        InputSplit trainData = inputSplit[0];
        InputSplit testData = inputSplit[1];

        trainModel(network, trainData, labelMaker);
        evaluateModel(network, testData, labelMaker);
    }

    @NotNull
    private static void trainModel(ComputationGraph network, InputSplit trainData, ParentPathLabelGenerator labelMaker) throws IOException {
        log.info("Train model....");
        ImageTransform flipTransform1 = new FlipImageTransform(new Random());
        ImageTransform flipTransform2 = new FlipImageTransform(new Random(123));
        ImageTransform resizeTransform2 = new RotateImageTransform(30);
        ImageTransform resizeTransform3 = new ColorConversionTransform(1);

        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);
//        recordReader.initialize(trainData, new MultiImageTransform(new Random(42), flipTransform1, resizeTransform2, new ResizeImageTransform(width, height)));
        recordReader.initialize(trainData, null);
        MultipleEpochsIterator trainDataIter;
        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
        ImagePreProcessingScaler preProcessor = new ImagePreProcessingScaler(0, 1);
        preProcessor.fit(dataIter);
        dataIter.setPreProcessor(preProcessor);
        trainDataIter = new MultipleEpochsIterator(epochs, dataIter);
        network.fit(trainDataIter, epochs);

//        List<ImageTransform> transforms = Arrays.asList(new ImageTransform[] {flipTransform1, flipTransform2, resizeTransform2, resizeTransform3});
//        for (ImageTransform transform : transforms) {
//            System.out.print("\nTraining on transformation: " + transform.getClass().toString() + "\n\n");
//            recordReader.reset();
//            recordReader.initialize(trainData, transform);
//            dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
//            preProcessor.fit(dataIter);
//            dataIter.setPreProcessor(preProcessor);
//            trainDataIter = new MultipleEpochsIterator(epochs, dataIter);
//            network.fit(trainDataIter);
//          //  ModelSerializer.writeModel(network, MODEL_PATH, true);
//        }
//        }
    }

    private static void evaluateModel(ComputationGraph network, InputSplit testData, ParentPathLabelGenerator labelMaker) throws IOException {
        log.info("Evaluate model....");
        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker, null);

        File testDirectory = new File("C:\\gitlab\\facial_expressions\\processed");
        FileSplit testSplit = new FileSplit(testDirectory, new String[] {"jpg"});
        recordReader.initialize(testSplit, null);
        RecordReaderDataSetIterator testDataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
//        testDataIter.setPreProcessor(new ImageFlatteningDataSetPreProcessor());
        testDataIter.setPreProcessor(new ImagePreProcessingScaler(0, 1));
        Evaluation eval = network.evaluate(testDataIter);
        log.info(eval.stats(false));
    }

    private static ComputationGraph initConvModel(String modelPath) throws IOException {
        File modelFile = new File(modelPath);
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
//        if (modelFile.exists()) {
//            MultiLayerNetwork multiLayerNetwork = ModelSerializer.restoreMultiLayerNetwork(modelFile);
//            multiLayerNetwork.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(10));
//            return multiLayerNetwork;
//        }

        ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .l2(1e-3)
                .updater(new Adam(1e-3))
                .weightInit( WeightInit.XAVIER_UNIFORM)
                .graphBuilder()
                .addInputs("trainFeatures")
                .setInputTypes(InputType.convolutional(height, width, channels))
                .setOutputs("out1")
                .addLayer("cnn1",  new ConvolutionLayer.Builder(new int[]{5, 5}, new int[]{1, 1}, new int[]{0, 0})
                        .nIn(channels).nOut(48).activation( Activation.RELU).build(), "trainFeatures")
                .addLayer("maxpool1",  new SubsamplingLayer.Builder(PoolingType.MAX, new int[]{2,2}, new int[]{2, 2}, new int[]{0, 0})
                        .build(), "cnn1")
                .addLayer("cnn2",  new ConvolutionLayer.Builder(new int[]{5, 5}, new int[]{1, 1}, new int[]{0, 0})
                        .nOut(64).activation( Activation.RELU).build(), "maxpool1")
                .addLayer("maxpool2",  new SubsamplingLayer.Builder(PoolingType.MAX, new int[]{2,1}, new int[]{2, 1}, new int[]{0, 0})
                        .build(), "cnn2")
                .addLayer("cnn3",  new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{0, 0})
                        .nOut(128).activation( Activation.RELU).build(), "maxpool2")
                .addLayer("maxpool3",  new SubsamplingLayer.Builder(PoolingType.MAX, new int[]{2,2}, new int[]{2, 2}, new int[]{0, 0})
                        .build(), "cnn3")
                .addLayer("ffn0",  new DenseLayer.Builder().nOut(3072)
                        .build(), "maxpool3")
                .addLayer("ffn1",  new DenseLayer.Builder().nOut(3072)
                        .build(), "ffn0")
                .addLayer("out1", new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(numLabels).activation(Activation.SOFTMAX).build(), "ffn1")
                .build();

        ComputationGraph network = new ComputationGraph(config);
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
