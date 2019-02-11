package com.smile;

import com.classification.AnimalsClassification;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.FlipImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.PipelineImageTransform;
import org.datavec.image.transform.WarpImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.TinyYOLO;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class SmileDetector {
    protected static final Logger log = LoggerFactory.getLogger(SmileDetector.class);

    protected static int height = 64;
    protected static int width = 64;
    protected static int channels = 3;
    protected static int batchSize = 20;
    protected static long seed = 42;
    protected static Random rng = new Random(seed);
    protected static int epochs = 50;
    private static int numLabels = 2;


    public static void main(String[] args) throws Exception {
        ComputationGraph network = initModel();
        log.info(network.summary());
        File parentDir = new File("C:\\GitHub\\mljava\\datasets\\train_folder");
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);
        FileSplit filesInDir = new FileSplit(parentDir, new String[]{"jpg"});

        System.out.println(filesInDir.length());
//        DataSetIterator dataIter;
//
//        InputSplit[] inputSplit = new InputSplit[]{};
//        InputSplit trainData = inputSplit[0];
//        InputSplit testData = inputSplit[1];
//        ImageTransform flipTransform1 = new FlipImageTransform(rng);
//        ImageTransform flipTransform2 = new FlipImageTransform(new Random(123));
//        ImageTransform warpTransform = new WarpImageTransform(rng, 42);
//        boolean shuffle = false;
//        List<Pair<ImageTransform, Double>> pipeline = Arrays.asList(new Pair<>(flipTransform1, 0.9),
//                new Pair<>(flipTransform2, 0.8),
//                new Pair<>(warpTransform, 0.5));
//
//        ImageTransform transform = new PipelineImageTransform(pipeline, shuffle);
//        /**
//         * Data Setup -> normalization
//         *  - how to normalize images and generate large dataset to train on
//         **/
//        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
//
//
//        log.info("Train model....");
//        // Train without transformations
//        recordReader.initialize(trainData, null);
//        dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
//        scaler.fit(dataIter);
//        dataIter.setPreProcessor(scaler);
//        network.fit(dataIter, epochs);
//
//        // Train with transformations
//        recordReader.initialize(trainData, transform);
//        dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
//        scaler.fit(dataIter);
//        dataIter.setPreProcessor(scaler);
//        network.fit(dataIter, epochs);
//
//        log.info("Evaluate model....");
//        recordReader.initialize(testData);
//        dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
//        scaler.fit(dataIter);
//        dataIter.setPreProcessor(scaler);
//        Evaluation eval = network.evaluate(dataIter);
//        log.info(eval.stats(true));
//
//        // Example on how to get predict results with trained model. Result for first example in minibatch is printed
//        dataIter.reset();
//        DataSet testDataSet = dataIter.next();
//        List<String> allClassLabels = recordReader.getLabels();
//        int labelIndex = testDataSet.getLabels().argMax(1).getInt(0);
//        int[] predictedClasses = network.predict(testDataSet.getFeatures());
//        String expectedResult = allClassLabels.get(labelIndex);
//        String modelPrediction = allClassLabels.get(predictedClasses[0]);
//        System.out.print("\nFor a single example that is labeled " + expectedResult + " the model predicted " + modelPrediction + "\n\n");
    }

    private static ComputationGraph initModel() throws Exception {
        Model model = VGG16.builder().build().initPretrained();
//        MultiLayerNetwork network = (MultiLayerNetwork) model;
        ZooModel zooModel = VGG16.builder().build();
        ComputationGraph pretrainedNet = (ComputationGraph) zooModel.initPretrained(PretrainedType.IMAGENET);
        System.out.println(model.conf().toJson());
        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Nesterovs(5e-5))
                .seed(seed)
                .build();

        ComputationGraph vgg16Transfer = new TransferLearning.GraphBuilder(pretrainedNet)
                .fineTuneConfiguration(fineTuneConf)
                .setFeatureExtractor("block5_pool")
                .nOutReplace("fc2", 1024, WeightInit.XAVIER)
                .removeVertexAndConnections("predictions")
                .addLayer("fc3", new DenseLayer.Builder()
                        .activation(Activation.RELU)
                        .nIn(1024).nOut(256).build(), "fc2")
                .addLayer("newpredictions", new OutputLayer
                        .Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX)
                        .nIn(256).nOut(numLabels).build(), "fc3")
                .setOutputs("newpredictions")
                .build();
        return vgg16Transfer;
    }
}
