package com.smile;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.inmemory.InMemoryRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.writable.Writable;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.*;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.jetbrains.annotations.NotNull;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ops.impl.transforms.ReluLayer;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.schedule.ScheduleType;
import org.nd4j.linalg.schedule.StepSchedule;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.*;

import static com.classification.AnimalsClassification.*;

public class SmileDetector {
    protected static final Logger log = LoggerFactory.getLogger(SmileDetector.class);
    public static final String MODEL_DATA = "C:\\gitlab\\ml\\ml-java\\models\\model.data";
    public static final String VGG_DATA = "C:\\gitlab\\ml\\ml-java\\models\\vgg-model.data";

    protected static int height = 64;
    protected static int width = 64;
    protected static int channels = 3;
    protected static int batchSize = 32;
    protected static long seed = 42;
    protected static Random rng = new Random(seed);
    protected static int epochs = 10;
    protected static int cycles = 2000;
    private static int numLabels = 2;


    public static void main(String[] args) throws Exception {
//        ComputationGraph network = initModel();
        String modelPath = "conv.data";
//        MultiLayerNetwork network = initPlainModel(modelPath);
        MultiLayerNetwork network = initConvModel(modelPath);
        log.info(network.summary());
//        RecordReader reader = prepareData();
        int cyclesDone = 0;
        while (cyclesDone < cycles) {
            trainModel(network);
         //   log.info("Model trained");
            evaluateModel(network);
           // ModelSerializer.writeModel(network, modelPath, true);
            cyclesDone++;
          //  log.info("{} cycles ****************Example finished********************", cyclesDone);
        }

    }

    private static RecordReader prepareData() throws IOException {
        File trainDirectory = new File("C:\\gitlab\\ml\\ml-java\\datasets\\train_folder");
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        ImageRecordReader recordReader = new ImageRecordReader(64, 64, channels, labelMaker);
        FileSplit trainSplit = new FileSplit(trainDirectory, new String[] {"jpg"});
        //recordReader.initialize(trainSplit, null);
        return recordReader;
//        List<List<Writable>> shuffled = new ArrayList<>();
//        while (recordReader.hasNext()) {
//            List<Writable> next = recordReader.next();
//            shuffled.add(next);
////            System.out.println(((IntWritable) next.get(1)).get());
//        }
//        Collections.shuffle(shuffled);
//        return new InMemoryRecordReader(shuffled);
    }


    @NotNull
    private static void trainModel(MultiLayerNetwork network) throws IOException {
        log.info("Train model....");
//        RecordReader reader = prepareData();
        ImageTransform flipTransform1 = new FlipImageTransform(new Random());
        ImageTransform flipTransform2 = new FlipImageTransform(new Random(123));
        ImageTransform resizeTransform2 = new RotateImageTransform(30);
        ImageTransform resizeTransform3 = new ColorConversionTransform(1);
        ImageTransform scaleTransform = new ScaleImageTransform(0.5f);
        ImageTransform scaleTransform2 = new ScaleImageTransform(2);

        List<ImageTransform> pipeline = Arrays.asList(new ImageTransform[]{flipTransform1, flipTransform2, resizeTransform2, resizeTransform3, scaleTransform, scaleTransform2});
        File trainDirectory = new File("C:\\gitlab\\ml\\ml-java\\datasets\\train_folder");
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        ImageRecordReader recordReader = new ImageRecordReader(64, 64, channels, labelMaker);
        FileSplit trainSplit = new FileSplit(trainDirectory, new String[] {"jpg"});

        for(ImageTransform transform : pipeline) {
            recordReader.initialize(trainSplit, transform);
//            reader.reset();
            RecordReaderDataSetIterator trainDataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
            trainDataIter.setPreProcessor(new ImagePreProcessingScaler(0, 1));
//            dataIter.setPreProcessor(new ImageFlatteningDataSetPreProcessor());
            network.fit(trainDataIter, epochs);
        }
    }

    private static void evaluateModel(MultiLayerNetwork network) throws IOException {
        log.info("Evaluate model....");
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        ImageRecordReader recordReader = new ImageRecordReader(64, 64, channels, labelMaker, null);

        File testDirectory = new File("C:\\gitlab\\ml\\ml-java\\datasets\\test_folder");
        FileSplit testSplit = new FileSplit(testDirectory, new String[] {"jpg"});
        recordReader.initialize(testSplit, null);
        RecordReaderDataSetIterator testDataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
//            dataIter.setPreProcessor(new ImageFlatteningDataSetPreProcessor());
        testDataIter.setPreProcessor(new ImagePreProcessingScaler(0,1));
        Evaluation eval = network.evaluate(testDataIter);
        log.info(eval.stats(false));

//
//        DataSet testDataSet = testDataIter.next();
//        List<String> allClassLabels = recordReader.getLabels();
//        int labelIndex = testDataSet.getLabels().argMax(1).getInt(0);
//        int[] predictedClasses = network.predict(testDataSet.getFeatures());
//        String expectedResult = allClassLabels.get(labelIndex);
//        String modelPrediction = allClassLabels.get(predictedClasses[0]);
//        System.out.print("\nFor a single example that is labeled " + expectedResult + " the model predicted " + modelPrediction + "\n\n");
        log.info("Save model....");

    }

    private static MultiLayerNetwork initNewModel(String modelPath) throws IOException {
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
        double dropOut = 0.1;
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.DISTRIBUTION)
                .dist(new NormalDistribution(0.0, 0.01))
                .activation(Activation.RELU)
                .updater(new Nesterovs(new StepSchedule(ScheduleType.ITERATION, 1e-2, 0.1, 100000), 0.9))
                .biasUpdater(new Nesterovs(new StepSchedule(ScheduleType.ITERATION, 2e-2, 0.1, 100000), 0.9))
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer) // normalize to prevent vanishing or exploding gradients
                .l2(5 * 1e-4)
                .list()
                .layer(0, convInit("cnn1", channels, 96, new int[] {11, 11}, new int[] {4, 4}, new int[] {3, 3}, 0))
                .layer(1, new LocalResponseNormalization.Builder().name("lrn1").build())
                .layer(2, maxPool("maxpool1", new int[] {3, 3}))
                .layer(3, conv5x5("cnn2", 256, new int[] {1, 1}, new int[] {2, 2}, nonZeroBias))
                .layer(4, new LocalResponseNormalization.Builder().name("lrn2").build())
                .layer(5, maxPool("maxpool2", new int[] {3, 3}))
                .layer(6, conv3x3("cnn3", 384, 0))
                .layer(7, conv3x3("cnn4", 384, nonZeroBias))
                .layer(8, conv3x3("cnn5", 256, nonZeroBias))
                .layer(9, maxPool("maxpool3", new int[] {3, 3}))
                .layer(10, fullyConnected("ffn1", 4096, nonZeroBias, dropOut, new GaussianDistribution(0, 0.005)))
                .layer(11, fullyConnected("ffn2", 4096, nonZeroBias, dropOut, new GaussianDistribution(0, 0.005)))
                .layer(12, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .name("output")
                        .nOut(numLabels)
                        .activation(Activation.SOFTMAX)
                        .build())
                .backprop(true)
                .pretrain(false)
                .setInputType(InputType.convolutional(height, width, channels))
                .build();
        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(10));
        return network;
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
        double dropOut = 0.1;
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .l2(0.0005) // tried 0.0001, 0.0005
                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam(0.01))
                .list()
                .layer(0, convInit("cnn1", channels, 96 ,  new int[]{5, 5}, new int[]{1, 1}, new int[]{0, 0}, 1))
                .layer(1, maxPool("maxpool1", new int[]{2,2}))
                .layer(2, conv3x3("cnn2", 64, 1))
                .layer(3, conv3x3("cnn3", 64,1))
                .layer(4, maxPool("maxpool2", new int[]{2,2}))
                .layer(5, new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(512).dropOut(0.7).build())
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

    public static MultiLayerNetwork initPlainModel(String modelPath) throws IOException {
        File modelFile = new File(modelPath);
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        if (modelFile.exists()) {
            MultiLayerNetwork multiLayerNetwork = ModelSerializer.restoreMultiLayerNetwork(modelFile);
            multiLayerNetwork.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(10));
            return multiLayerNetwork;
        }
        int numOutputs = 2;
        int numHiddenNodes = 1024;
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .biasInit(1)
                .l2(1e-4)
                .updater(new Nesterovs(0.01, 0.9))
                .list()
                .layer(0, new DenseLayer.Builder().nIn(64 * 64 * 3).nOut(numHiddenNodes)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.SOFTMAX)
                        .nIn(numHiddenNodes).nOut(numOutputs).build())
//                .setInputType(InputType.convolutional(height, width, channels))
                .build();
        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(10));
        return network;
    }

    public static ComputationGraph initModel() throws Exception {
        File modelFile = new File(VGG_DATA);
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        if (modelFile.exists()) {
            ComputationGraph multiLayerNetwork = ModelSerializer.restoreComputationGraph(modelFile);
            multiLayerNetwork.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(10));
            return multiLayerNetwork;
        }

        Model model = VGG16.builder().build().initPretrained();
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
                .addLayer("newpredictions", new OutputLayer
                        .Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX)
                        .nIn(256).nOut(numLabels).build(), "fc2")
                .setOutputs("newpredictions")
                .build();
        vgg16Transfer.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(10));


        return vgg16Transfer;
    }
}
