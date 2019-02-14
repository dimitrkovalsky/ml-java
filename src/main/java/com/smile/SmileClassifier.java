package com.smile;

import lombok.extern.slf4j.Slf4j;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
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
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.model.VGG16;
import org.jetbrains.annotations.NotNull;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static com.classification.AnimalsClassification.conv5x5;

@Slf4j
public class SmileClassifier {
    public static final String DATA_FOLDER = "C:\\gitlab\\ml\\ml-java\\datasets\\merged";
    public static final String MODEL_FILENAME = "smile.data";
    protected static int height = 32;
    protected static int width = 32;

    protected static int channels = 3;
    protected static int batchSize = 50;
    protected static long seed = 123;
    protected static Random rng = new Random(seed);
    protected static int nEpochs = 30; // tested 50, 100, 200
    protected static double splitTrainTest = 0.8;
    private int numLabels;

    public void recognizeSmile(String imagePath) throws IOException {
        File modelFile = new File(MODEL_FILENAME);
        if (!modelFile.exists()) {
            log.info("Models does not exists");
            return;
        }
        MultiLayerNetwork multiLayerNetwork = ModelSerializer.restoreMultiLayerNetwork(modelFile);
        INDArray matrix = new NativeImageLoader(height, width, 3).asMatrix(new File(imagePath));
        normalizeImage(matrix);
        System.out.println(multiLayerNetwork.getLabels());
        System.out.println(multiLayerNetwork.output(matrix));
        System.out.println(Arrays.toString(multiLayerNetwork.predict(matrix)));
    }

    private void normalizeImage(final INDArray image) {
        ImagePreProcessingScaler preProcessor = new ImagePreProcessingScaler(0, 1);
        preProcessor.transform(image);
    }

    private void classifyTensorflow(INDArray arr) {
        File file = new File("");
        arr = Nd4j.expandDims(arr, 0);
        SameDiff sd = TFGraphMapper.getInstance().importGraph(file);
        sd.associateArrayWithVariable(arr, sd.variables().get(0));
        INDArray outArr = sd.execAndEndResult();
        double pred = outArr.getDouble(0);
        System.out.println(pred);
    }

    public void execute() throws Exception {
        log.info("Loading data....");
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        File mainPath = new File(DATA_FOLDER);
        FileSplit fileSplit = new FileSplit(mainPath, NativeImageLoader.ALLOWED_FORMATS, rng);
        int numExamples = Math.toIntExact(fileSplit.length());
        numLabels = fileSplit.getRootDir().listFiles(File::isDirectory).length;
        BalancedPathFilter pathFilter = new BalancedPathFilter(new Random(100500), labelMaker, numExamples, numLabels, batchSize/2);

        InputSplit[] inputSplit = fileSplit.sample(pathFilter, splitTrainTest, 1 - splitTrainTest);
        InputSplit trainData = inputSplit[0];
        InputSplit testData = inputSplit[1];

        ImageTransform flipTransform1 = new FlipImageTransform(rng);
        ImageTransform flipTransform2 = new FlipImageTransform(new Random(123));
//        ImageTransform resizeTransform2 = new RotateImageTransform(30);
//        ImageTransform resizeTransform3 = new ColorConversionTransform(1);
//        ImageTransform scaleTransform = new ScaleImageTransform(0.5f);
//        ImageTransform scaleTransform2 = new ScaleImageTransform(2);

        List<ImageTransform> transforms = Arrays.asList(new ImageTransform[] {flipTransform1, flipTransform2});


        log.info("Fitting to dataset");



        MultiLayerNetwork network = buildNetwork(MODEL_FILENAME);
        network.init();
        System.out.println(network.summary());

        log.info("Load data....");
        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);
        trainModel(trainData, transforms, network, recordReader);

        evaluateModel(testData, network, recordReader);
        log.info("**************** Smile Classification finished ********************");
    }

    public ComputationGraph tuned() throws IOException {
        ComputationGraph preTrainedNet = (ComputationGraph)
                VGG16.builder().build().initPretrained(PretrainedType.IMAGENET);
        log.info(preTrainedNet.summary());
        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.NESTEROVS)
                .seed(seed)
                .build();
        String featurizeExtractionLayer = "fc2";
        ComputationGraph vgg16Transfer = new TransferLearning.GraphBuilder(preTrainedNet)
                .fineTuneConfiguration(fineTuneConf)
                .setFeatureExtractor(featurizeExtractionLayer)
                .removeVertexKeepConnections("predictions")
                .addLayer("predictions", new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(4096).nOut(2)
                        .weightInit(WeightInit.XAVIER).activation(Activation.SOFTMAX).build(), featurizeExtractionLayer).build();
        return vgg16Transfer;
    }


    private void evaluateModel(InputSplit testData, MultiLayerNetwork network, ImageRecordReader recordReader) throws IOException {
        DataSetIterator dataIter;
        ImagePreProcessingScaler preProcessor = new ImagePreProcessingScaler(0, 1);
        log.info("Evaluate model....");
        recordReader.initialize(testData);
        dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
        preProcessor.fit(dataIter);
        dataIter.setPreProcessor(preProcessor);
        Evaluation eval = network.evaluate(dataIter);
        log.info(eval.stats(false));
    }

    private void trainModel(InputSplit trainData, List<ImageTransform> transforms, MultiLayerNetwork network, ImageRecordReader recordReader) throws IOException {
        DataSetIterator dataIter;
        MultipleEpochsIterator trainIter;
        ImagePreProcessingScaler preProcessor = new ImagePreProcessingScaler(0, 1);
        log.info("Train model....");

        recordReader.initialize(trainData, null);
        dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
        preProcessor.fit(dataIter);
        dataIter.setPreProcessor(preProcessor);
        trainIter = new MultipleEpochsIterator(nEpochs, dataIter);

        network.fit(trainIter);

        // Train with transformations
        for (ImageTransform transform : transforms) {
            System.out.print("\nTraining on transformation: " + transform.getClass().toString() + "\n\n");
            recordReader.initialize(trainData, transform);
            dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
            preProcessor.fit(dataIter);
            dataIter.setPreProcessor(preProcessor);
            trainIter = new MultipleEpochsIterator(nEpochs, dataIter);
            network.fit(trainIter);
            ModelSerializer.writeModel(network, MODEL_FILENAME, true);
        }
    }

    @NotNull
    private MultiLayerNetwork buildNetwork(String modelPath) throws IOException {
        File modelFile = new File(modelPath);
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        if (modelFile.exists()) {
            log.info("Restoring model....");
            MultiLayerNetwork multiLayerNetwork = ModelSerializer.restoreMultiLayerNetwork(modelFile);
            multiLayerNetwork.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(10));
            return multiLayerNetwork;
        }
        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .l2(0.005) // tried 0.0001, 0.0005
                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.ADAM)
                .list()
                .layer(0, convInit("cnn1", channels, 128, new int[] {5, 5}, new int[] {1, 1}, new int[] {0, 0}, 0))
                .layer(1, maxPool("maxpool1", new int[] {2, 2}))
                .layer(2, conv3x3("cnn2", 128, 0))
                .layer(3, conv3x3("cnn3", 64, 1))
                .layer(4, maxPool("maxpool2", new int[] {3, 3}))
                .layer(5, new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(512).dropOut(0.6).build())
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
    private MultiLayerNetwork buildNetwork2(String modelPath) throws IOException {
        File modelFile = new File(modelPath);
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        if (modelFile.exists()) {
            log.info("Restoring model....");
            MultiLayerNetwork multiLayerNetwork = ModelSerializer.restoreMultiLayerNetwork(modelFile);
            multiLayerNetwork.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(10));
            return multiLayerNetwork;
        }
        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .l2(0.005) // tried 0.0001, 0.0005
                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.ADADELTA)
                .list()
//                .layer(0, convInit("cnn1", channels, 32, new int[] {5, 5}, new int[] {1, 1}, new int[] {0, 0}, 0))
//                .layer(1, maxPool("maxpool1", new int[] {3, 3}))
                .layer(0, conv5x5("cnn1", 32, new int[] {2, 2}, new int[] {1, 1}, 0))
                .layer(1, maxPool("maxpool1", new int[] {2, 2}))
                .layer(2, new LocalResponseNormalization.Builder().name("LRN1").build())
                .layer(3, conv5x5("cnn2", 32, new int[] {2, 2}, new int[] {1, 1}, 0))
                .layer(4, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG, new int[] {3, 3}).name("avgpool1").build())
                .layer(5, new LocalResponseNormalization.Builder().name("LRN2").build())
                .layer(6, conv5x5("cnn3", 64, new int[] {2, 2}, new int[] {1, 1}, 0))
                .layer(7, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG, new int[] {1, 1}).name("avgpool2").build())
                .layer(8, new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(1024).build())
                .layer(9, new DenseLayer.Builder()
                        .nOut(2)
                        .activation(Activation.RELU)
                        .build())
                .layer(10, new OutputLayer.Builder(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY)
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

    private ConvolutionLayer convInit(String name, int in, int out, int[] kernel, int[] stride, int[] pad, double bias) {
        return new ConvolutionLayer.Builder(kernel, stride, pad).name(name).nIn(in).nOut(out).biasInit(bias).build();
    }

    private ConvolutionLayer conv3x3(String name, int out, double bias) {
        return new ConvolutionLayer.Builder(new int[] {3, 3}, new int[] {1, 1}, new int[] {1, 1}).name(name).nOut(out).biasInit(bias).build();
    }


    private SubsamplingLayer maxPool(String name, int[] kernel) {
        return new SubsamplingLayer.Builder(kernel, new int[] {2, 2}).name(name).build();
    }

    public ComputationGraph vgg(String modelPath) throws IOException {
        File modelFile = new File(modelPath);
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        if (modelFile.exists()) {
            log.info("Restoring model....");
            ComputationGraph multiLayerNetwork = ModelSerializer.restoreComputationGraph(modelFile);
            multiLayerNetwork.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(10));
            return multiLayerNetwork;
        }

        int[] inputShape = new int[] {3, 32, 32};
        int numClasses = 2;
        IUpdater updater = new Nesterovs();
        CacheMode cacheMode = CacheMode.NONE;
        WorkspaceMode workspaceMode = WorkspaceMode.ENABLED;
        ConvolutionLayer.AlgoMode cudnnAlgoMode = ConvolutionLayer.AlgoMode.PREFER_FASTEST;
        ComputationGraphConfiguration conf =
                new NeuralNetConfiguration.Builder().seed(seed)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .updater(updater)
                        .activation(Activation.RELU)
                        .cacheMode(cacheMode)
                        .trainingWorkspaceMode(workspaceMode)
                        .inferenceWorkspaceMode(workspaceMode)
                        .graphBuilder()
                        .addInputs("in")
                        // block 1
                        .layer(0, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                .padding(1, 1).nIn(inputShape[0]).nOut(64)
                                .cudnnAlgoMode(cudnnAlgoMode).build(), "in")
                        .layer(1, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                .padding(1, 1).nOut(64).cudnnAlgoMode(cudnnAlgoMode).build(), "0")
                        .layer(2, new SubsamplingLayer.Builder()
                                .poolingType(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
                                .stride(2, 2).build(), "1")
                        // block 2
                        .layer(3, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                .padding(1, 1).nOut(128).cudnnAlgoMode(cudnnAlgoMode).build(), "2")
                        .layer(4, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                .padding(1, 1).nOut(128).cudnnAlgoMode(cudnnAlgoMode).build(), "3")
                        .layer(5, new SubsamplingLayer.Builder()
                                .poolingType(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
                                .stride(2, 2).build(), "4")
                        // block 3
                        .layer(6, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                .padding(1, 1).nOut(256).cudnnAlgoMode(cudnnAlgoMode).build(), "5")
                        .layer(7, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                .padding(1, 1).nOut(256).cudnnAlgoMode(cudnnAlgoMode).build(), "6")
                        .layer(8, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                .padding(1, 1).nOut(256).cudnnAlgoMode(cudnnAlgoMode).build(), "7")
                        .layer(9, new SubsamplingLayer.Builder()
                                .poolingType(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
                                .stride(2, 2).build(), "8")
                        // block 4
                        .layer(10, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                .padding(1, 1).nOut(512).cudnnAlgoMode(cudnnAlgoMode).build(), "9")
                        .layer(11, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                .padding(1, 1).nOut(512).cudnnAlgoMode(cudnnAlgoMode).build(), "10")
                        .layer(12, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                .padding(1, 1).nOut(512).cudnnAlgoMode(cudnnAlgoMode).build(), "11")
                        .layer(13, new SubsamplingLayer.Builder()
                                .poolingType(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
                                .stride(2, 2).build(), "12")
                        // block 5
                        .layer(14, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                .padding(1, 1).nOut(512).cudnnAlgoMode(cudnnAlgoMode).build(), "13")
                        .layer(15, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                .padding(1, 1).nOut(512).cudnnAlgoMode(cudnnAlgoMode).build(), "14")
                        .layer(16, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                .padding(1, 1).nOut(512).cudnnAlgoMode(cudnnAlgoMode).build(), "15")
                        .layer(17, new SubsamplingLayer.Builder()
                                .poolingType(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
                                .stride(2, 2).build(), "16")
                        .layer(18, new DenseLayer.Builder().nOut(4096).dropOut(0.5)
                                .build(), "17")
                        .layer(19, new DenseLayer.Builder().nOut(4096).dropOut(0.5)
                                .build(), "18")
                        .layer(20, new OutputLayer.Builder(
                                LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).name("output")
                                .nOut(numClasses).activation(Activation.SOFTMAX) // radial basis function required
                                .build(), "19")
                        .setOutputs("20")
                        .backprop(true).pretrain(false)
                        .setInputTypes(InputType.convolutional(height, width, channels))
                        .build();

        ComputationGraph network = new ComputationGraph(conf);
        network.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(10));
        return network;
    }


    public static void main(String[] args) throws Exception {
        new SmileClassifier().execute();
    }
}
