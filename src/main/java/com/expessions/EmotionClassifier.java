package com.expessions;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FileUtils;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.api.writable.Writable;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.saver.LocalFileGraphSaver;
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.ClassificationScoreCalculator;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculatorCG;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxTimeIterationTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingGraphTrainer;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
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
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.AdaDelta;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.TimeUnit;

@Slf4j
public class EmotionClassifier {
    public static final String MODEL_PATH = "emotion-rnn.data";
    protected static int height = 48;
    protected static int width = 48;
    protected static int channels = 3;
    protected static int batchSize = 50;
    protected static long seed = 42;
    protected static Random rng = new Random(seed);
    protected static int epochs = 15; //epochs * 21 iterations per 1 transformation
    protected static int cycles = 2000;
    private static int numLabels = 5;
    protected static double splitTrainTest = 0.7;

    public static void main(String[] args) throws Exception, InterruptedException {
        ComputationGraph network = initConvModel(MODEL_PATH);
        System.out.println(network.summary());
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        File mainPath = new File("C:\\gitlab\\facial_expressions\\processed");
        FileSplit fileSplit = new FileSplit(mainPath, NativeImageLoader.ALLOWED_FORMATS, rng);
        int numExamples = Math.toIntExact(fileSplit.length());

        System.out.println("Num examples : " + numExamples);
        numLabels = fileSplit.getRootDir().listFiles(File::isDirectory).length;
        BalancedPathFilter pathFilter = new BalancedPathFilter(new Random(42), labelMaker, numExamples, numLabels, batchSize);

//        InputSplit[] inputSplit = fileSplit.sample(pathFilter, 10000, 3000);
        InputSplit[] inputSplit = fileSplit.sample(pathFilter, numExamples * splitTrainTest, numExamples * (1 - splitTrainTest));
        InputSplit trainData = inputSplit[0];
        InputSplit testData = inputSplit[1];


        // System.out.println("Test data: " + trainData.length());
        trainModelWithTrainer(network, trainData, testData, labelMaker);
        //   evaluateModel(network, trainData, testData, labelMaker);
        ModelSerializer.writeModel(network, MODEL_PATH, true);
    }

    public String classify(String imagePath) throws IOException {
        File modelFile = new File(MODEL_PATH);
        if (!modelFile.exists()) {
            log.info("Models does not exists");
            return imagePath;
        }
        String[] labels = {"anger", "happiness", "neutral", "sadness", "surprise"};
        ComputationGraph multiLayerNetwork = ModelSerializer.restoreComputationGraph(modelFile, false);
        INDArray matrix = new NativeImageLoader(height, width, 3).asMatrix(new File(imagePath));
        normalizeImage(matrix);
        System.out.println(Arrays.toString(multiLayerNetwork.rnnTimeStep(matrix)));
        INDArray[] output = multiLayerNetwork.output(matrix);
        int indexOfLargest = getIndexOfLargest(output[0].toDoubleVector());
        //System.out.println(Arrays.toString(output));
        return labels[indexOfLargest];
    }

    public int getIndexOfLargest(double[] array) {
        if (array == null || array.length == 0) return -1; // null or empty

        int largest = 0;
        for (int i = 1; i < array.length; i++) {
            if (array[i] > array[largest]) largest = i;
        }
        return largest; // position of the first largest found
    }

    private void normalizeImage(final INDArray image) {
        ImagePreProcessingScaler preProcessor = new ImagePreProcessingScaler(0, 1);
        preProcessor.transform(image);
    }

    private static ComputationGraph initKeras(String modelPath) throws IOException, UnsupportedKerasConfigurationException, InvalidKerasConfigurationException {
        MultiLayerNetwork model = KerasModelImport.importKerasSequentialModelAndWeights("C:\\gitlab\\ml\\ml-java\\model_5-49-0.62.hdf5");
        System.out.println(model.summary());
        return null;
    }

    @NotNull
    private static void trainModel(ComputationGraph network, InputSplit trainData, InputSplit testData, ParentPathLabelGenerator labelMaker) throws IOException {
        log.info("Train model....");

        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);
        DataSetIterator dataIter;
        MultipleEpochsIterator trainIter;


        log.info("Train model....");
        recordReader.initialize(trainData, null);
        dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        scaler.fit(dataIter);
        dataIter.setPreProcessor(scaler);
        trainIter = new MultipleEpochsIterator(epochs, dataIter, 4);
        network.fit(trainIter, epochs);

        log.info("Evaluate model test data....");
        recordReader.initialize(testData);
        dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
        scaler.fit(dataIter);
        dataIter.setPreProcessor(scaler);
        Evaluation eval = network.evaluate(dataIter);
        log.info(eval.stats(false));

        log.info("Evaluate model train data....");
        recordReader.initialize(trainData);
        dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
        scaler.fit(dataIter);
        dataIter.setPreProcessor(scaler);
        eval = network.evaluate(dataIter);
        log.info(eval.stats(false));
    }

    @NotNull
    private static void trainModelWithTrainer(ComputationGraph network, InputSplit trainData, InputSplit testData, ParentPathLabelGenerator labelMaker) throws IOException {
        log.info("Train model....");

        ImageRecordReader testRecordReader = new ImageRecordReader(height, width, channels, labelMaker);
        testRecordReader.initialize(testData);
        DataSetIterator testDataIter = new RecordReaderDataSetIterator(testRecordReader, batchSize, 1, numLabels);
        DataNormalization testScaler = new ImagePreProcessingScaler(0, 1);
        testScaler.fit(testDataIter);
        testDataIter.setPreProcessor(testScaler);

        EarlyStoppingConfiguration esConf = new EarlyStoppingConfiguration.Builder<ComputationGraph>()
                .iterationTerminationConditions(new MaxTimeIterationTerminationCondition(10, TimeUnit.MINUTES))
//                .scoreCalculator(new DataSetLossCalculator(testDataIter, true))
                .scoreCalculator(new ClassificationScoreCalculator(Evaluation.Metric.ACCURACY, testDataIter))
                .evaluateEveryNEpochs(1)
                .modelSaver(new LocalFileGraphSaver("models"))
                .build();



//Conduct early stopping training:
        DataNormalization trainScaler = new ImagePreProcessingScaler(0, 1);


        log.info("Train model....");
        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);
        recordReader.initialize(trainData, null);
        DataSetIterator trainDataIterator = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
        trainScaler.fit(trainDataIterator);
        trainDataIterator.setPreProcessor(trainScaler);
        MultipleEpochsIterator trainIter = new MultipleEpochsIterator(epochs, trainDataIterator, 4);
        EarlyStoppingGraphTrainer trainer = new EarlyStoppingGraphTrainer(esConf,network,trainIter);
        trainer.setListener(new LoggingEarlyStoppingListener());
        EarlyStoppingResult<ComputationGraph> result = trainer.fit();
//        network.fit(trainIter, epochs);

        System.out.println("Termination reason: " + result.getTerminationReason());
        System.out.println("Termination details: " + result.getTerminationDetails());
        System.out.println("Total epochs: " + result.getTotalEpochs());
        System.out.println("Best epoch number: " + result.getBestModelEpoch());
        System.out.println("Score at best epoch: " + result.getBestModelScore());

//Get the best model:
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        ComputationGraph bestModel = result.getBestModel();
        System.out.println(bestModel);
//
        log.info("Evaluate model test data....");
        recordReader = new ImageRecordReader(height, width, channels, labelMaker);
        recordReader.initialize(testData, null);
        testDataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
        scaler.fit(testDataIter);
        Evaluation eval = bestModel.evaluate(testDataIter);
        log.info(eval.stats(false));

//
        log.info("Evaluate model trainDataIterator data....");
        recordReader.initialize(trainData);
        trainDataIterator = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
        scaler.fit(trainDataIterator);
        trainDataIterator.setPreProcessor(scaler);
        eval = bestModel.evaluate(trainDataIterator);
        log.info(eval.stats(false));
    }

    private static void evaluateModel(ComputationGraph network, InputSplit trainData, InputSplit testData, ParentPathLabelGenerator labelMaker) throws IOException {
        log.info("Evaluate model....");
        trainData.reset();
        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);
        DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
        DataSet trainDataSet = iterator.next();

        RecordReaderDataSetIterator testDataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
        testDataIter.setPreProcessor(new ImagePreProcessingScaler(0, 1));
        Evaluation eval = network.evaluate(testDataIter);

        log.info("Training data...");
        log.info(eval.stats(false));

        recordReader.reset();
        testDataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
        recordReader.initialize(testData, null);
        testDataIter.setPreProcessor(new ImagePreProcessingScaler(0, 1));
        eval = network.evaluate(testDataIter);
        log.info("Test data...");
        log.info(eval.stats(false));
    }

    private static ComputationGraph initAlex(String modelPath) throws IOException {
        File modelFile = new File(modelPath);
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
//        if (modelFile.exists()) {
//            log.info("Restoring model....");
//            ComputationGraph multiLayerNetwork = ModelSerializer.restoreComputationGraph(modelFile);
//            multiLayerNetwork.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(10));
//            return multiLayerNetwork;
//        }

        int[] inputShape = new int[] {3, 32, 32};
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
                                .nOut(numLabels).activation(Activation.SOFTMAX) // radial basis function required
                                .build(), "19")
                        .setOutputs("20")
                        .backprop(true).pretrain(false)
                        .setInputTypes(InputType.convolutional(height, width, channels))
                        .build();

        ComputationGraph network = new ComputationGraph(conf);
        network.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(10));
        return network;
    }

    private static ComputationGraph initConvModel(String modelPath) throws IOException {
        File modelFile = new File(modelPath);
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
//        if (modelFile.exists()) {
//            ComputationGraph multiLayerNetwork = ModelSerializer.restoreComputationGraph(modelFile);
//            multiLayerNetwork.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(10));
//            return multiLayerNetwork;
//        }

        ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .l2(0.0001)
                .updater(new Adam(0.0001))
                .weightInit(WeightInit.XAVIER_UNIFORM)
                .graphBuilder()
                .addInputs("trainFeatures")
                .setInputTypes(InputType.convolutional(height, width, channels))
                .setOutputs("out1")
                .addLayer("cnn1", new ConvolutionLayer.Builder(new int[] {5, 5}, new int[] {1, 1}, new int[] {1, 1})
                        .nIn(channels).nOut(64).activation(Activation.RELU).build(), "trainFeatures")
                .addLayer("LRN1",  new LocalResponseNormalization.Builder().name("LRN1").build(), "cnn1")
                .addLayer("maxpool1", new SubsamplingLayer.Builder(PoolingType.MAX, new int[] {3, 3}, new int[] {2, 2}, new int[] {0, 0})
                        .build(), "LRN1")
                .addLayer("cnn2", new ConvolutionLayer.Builder(new int[] {5, 5}, new int[] {2, 2}, new int[] {0, 0})
                        .nOut(64).activation(Activation.RELU).build(), "maxpool1")
                .addLayer("maxpool2", new SubsamplingLayer.Builder(PoolingType.MAX, new int[] {3, 3}, new int[] {2, 2}, new int[] {0, 0})
                        .build(), "cnn2")
                .addLayer("cnn3", new ConvolutionLayer.Builder(new int[] {4, 4}, new int[] {1, 1}, new int[] {0, 0})
                        .nOut(128).activation(Activation.RELU).build(), "maxpool2")
//                .addLayer("maxpool3", new SubsamplingLayer.Builder(PoolingType.MAX, new int[] {2, 2}, new int[] {2, 2}, new int[] {0, 0})
//                        .build(), "cnn3")
                .addLayer("ffn1", new DenseLayer.Builder().nOut(3072)
                        .dropOut(0.5).build(), "cnn3")
//                .addLayer("ffn1", new DenseLayer.Builder().nOut(2048)
//                        .dropOut(0.5).build(), "ffn0")
                .addLayer("out1", new OutputLayer.Builder(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY)
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
