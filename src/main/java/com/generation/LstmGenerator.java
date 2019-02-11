package com.generation;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * @author dkovalskyi
 * @since 26.05.2017
 */
@Slf4j
public class LstmGenerator {
    private int lstmLayerSize = 200;                    //Number of units in each LSTM layer
    private int miniBatchSize = 32;                        //Size of mini batch to use when  training
    private int exampleLength = 1000;                    //Length of each training example sequence to use. This could certainly be increased
    private int tbpttLength = 50;                       //Length for truncated backpropagation through time. i.e., do parameter updates ever 50 characters
    private int epochs = 1;                            //Total number of training epochs
    private int generateSamplesEveryNMinibatches = 10;  //How frequently to generate samples from the network? 1000 characters / 50 tbptt length: 20 parameter updates per minibatch
    private int nSamplesToGenerate = 4;                    //Number of samples to generate after each training epoch
    private int nCharactersToSample = 300;

    private String trainFilePath;
    private String modelSavePath;
    private String logFilePath;

    public LstmGenerator(String trainFilePath, String modelSavePath, String logFilePath, int epochs) {
        this.trainFilePath = trainFilePath;
        this.modelSavePath = modelSavePath;
        this.logFilePath = logFilePath;
        this.epochs = epochs;
    }

    public void run() throws Exception {
        MultiLayerNetwork model;
        String generationInitialization = null;
        Random rng = new Random(12345);
        CharacterIterator iter = getIterator(miniBatchSize, exampleLength);
        int nOut = iter.totalOutcomes();
        File modelFile = new File(modelSavePath);
        model = initModel(iter, nOut, modelFile);
        log.info("Network summary: " + model.summary());

        int miniBatchNumber = 0;
        for (int i = 0; i < epochs; i++) {
            while (iter.hasNext()) {
                DataSet ds = iter.next();
                model.fit(ds);
                if (++miniBatchNumber % generateSamplesEveryNMinibatches == 0) {
                    log.info("--------------------");
                    log.info("Completed " + miniBatchNumber + " minibatches of size " + miniBatchSize + "x" + exampleLength + " characters");
                    log.info("Sampling characters from network given initialization \"" + "" + "\"");
                    String[] samples = sampleCharactersFromNetwork(generationInitialization, model, iter, rng, nCharactersToSample, nSamplesToGenerate);
                    for (int j = 0; j < samples.length; j++) {
                        log.info("----- Sample " + j + " -----");
                        log.info(samples[j]);
                    }
                    saveToFile(i, samples);
                }
            }
            saveModel(model);
            iter.reset();
        }

        log.info("\n\nExample complete");
    }

    public String[] sample() throws Exception {
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(new File(modelSavePath));
        return sampleCharactersFromNetwork(null, model, getIterator(miniBatchSize, exampleLength),
                new Random(), nCharactersToSample, nSamplesToGenerate);
    }

    private MultiLayerNetwork initModel(CharacterIterator iter, int nOut, File modelFile) throws IOException {
        MultiLayerNetwork model;
        if (modelFile.exists()) {
            log.info("Model exist in saved file. Restoring...");
            model = ModelSerializer.restoreMultiLayerNetwork(modelFile);
        } else {
            log.info("Creating new network");
            model = createNetwork(lstmLayerSize, iter, nOut);
        }
        return model;
    }


    private void saveModel(MultiLayerNetwork model) throws IOException {
        File modelFile = new File(modelSavePath);
        ModelSerializer.writeModel(model, modelFile, true);
    }

    private void saveToFile(int epoch, String[] samples) throws IOException {
        File file = new File(logFilePath);
        List<String> list = new ArrayList<>();
        list.add("Completed epoch " + epoch);
        list.addAll(Arrays.asList(samples));
        FileUtils.writeLines(file, list, true);
    }


    private MultiLayerNetwork createNetwork(int lstmLayerSize, CharacterIterator iter, int nOut) {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .l2(0.0001)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(0.005))
                .list()
                .layer(0, new LSTM.Builder().nIn(iter.inputColumns()).nOut(lstmLayerSize)
                        .activation(Activation.TANH).build())
                .layer(1, new LSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize)
                        .activation(Activation.TANH).build())
                .layer(2, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT).activation(Activation.SOFTMAX)        //MCXENT + softmax for classification
                        .nIn(lstmLayerSize).nOut(nOut).build())
                .backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(tbpttLength).tBPTTBackwardLength(tbpttLength)
                .build();


        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(1));
        return net;
    }

    private CharacterIterator getIterator(int miniBatchSize, int exampleLength) throws Exception {

        char[] validCharacters = CharacterIterator.getRussianCharacterSet();
        return new CharacterIterator(trainFilePath, Charset.forName("UTF-8"),
                miniBatchSize, exampleLength, validCharacters, new Random(12345));
    }


    /**
     * Generate a sample from the network, given an (optional, possibly null) initialization. Initialization
     * can be used to 'prime' the RNN with a sequence you want to extend/continue.<br>
     * Note that the initalization is used for all samples.txt
     *
     * @param initialization     String, may be null. If null, select a random character as initialization for all samples.txt
     * @param charactersToSample Number of characters to sample from network (excluding initialization)
     * @param net                MultiLayerNetwork with one or more GravesLSTM/RNN layers and a softmax output layer
     * @param iter               CharacterIterator. Used for going from indexes back to characters
     */
    private static String[] sampleCharactersFromNetwork(String initialization, MultiLayerNetwork net,
                                                        CharacterIterator iter, Random rng, int charactersToSample, int numSamples) {
        //Set up initialization. If no initialization: use a random character
        if (initialization == null) {
            initialization = String.valueOf(iter.getRandomCharacter());
        }

        //Create input for initialization
        INDArray initializationInput = Nd4j.zeros(numSamples, iter.inputColumns(), initialization.length());
        char[] init = initialization.toCharArray();
        for (int i = 0; i < init.length; i++) {
            int idx = iter.convertCharacterToIndex(init[i]);
            for (int j = 0; j < numSamples; j++) {
                initializationInput.putScalar(new int[]{j, idx, i}, 1.0f);
            }
        }

        StringBuilder[] sb = new StringBuilder[numSamples];
        for (int i = 0; i < numSamples; i++) sb[i] = new StringBuilder(initialization);

        //Sample from network (and feed samples back into input) one character at a time (for all samples)
        //Sampling is done in parallel here
        net.rnnClearPreviousState();
        INDArray output = net.rnnTimeStep(initializationInput);
        output = output.tensorAlongDimension((int) output.size(2) - 1, 1, 0);    //Gets the last time step output

        for (int i = 0; i < charactersToSample; i++) {
            //Set up next input (single time step) by sampling from previous output
            INDArray nextInput = Nd4j.zeros(numSamples, iter.inputColumns());
            //Output is a probability distribution. Sample from this for each example we want to generate, and add it to the new input
            for (int s = 0; s < numSamples; s++) {
                double[] outputProbDistribution = new double[iter.totalOutcomes()];
                for (int j = 0; j < outputProbDistribution.length; j++)
                    outputProbDistribution[j] = output.getDouble(s, j);
                int sampledCharacterIdx = sampleFromDistribution(outputProbDistribution, rng);

                nextInput.putScalar(new int[]{s, sampledCharacterIdx}, 1.0f);        //Prepare next time step input
                sb[s].append(iter.convertIndexToCharacter(sampledCharacterIdx));    //Add sampled character to StringBuilder (human readable output)
            }

            output = net.rnnTimeStep(nextInput);    //Do one time step of forward pass
        }

        String[] out = new String[numSamples];
        for (int i = 0; i < numSamples; i++) out[i] = sb[i].toString();
        return out;
    }

    /**
     * Given a probability distribution over discrete classes, sample from the distribution
     * and return the generated class index.
     *
     * @param distribution Probability distribution over classes. Must sum to 1.0
     */
    public static int sampleFromDistribution(double[] distribution, Random rng) {
        double d = 0.0;
        double sum = 0.0;
        for (int t = 0; t < 10; t++) {
            d = rng.nextDouble();
            sum = 0.0;
            for (int i = 0; i < distribution.length; i++) {
                sum += distribution[i];
                if (d <= sum) return i;
            }
            //If we haven't found the right index yet, maybe the sum is slightly
            //lower than 1 due to rounding error, so try again.
        }
        //Should be extremely unlikely to happen if distribution is a valid probability distribution
        throw new IllegalArgumentException("Distribution is invalid? d=" + d + ", sum=" + sum);
    }
}
