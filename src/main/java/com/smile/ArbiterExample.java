package com.smile;


import org.apache.commons.math3.distribution.IntegerDistribution;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.arbiter.MultiLayerSpace;
import org.deeplearning4j.arbiter.conf.updater.SgdSpace;
import org.deeplearning4j.arbiter.layers.*;
import org.deeplearning4j.arbiter.optimize.api.CandidateGenerator;
import org.deeplearning4j.arbiter.optimize.api.OptimizationResult;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.deeplearning4j.arbiter.optimize.api.adapter.ParameterSpaceAdapter;
import org.deeplearning4j.arbiter.optimize.api.data.DataSource;
import org.deeplearning4j.arbiter.optimize.api.saving.ResultReference;
import org.deeplearning4j.arbiter.optimize.api.saving.ResultSaver;
import org.deeplearning4j.arbiter.optimize.api.score.ScoreFunction;
import org.deeplearning4j.arbiter.optimize.api.termination.MaxCandidatesCondition;
import org.deeplearning4j.arbiter.optimize.api.termination.MaxTimeCondition;
import org.deeplearning4j.arbiter.optimize.api.termination.TerminationCondition;
import org.deeplearning4j.arbiter.optimize.config.OptimizationConfiguration;
import org.deeplearning4j.arbiter.optimize.generator.RandomSearchGenerator;
import org.deeplearning4j.arbiter.optimize.parameter.FixedValue;
import org.deeplearning4j.arbiter.optimize.parameter.continuous.ContinuousParameterSpace;
import org.deeplearning4j.arbiter.optimize.parameter.discrete.DiscreteParameterSpace;
import org.deeplearning4j.arbiter.optimize.parameter.integer.IntegerParameterSpace;
import org.deeplearning4j.arbiter.optimize.runner.IOptimizationRunner;
import org.deeplearning4j.arbiter.optimize.runner.LocalOptimizationRunner;
import org.deeplearning4j.arbiter.saver.local.FileModelSaver;
import org.deeplearning4j.arbiter.scoring.impl.EvaluationScoreFunction;
import org.deeplearning4j.arbiter.task.MultiLayerNetworkTaskCreator;
import org.deeplearning4j.arbiter.ui.listener.ArbiterStatusListener;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.storage.FileStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Properties;
import java.util.concurrent.TimeUnit;

public class ArbiterExample {
    protected static int channels = 3;

    public static void main(String[] args) throws InterruptedException, IOException {
        //First: Set up the hyperparameter configuration space. This is like a MultiLayerConfiguration, but can have either
        // fixed values or values to optimize, for each hyperparameter

        ParameterSpace<Double> learningRateHyperparam = new ContinuousParameterSpace(0.01, 0.1);  //Values will be generated uniformly at random between 0.0001 and 0.1 (inclusive)
        ParameterSpace<Integer> layerSizeHyperparam = new IntegerParameterSpace(32, 512);            //Integer values will be generated uniformly at random between 16 and 256 (inclusive)
        ParameterSpace<Integer> layerSizeHyperparam2 = new IntegerParameterSpace(128, 512);            //Integer values will be generated uniformly at random between 16 and 256 (inclusive)
        MultiLayerSpace hyperparameterSpace = new MultiLayerSpace.Builder()
                .seed(42)
                .l2(0.005) // tried 0.0001, 0.0005
                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new SgdSpace(learningRateHyperparam))
                .addLayer(convInitSpace(channels))
                .addLayer(maxPool())
                .addLayer(convInnerSpace())
                .addLayer(convInnerSpace())
                .addLayer(maxPool())
                .addLayer(new DenseLayerSpace.Builder()
                        .nOut(new IntegerParameterSpace(128, 1024))
                        .activation(new DiscreteParameterSpace<>(Activation.LEAKYRELU, Activation.SOFTMAX, Activation.RELU, Activation.SIGMOID, Activation.SOFTSIGN, Activation.SWISH))
                        .build())
//                .addLayer(new DropoutLayerSpace.Builder().dropOut(0.8).build())
                .addLayer(new OutputLayerSpace.Builder().nOut(2)
                        .lossFunction(new DiscreteParameterSpace<>(LossFunctions.LossFunction.SQUARED_LOSS, LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY, LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD))
                        .activation(new DiscreteParameterSpace<>(Activation.LEAKYRELU, Activation.SOFTMAX, Activation.RELU, Activation.SIGMOID, Activation.SOFTSIGN, Activation.SWISH))
                        .build())
                .setInputType(InputType.convolutional(32, 32, channels))
                .numEpochs(100)
                .build();


        CandidateGenerator candidateGenerator = new RandomSearchGenerator(hyperparameterSpace, null);    //Alternatively: new GridSearchCandidateGenerator<>(hyperparameterSpace, 5, GridSearchCandidateGenerator.Mode.RandomOrder);
        Class<? extends DataSource> dataSourceClass = SmilesDataSource.class;
        Properties dataSourceProperties = new Properties();
        dataSourceProperties.setProperty("minibatchSize", "30");
        String baseSaveDirectory = "arbiterExample/";
        File f = new File(baseSaveDirectory);
        if (f.exists()) f.delete();
        f.mkdir();
        ResultSaver modelSaver = new FileModelSaver(baseSaveDirectory);
        ScoreFunction scoreFunction = new EvaluationScoreFunction(Evaluation.Metric.ACCURACY);
        TerminationCondition[] terminationConditions = {
                new MaxTimeCondition(15, TimeUnit.MINUTES),
                new MaxCandidatesCondition(100)};
        OptimizationConfiguration configuration = new OptimizationConfiguration.Builder()
                .candidateGenerator(candidateGenerator)
                .dataSource(dataSourceClass, dataSourceProperties)
                .modelSaver(modelSaver)
                .scoreFunction(scoreFunction)
                .terminationConditions(terminationConditions)
                .build();

        IOptimizationRunner runner = new LocalOptimizationRunner(configuration, new MultiLayerNetworkTaskCreator());


        StatsStorage ss = new FileStatsStorage(new File("arbiterExampleUiStats.dl4j"));
        runner.addListeners(new ArbiterStatusListener(ss));
        UIServer.getInstance().attach(ss);
        runner.execute();

        String s = "Best score: " + runner.bestScore() + "\n" +
                "Index of model with best score: " + runner.bestScoreCandidateIndex() + "\n" +
                "Number of configurations evaluated: " + runner.numCandidatesCompleted() + "\n";
        System.out.println(s);


        //Get all results, and print out details of the best result:
        int indexOfBestResult = runner.bestScoreCandidateIndex();
        List<ResultReference> allResults = runner.getResults();

        OptimizationResult bestResult = allResults.get(indexOfBestResult).getResult();
        MultiLayerNetwork bestModel = (MultiLayerNetwork) bestResult.getResultReference().getResultModel();

        System.out.println("\n\nConfiguration of best model:\n");
        System.out.println(bestModel.getLayerWiseConfigurations().toJson());


        //Wait a while before exiting
        Thread.sleep(60000);
        UIServer.getInstance().stop();
    }

    private static ConvolutionLayerSpace convInitSpace(int in) {
        return new ConvolutionLayerSpace.Builder()
                .kernelSize(new int[] {5, 5})
                .stride(new int[] {1, 1})
                .padding(new int[] {0, 0})
                .nIn(in)
                .nOut(new IntegerParameterSpace(16, 128))
                .biasInit(0)
                .build();
    }

    private static SubsamplingLayerSpace maxPool() {
        return new SubsamplingLayerSpace.Builder().kernelSize(new int[] {2, 2}).build();
    }

    private static SubsamplingLayerSpace maxPool3() {
        return new SubsamplingLayerSpace.Builder().kernelSize(new int[] {3, 3}).build();
    }

    private static ConvolutionLayerSpace convInnerSpace() {
        return new ConvolutionLayerSpace.Builder()
                .kernelSize(new int[] {2, 2})
                .stride(new int[] {1, 1})
                .padding(new int[] {0, 0})
                .nOut(new IntegerParameterSpace(16, 128))
                .biasInit(new ContinuousParameterSpace(0, 1))
                .build();
    }
}