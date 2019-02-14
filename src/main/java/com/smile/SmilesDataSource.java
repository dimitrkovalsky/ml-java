package com.smile;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.FlipImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.deeplearning4j.arbiter.optimize.api.data.DataSource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImageFlatteningDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import java.io.File;
import java.io.IOException;
import java.util.*;

import static com.smile.SmileClassifier.DATA_FOLDER;

public class SmilesDataSource implements DataSource {
    private int minibatchSize;
    protected static double splitTrainTest = 0.7;

    @Override
    public void configure(Properties properties) {
        this.minibatchSize = Integer.parseInt(properties.getProperty("minibatchSize", "16"));
    }


    @Override
    public Object trainData() {
        File mainPath = new File(DATA_FOLDER);
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        Random rng = new Random();
        FileSplit fileSplit = new FileSplit(mainPath, NativeImageLoader.ALLOWED_FORMATS, rng);
        int numExamples = Math.toIntExact(fileSplit.length());

        BalancedPathFilter pathFilter = new BalancedPathFilter(rng, labelMaker, numExamples, 2, minibatchSize);

        InputSplit[] inputSplit = fileSplit.sample(pathFilter, splitTrainTest, 1 - splitTrainTest);
        InputSplit trainData = inputSplit[0];

        ImageRecordReader recordReader = new ImageRecordReader(32, 32, 3, labelMaker);
        try {
            recordReader.initialize(trainData, null);
        } catch (IOException e) {
            e.printStackTrace();
        }
        RecordReaderDataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, minibatchSize, 1, 2);
        ImagePreProcessingScaler preProcessor = new ImagePreProcessingScaler(0, 1);
        iterator.setPreProcessor(preProcessor);
        return iterator;
    }

    @Override
    public Object testData() {
        File mainPath = new File(DATA_FOLDER);
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        Random rng = new Random();
        FileSplit fileSplit = new FileSplit(mainPath, NativeImageLoader.ALLOWED_FORMATS, rng);
        int numExamples = Math.toIntExact(fileSplit.length());

        BalancedPathFilter pathFilter = new BalancedPathFilter(rng, labelMaker, numExamples, 2, minibatchSize);

        InputSplit[] inputSplit = fileSplit.sample(pathFilter, splitTrainTest, 1 - splitTrainTest);
        InputSplit testData = inputSplit[1];

        ImageRecordReader recordReader = new ImageRecordReader(32, 32, 3, labelMaker);
        ImageTransform flipTransform1 = new FlipImageTransform(rng);
        ImageTransform flipTransform2 = new FlipImageTransform(new Random(123));

        List<ImageTransform> transforms = Arrays.asList(flipTransform1, flipTransform2);

        List<DataSet> dataSets = new ArrayList<>();
        try {
            recordReader.initialize(testData, null);

            RecordReaderDataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, minibatchSize, 1, 2);
            ImagePreProcessingScaler preProcessor = new ImagePreProcessingScaler(0, 1);
            iterator.setPreProcessor(preProcessor);
            while (iterator.hasNext()) {
                DataSet next = iterator.next();
                dataSets.add(next);
            }
            for (ImageTransform transform : transforms) {
                recordReader.reset();
                recordReader.initialize(testData, transform);
                while (iterator.hasNext()) {
                    DataSet next = iterator.next();
                    dataSets.add(next);
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        ListDataSetIterator<DataSet> iterator = new ListDataSetIterator<>(dataSets);
        return iterator;
    }

    @Override
    public Class<?> getDataType() {
        return DataSetIterator.class;
    }
}
