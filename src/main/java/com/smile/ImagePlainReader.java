package com.smile;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputStreamInputSplit;
import org.datavec.api.util.ndarray.RecordConverter;
import org.datavec.api.writable.IntWritable;
import org.datavec.api.writable.NDArrayWritable;
import org.datavec.api.writable.Writable;
import org.datavec.image.loader.ImageLoader;
import org.datavec.image.recordreader.BaseImageRecordReader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.PipelineImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.concurrency.AffinityManager;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImageFlatteningDataSetPreProcessor;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

public class ImagePlainReader {
    public static void main(String[] args) throws Exception {
        MultiLayerNetwork layerNetwork = SmileDetector.initPlainModel("plain.data");
        File trainDirectory = new File("C:\\gitlab\\ml\\ml-java\\datasets\\train_folder");
        File testDirectory = new File("C:\\gitlab\\ml\\ml-java\\datasets\\test_folder");
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        ImageTransform transform = new PipelineImageTransform();
            ImageRecordReader recordReader = new ImageRecordReader(64, 64, 3, labelMaker, transform);
        FileSplit trainSplit = new FileSplit(trainDirectory, new String[] {"jpg"});

        recordReader.initialize(trainSplit, null);
        RecordReaderDataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, 20, 1, 2);
        dataIter.setPreProcessor(new ImageFlatteningDataSetPreProcessor());
        layerNetwork.fit(dataIter, 1000);
    }
}
