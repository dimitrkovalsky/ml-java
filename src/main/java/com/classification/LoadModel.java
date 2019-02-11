package com.classification;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;

import java.io.File;
import java.io.IOException;

public class LoadModel {
    public static void main(String[] args) throws IOException {
        File locationToSave = new File("C:\\Users\\Dmytro_Kovalskyi\\Downloads\\ml\\pretrained.zip");
        MultiLayerNetwork restored = ModelSerializer.restoreMultiLayerNetwork(locationToSave);
    }
}
