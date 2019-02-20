package com;

import com.expessions.EmotionClassifier;
import com.smile.SmileClassifier;

import java.io.IOException;

public class SmileRunner {
    public static void main(String[] args) throws IOException {
      //  new EmotionClassifier();
//        new SmileClassifier().recognizeSmile("C:\\gitlab\\ml\\ml-java\\datasets\\merged\\smile\\file0029.jpg");
        System.out.println("smile\t" + new EmotionClassifier().classify("C:\\gitlab\\ml\\ml-java\\datasets\\merged\\smile\\file0029.jpg"));
//        new SmileClassifier().recognizeSmile("C:\\gitlab\\facial_expressions\\processed\\neutral\\Adam_Ant_0001.jpg");
        System.out.println("neutral\t" + new EmotionClassifier().classify("C:\\gitlab\\facial_expressions\\processed\\neutral\\Adam_Ant_0001.jpg"));
//        new SmileClassifier().recognizeSmile("C:\\gitlab\\facial_expressions\\processed\\sadness\\Avinash_12.jpg");
        System.out.println("sadness\t" + new EmotionClassifier().classify("C:\\gitlab\\facial_expressions\\processed\\sadness\\Avinash_12.jpg"));
    }
}
