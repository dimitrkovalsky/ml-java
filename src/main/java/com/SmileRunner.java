package com;

import com.expessions.EmotionClassifier;
import com.smile.SmileClassifier;

import java.io.IOException;

public class SmileRunner {
    public static void main(String[] args) throws IOException {
        new EmotionClassifier();
//        new SmileClassifier().recognizeSmile("C:\\gitlab\\ml\\ml-java\\datasets\\merged\\smile\\file0005.jpg");
//        new SmileClassifier().recognizeSmile("C:\\gitlab\\ml\\ml-java\\datasets\\merged\\not_smile\\file2168.jpg");
//        new SmileClassifier().recognizeSmile("C:\\gitlab\\facial_expressions\\processed\\sadness\\Avinash_12.jpg");
    }
}
