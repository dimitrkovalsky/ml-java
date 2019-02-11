package com;

import com.generation.LstmGenerator;

public class ShvechenkoGeneratorRunner {

    public static void main(String[] args) throws Exception {
        LstmGenerator generator = new LstmGenerator(
                "C:\\GitHub\\mljava\\src\\main\\resources\\kobzar-clean.txt",
                "shevchenkoModel",
                "shevchenko-samples.txt", 100);
     //   generator.run();
        for (String sample : generator.sample()) {
            System.out.println(sample);
        }
    }
}

