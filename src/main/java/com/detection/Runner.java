package com.detection;
import javax.swing.*;
import java.util.concurrent.Executors;
public class Runner {
    private static JFrame mainFrame = new JFrame();
//    private static  CarPredictor carDetector = new CarPredictor();

    public static void main(String[] args) throws Exception {

        ProgressBar progressBar = new ProgressBar(mainFrame, true);
        progressBar.showProgressBar("Loading model this make take several seconds!");
        UI ui = new UI();
        Executors.newCachedThreadPool().submit(()->{
            try {
                ui.initUI();
            } catch (Exception e) {
                throw new RuntimeException(e);
            } finally {
                progressBar.setVisible(false);
                mainFrame.dispose();
            }
        });
    }
}
