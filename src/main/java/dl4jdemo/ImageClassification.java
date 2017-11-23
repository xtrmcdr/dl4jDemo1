package dl4jdemo;

import java.io.File;
import java.util.Random;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.api.util.ClassPathResource;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Random;


public class ImageClassification {

    private static Logger log = LoggerFactory.getLogger(ImageClassification.class);
    public static void main(String[] args) throws Exception {
        int seed = 123;
        int height = 28;
        int width = 28;
        int channels = 3;
        int batchSize = 10;
        int numEpochs = 15;

        Random rand = new Random(seed);

        File topDir =new ClassPathResource("101_ObjectCategories").getFile();

        FileSplit filesInDir = new FileSplit(topDir, NativeImageLoader.ALLOWED_FORMATS , rand);

        //parse the parent dir and use the name of the subdirectories as label/class names
        ParentPathLabelGenerator labelGenerator = new ParentPathLabelGenerator();

        BalancedPathFilter pathFilter = new BalancedPathFilter(rand, NativeImageLoader.ALLOWED_FORMATS, labelGenerator);

        //Split the image files into train and test. Specify the train test split as 80%,20%
        InputSplit[] filesTrainTestSplit = filesInDir.sample(pathFilter, 80, 20);
        InputSplit train = filesTrainTestSplit[0];
        InputSplit test = filesTrainTestSplit[1];

        ImageRecordReader recordReader = new ImageRecordReader(height,width,channels,labelGenerator);


        //Initialize the record reader with the train data and the transform chain
        recordReader.initialize(train);
        int outputNum = recordReader.numLabels();


        int labelIndex = 1;

        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, labelIndex, outputNum);


        // Scale pixel values to 0-1

        DataNormalization scaler = new ImagePreProcessingScaler(0,1);
        scaler.fit(dataIter);
        dataIter.setPreProcessor(scaler);


        // Build Our Neural Network

        log.info("**** Build Model ****");

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(1)
                .learningRate(0.006)
                .updater(Updater.NESTEROVS)
                .regularization(true).l2(1e-4)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(width*height*channels)
                        .nOut(250)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(250)
                        .nOut(outputNum)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .pretrain(false).backprop(true)
                .setInputType(InputType.convolutional(height,width,channels))
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        model.setListeners(new ScoreIterationListener(10));

        log.info("*****TRAIN MODEL********");
        System.out.println("*****TRAIN MODEL********");
        for(int i = 0; i<numEpochs; i++){
            model.fit(dataIter);
        }

        log.info("******EVALUATE MODEL******");
        System.out.println("******EVALUATE MODEL******");
        recordReader.reset();

        recordReader.initialize(test);
        DataSetIterator testIter = new RecordReaderDataSetIterator(recordReader,batchSize,1,outputNum);
        scaler.fit(testIter);
        testIter.setPreProcessor(scaler);


        Evaluation eval = new Evaluation(outputNum);

        while(testIter.hasNext()){
            DataSet next = testIter.next();
            INDArray output = model.output(next.getFeatureMatrix());
            eval.eval(next.getLabels(),output);
        }

        log.info(eval.stats());
        System.out.println(eval.stats());


    }
}
