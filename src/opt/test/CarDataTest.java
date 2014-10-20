package opt.test;

import dist.*;
import opt.*;
import opt.example.*;
import opt.ga.*;
import shared.*;
import util.linalg.Vector;
import func.nn.backprop.*;

import java.util.*;
import java.io.*;
import java.text.*;

/**
 * Implementation of randomized hill climbing, simulated annealing, and genetic algorithm to
 * find optimal weights to a neural network that is classifying car data.
 *
 * @author Hannah Lau (AbaloneTest)
 * @author Jeff Copeland
 * @version 1.0
 */

public class CarDataTest {
	private static int NUM_OUTPUT = 4;
    private static Instance[] instances = initializeInstances("c:/Users/mongi_000/Dropbox/GT/MachineLearning/project2/car-modfied-train80-binary-outfix.csv",NUM_OUTPUT);
    private static Instance[] testInstances = initializeInstances("c:/Users/mongi_000/Dropbox/GT/MachineLearning/project2/car-modfied-test20-binary-outfix.csv",NUM_OUTPUT);

    private static int inputLayer = 21, hiddenLayer = 12, outputLayer = NUM_OUTPUT, trainingIterations = 100001;
    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();
    
    private static ErrorMeasure measure = new SumOfSquaresError(); // HammingDistance(); //

    private static DataSet set = new DataSet(instances);
    private static DataSet testSet = new DataSet(testInstances);

    private static BackPropagationNetwork networks[] = new BackPropagationNetwork[3];
    private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[3];

    private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[3];
    private static String[] oaNames = {"RHC", "SA", "GA"};
    private static String results = "";

    private static DecimalFormat df = new DecimalFormat("0.000");

    public static void main(String[] args) {
        for(int i = 0; i < oa.length; i++) {
            networks[i] = factory.createClassificationNetwork(
                new int[] {inputLayer, hiddenLayer, outputLayer});
            nnop[i] = new NeuralNetworkOptimizationProblem(set, networks[i], measure);
        }

        oa[0] = new RandomizedHillClimbing(nnop[0]);
        oa[1] = new SimulatedAnnealing(1E11, .95, nnop[1]);
        oa[2] = new StandardGeneticAlgorithm(200, 100, 10, nnop[2]);

        //TODO: remove comment below when ready to process entire data
        for(int i = 0; i < oa.length; i++) {
            double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
            train(oa[i], networks[i], oaNames[i]); //trainer.train();
            end = System.nanoTime();
            trainingTime = end - start;
            trainingTime /= Math.pow(10,9);

            Instance optimalInstance = oa[i].getOptimal();
            networks[i].setWeights(optimalInstance.getData());

            start = System.nanoTime();
            for(int j = 0; j < instances.length; j++) {
                networks[i].setInputValues(instances[j].getData());
                networks[i].run();

				int predictedOutput = networks[i].getDiscreteOutputValue();
				if (instances[j].getLabel().getDiscrete(predictedOutput) == 1)
					correct++;
				else
					incorrect++;

            }
            end = System.nanoTime();
            testingTime = end - start;
            testingTime /= Math.pow(10,9);

            results +=  "\nResults for " + oaNames[i] + ": \nCorrectly classified " + correct + " instances." +
                        "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                        + df.format(correct/(correct+incorrect)*100) + "%\nTraining time: " + df.format(trainingTime)
                        + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";
        }

        System.out.println(results);
    }

    private static void train(OptimizationAlgorithm oa, BackPropagationNetwork network, String oaName) {
        System.out.println("\nError results for " + oaName + "\n---------------------------");

        double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
        double totalTrainTime = 0;
        for(int i = 0; i < trainingIterations; i++) {
        	start = System.nanoTime();
            oa.train();
            end = System.nanoTime();
            totalTrainTime += (end - start);

            if( (i % 1000) == 0) {
                double error = 0;
                int trainCorrect = 0;
                int trainIncorrect = 0;
                for(int j = 0; j < instances.length; j++) {
                    network.setInputValues(instances[j].getData());
                    network.run();
                    error += measure.value(new Instance(network.getOutputValues()), instances[j]);
                    
    				int predictedOutput = network.getDiscreteOutputValue();
    				if (instances[j].getLabel().getDiscrete(predictedOutput) == 1)
    					trainCorrect++;
    				else
    					trainIncorrect++;
                }

                double testError = 0;
                int testCorrect = 0;
                int testIncorrect = 0;
                for(int j = 0; j < testInstances.length; j++) {
                    network.setInputValues(testInstances[j].getData());
                    network.run();
                    testError += measure.value(new Instance(network.getOutputValues()), testInstances[j]);
                    
    				int predictedOutput = network.getDiscreteOutputValue();
    				if (testInstances[j].getLabel().getDiscrete(predictedOutput) == 1)
    					testCorrect++;
    				else
    					testIncorrect++;
                }
            	
            	System.out.println(i + "\t" + df.format(error) + "\t" + df.format((double)trainCorrect/(trainCorrect+trainIncorrect)) + "\t" + df.format((double)testCorrect/(testCorrect+testIncorrect)) + "\t" + df.format(totalTrainTime / Math.pow(10,9)));
            }
        }
    }

    private static Instance[] initializeInstances(String fn, int numOutputs) {

        double[][][] attributes = null;

        try {
            BufferedReader br;

            // Count lines and allocate space for storage
            br = new BufferedReader(new FileReader(new File(fn)));
            int lines = 0;
            while (br.readLine() != null) lines++;
            br.close();
            attributes = new double[lines][][];
            
            // Reopen file for actual reading of inputs 
            br = new BufferedReader(new FileReader(new File(fn)));
            int numColumns = 0;
            for(int i = 0; i < attributes.length; i++) {
            	String line = br.readLine();
                Scanner scan = new Scanner(line);
                scan.useDelimiter(",");
                if(i == 0) {
                	// On first pass through, count the number of columns
					try {
						while (scan.next() != null)
							numColumns++;
						
					} catch (NoSuchElementException e) {
						//pass
					}
                	
                	// Reset scanner for actual attribute reading
                	scan.close();
                	scan = new Scanner(line);
                    scan.useDelimiter(",");
                }
                attributes[i] = new double[2][];
                attributes[i][0] = new double[numColumns-numOutputs]; // input attributes
                attributes[i][1] = new double[numOutputs]; // output labels

                for(int j = 0; j < (numColumns-numOutputs); j++)
                    attributes[i][0][j] = Double.parseDouble(scan.next());

                for(int j = 0; j < numOutputs; j++)
                	attributes[i][1][j] = Double.parseDouble(scan.next());
                
                scan.close();
            }
        }
        catch(Exception e) {
            e.printStackTrace();
        }

        Instance[] instances = new Instance[attributes.length];

        for(int i = 0; i < instances.length; i++) {
            instances[i] = new Instance(attributes[i][0]);
            instances[i].setLabel(new Instance(attributes[i][1]));
        }

        return instances;
    }
}
