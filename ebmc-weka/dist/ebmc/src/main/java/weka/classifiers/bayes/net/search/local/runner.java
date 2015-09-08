package weka.classifiers.bayes.net.search.local;

import java.io.File;
import java.io.IOException;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.BayesNet;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

public class runner {
	
public static void main(String args[]) throws Exception{
		
		try {
			ArffLoader arff = new ArffLoader();
			arff.setFile(new File("australian-train.arff"));
			Instances instances = arff.getDataSet();
			instances.setClassIndex(instances.numAttributes() - 1);
			
			System.out.println("Train: ");
			BayesNet bn = new BayesNet();
			
			bn.setSearchAlgorithm(new EBMC(8, 14, 13)); //K2
			//bn.setSearchAlgorithm(new EBMC(8, 8, 8, 2)); //BDeu
			bn.buildClassifier(instances);
			
			
			
			//System.out.println(bn.graph());
			
			
			//System.out.println("Test: ");
			Evaluation eval = new Evaluation(instances);
			eval.evaluateModel(bn, instances);
			
			//System.out.println(bn.m_Distributions[0][0]);
			
			
			
		} catch (IOException e) {
			e.printStackTrace();
		}
		
	}



}
