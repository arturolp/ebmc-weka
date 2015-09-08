/*
 *   This program is under development
 */

/*
 * EBMC.java
 * Copyright (C) 2015 University of Pittsburgh, PA, USA
 * 
 */
package weka.classifiers.bayes.net.search.local;

import java.util.Enumeration;
import java.util.Vector;

import weka.classifiers.bayes.net.search.SearchAlgorithm;
import weka.classifiers.bayes.net.search.local.data.FileCaseRecord;
import weka.classifiers.bayes.net.search.local.data.NodeInfoRecord;
import weka.classifiers.bayes.BayesNet;
import weka.core.Instances;
import weka.core.Option;
import weka.core.RevisionUtils;
import weka.core.SelectedTag;
import weka.core.Tag;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;


/**
 * <!-- globalinfo-start --> This algorithm performs greedy search in a subspace
 * of Bayesian Networks to find the one that best predicts a target node.<br/>
 * <br/>
 * For more information refer to:<br/>
 * <br/>
 * G. F. Cooper, P. Hennings-Yeomans, S. Visweswaran, & M. Barmada, (2010). An
 * efficient Bayesian method for predicting clinical outcomes from genome-wide
 * data. AMIA Annual Symposium Proceedings. 127-131.
 * <p/>
 * <!-- globalinfo-end -->
 * 
 * <!-- technical-bibtex-start --> BibTeX:
 * 
 * <pre>
 * &#64;inproceedings{Cooper 2010,
 *    author = {G.F. Cooper, P. Hennings-Yeomans, S. Visweswaran, & M. Barmada},
 *    pages = {127-131},
 *    publisher = {AMIA Annual Symposium Proceedings},
 *    title = {An efficient Bayesian method for predicting clinical outcomes from genome-wide data},
 *    year = {2010}
 * }
 * </pre>
 * <p/>
 * <!-- technical-bibtex-end -->
 * 
 * <!-- options-start --> Valid options are:
 * <p/>
 * 
 * <pre>
 * -D
 *  If set, classifier is run in debug mode and
 *  may output additional info to the console
 * </pre>
 * 
 * <!-- options-end -->
 * 
 * @author Arturo Lopez Pineda (arl68@pitt.edu)
 * with functions from Kevin V. Bui (kvb2@pitt.edu)
 * @version $Revision: 1.0 $
 */
public class EBMC extends SearchAlgorithm implements TechnicalInformationHandler {

	protected static final int MAXNEWCHILDREN = 10;  // maximum new children that any node is allowed to have in a "rule"
	protected static final int MININUM_EXPONENT = -1022;
	/**
	 * Holds prior on count
	 */
	//double m_fAlpha = 0.5;
	protected static final double MINUS_INFINITY =  -1.0e308;

	/** for serialization */
	static final long serialVersionUID = 6176545934752116631L;

	protected byte[][] cases;  // contains all the training (and possibly testing) data
	protected boolean childJustAddedFlag = false;
	protected boolean[] childPresent;

	protected int[] children;
	protected int[] counts;  // the counts at the leaves of the tree created by FileCase

	protected int[] countsTree;  // the counts at the leaves of the tree created by FileCase
	protected int countsTreePtr, countsPtr;
	protected FileCaseRecord[] fileCaseCache;

	protected int firstCase, lastCase;
	protected Instances inst;

	protected double[][][] lnChildProb;
	protected double[][] lnChildrenProb;

	protected double lnTotalProb;
	protected int lowerBound;  // usually 1 -- first case

	protected String my_model= "";

	/** points to Bayes network for which a structure is searched for **/
	BayesNet m_BayesNet;
	/*** Override the initialization as naive Bayes to False */
	//private final boolean m_bInitAsNaiveBayes = false;
	/*** Overrides the Markov blanket correction */
	//private final boolean m_bMarkovBlanketClassifier = false;
	/*** Overrides the scoring metric to prequential */
	//private int m_nScoreType = 5;
	/*** The expected parents of target  */
	private int m_ExpectedParentsOfTarget=2;
	/*** The maximum number of children that any node is allowed to have */
	private int m_MaxNrOfChildren=2;

	/*** The maximum number of parents that any node is allowed to have */
	private int m_MaxNrOfParents=2;

	/*** The scoring metric to use */
	private int m_ScoreMetric = 0; // 0 is K2, 1 is BDeu

	/*** The prior equivalent sample size value, when usign BDEu */
	private int m_PriorEquivalentSampleSize;

	protected int[] map;

	protected int maxCell;
	protected int maxChildren;
	protected int maxParents;

	protected int maxValue;
	protected boolean[] newChildPresent;

	protected int[] newChildren;

	protected byte[] nodeDimension;

	protected NodeInfoRecord[] nodeInfo;


	protected double[] nodeProb;

	protected double nodeScore;
	protected int numberOfCases;
	protected int numberOfChildren;
	protected int numberOfModelsScored;

	protected int numberOfNodes;
	protected boolean[] parentPresent;
	protected int[][] parents;  // note: parents[i, 0] represents the number of parents of node i
	protected double priorSampleSize;

	private Tag[] SCORING_METRICS={
			new Tag(0, "K2"),
			new Tag(1, "BDeu")
	};
	protected int[] targetCounts;



	public int targetNode;  // the outcome node being predicted
	protected int totalModelsScored;



	// the range of cases for training and testing
	protected int upperBound;  // usually the number of cases

	protected int[] values;  // for a given case, it contains the values of the parents of the target and then value of the target

	/**
	 * default constructor
	 */
	public EBMC() {
	} // c'tor

	/**
	 * default constructor
	 */
	public EBMC(int predictors, int max_parents, int max_children) {
		setExpectedParentsOfTarget(predictors);
		setMaxNrOfParents(max_parents);
		setMaxNrOfChildren(max_children);
	} // c'tor

	/**
	 * default constructor
	 */
	public EBMC(int predictors, int max_parents, int max_children, int pess) {
		setExpectedParentsOfTarget(predictors);
		setMaxNrOfParents(max_parents);
		setMaxNrOfChildren(max_children);
		//setScoreMetric(new SelectedTag("BDeu", SCORING_METRICS));
		m_ScoreMetric=1; //for BDeu
		setPriorEquivalentSampleSize(pess);
	} // c'tor

	/**
	 * constructor
	 * 
	 * @param bayesNet the network
	 * @param instances the data

	public EBMC(BayesNet bayesNet, Instances instances) {
		m_BayesNet = bayesNet;
		//		m_Instances = instances;
	}*/ // c'tor

	protected void addChild(int child) {
		newChildPresent[child] = true;

		// When "child" is already part of the permanent model, we make it a parent
		// of all the nodes in the current newChildren rule. By doing so, we avoid
		// having to add new parents to existing nodes in the permanent model, which
		// might significantly disrupt the model score in a bad way.
		if (childPresent[child]) {
			childJustAddedFlag = false;

			for (int i = 1; i <= newChildren[0]; i++) {
				int existingNewChild = newChildren[i];

				if (parents[existingNewChild][0] < maxParents) {
					removeNodeScores(existingNewChild);
					parents[existingNewChild][0]++;
					parents[existingNewChild][parents[existingNewChild][0]] = child;
					incorporateNodeScores(existingNewChild);
				}
			}
		} else {  // when child is not part of the permanent model, we make it a child of each the nodes in the current newChildren rule
			childJustAddedFlag = true;
			numberOfChildren++;
			//System.out.println("numberOfChildren = "+numberOfChildren);
			children[numberOfChildren] = child;
			map[child] = numberOfChildren;
			childPresent[child] = true;
			parents[child][0] = 1;
			parents[child][parents[child][0]] = targetNode;

			int min = Math.min(newChildren[0], maxParents - 1);
			for (int i = 1; i <= min; i++) {
				int existingNewChild = newChildren[i];
				parents[child][0]++;
				parents[child][parents[child][0]] = existingNewChild;
			}

			newChildren[0]++;
			newChildren[newChildren[0]] = child;
			incorporateNodeScores(child);
		}
	}


	/**
	 * Adds a given parent to a node
	 *
	 * @param node
	 * @return void

	protected void addParent(int node, int parent){
		if(!isParentXofY(parent, node) && node != parent){
			m_BayesNet.getParentSet(node).addParent(parent, inst);
		}
	}*/

	/**
	 * Calc Node Score With AddedParent
	 * 
	 * @param nNode node for which the score is calculate
	 * @param nCandidateParent candidate parent to add to the existing parent set
	 * @return log score

	protected double calcScoreWithExtraParent(int nNode, int nCandidateParent) {
		ParentSet oParentSet = m_BayesNet.getParentSet(nNode);

		// sanity check: nCandidateParent should not be in parent set already
		if (oParentSet.contains(nCandidateParent)) {
			return -1e100;
		}

		// set up candidate parent
		oParentSet.addParent(nCandidateParent, m_BayesNet.m_Instances);

		// calculate the score
		double logScore = scoreNode(targetNode) + prior(targetNode);

		// delete temporarily added parent
		oParentSet.deleteLastParent(m_BayesNet.m_Instances);

		return logScore;
	} // CalcScoreWithExtraParent
	 */


	/**
	 * Calc Node Score With Parent Deleted
	 * 
	 * @param nNode node for which the score is calculate
	 * @param nCandidateParent candidate parent to delete from the existing parent set
	 * @return log score

	protected double calcScoreWithMissingParent(int nNode, int nCandidateParent) {
		ParentSet oParentSet = m_BayesNet.getParentSet(nNode);

		// sanity check: nCandidateParent should be in parent set already
		if (!oParentSet.contains( nCandidateParent)) {
			return -1e100;
		}

		// set up candidate parent
		int iParent = oParentSet.deleteParent(nCandidateParent, m_BayesNet.m_Instances);

		// calculate the score
		double logScore = scoreNode(targetNode) + prior(targetNode);;

		// restore temporarily deleted parent
		oParentSet.addParent(nCandidateParent, iParent, m_BayesNet.m_Instances);

		return logScore;
	} // CalcScoreWithMissingParent
	 */

	/**
	 * Remove the parents of a node, keeps children
	 *
	 * @param node
	 * @return

	public void deleteAllParents(int node){

		int parents[] = m_BayesNet.getParentSet(node).getParents();

		for(int i=0; i < parents.length; i++){
			m_BayesNet.getParentSet(node).deleteParent(parents[i], inst);
		}
	}*/

	/**
	 * Removes a given parent from a node
	 *
	 * @param node
	 * @return

	protected void deleteParent(int node, int parent){
		if(isParentXofY(parent, node) && node != parent){
			m_BayesNet.getParentSet(node).deleteParent(parent, inst);
		}
	}*/

	/**
	 * Derives the probability distribution over the target node
	 * given the values of its parents in casei. The cases from
	 * 1 to casei-1 are used to parameterize this predictive distribution.
	 *
	 * @param casei
	 * @param node
	 * @param k
	 * @param v
	 */
	protected void deriveNodeProbs(int casei, int node, double k, double v) {
		int numberOfParents, parentValue;
		int ctPtr, cPtr, ptr;

		numberOfParents = parents[node][0];

		for (int i = 1; i <= numberOfParents; i++)
			values[i] = cases[casei][parents[node][i]];

		ctPtr = 1;
		for (int i = 1; i <= numberOfParents; i++) {
			parentValue = values[i];
			ptr = countsTree[ctPtr + parentValue - 1];

			if (ptr > 0)
				ctPtr = ptr;
			else {  // there are no previous cases that match the current parent values of node, so return a uniform distribution
				for (int nodeValue = 1; nodeValue <= nodeDimension[node]; nodeValue++)
					nodeProb[nodeValue] = k / v;

				return;
			}
		}
		cPtr = ctPtr;

		double b = 0;
		for (int nodeValue = 1; nodeValue <= nodeDimension[node]; nodeValue++)
			b += counts[cPtr + nodeValue - 1];

		for (int nodeValue = 1; nodeValue <= nodeDimension[node]; nodeValue++) {
			double a = counts[cPtr + nodeValue - 1];
			nodeProb[nodeValue] = (a + k) / (b + v); // nodeProb[nodeValue] = P(node = nodeValue | parents_node, case1,...,casei-1)
		}
	}

	/**
	 * @return a string to describe the expectedParentsOfTarget option.
	 */
	public String expectedParentsOfTargetTipText() {
		return "Set the number of parents that the target node is expected to have.";
	} // expectedParentsOfTarget

	/**
	 * This creates a tree that stores and indexes cases.
	 * The branches of the tree correspond values of the parent nodes.
	 * The leaves of the tree correspond to counts of target given the parent values.
	 *
	 * @param node
	 * @param casei
	 */
	protected void fileCase(int node, int casei) {
		int parent = 0;
		int parentValue = 0;
		int cPtr = 0;
		int parenti = 0;
		int nodeValue = cases[casei][node];
		int numberOfParents = parents[node][0];
		int ctPtr = 1;
		int ptr = 1;

		for (int i = 1; i <= numberOfParents; i++) {
			parent = parents[node][i];
			parentValue = cases[casei][parent];
			ptr = countsTree[ctPtr + parentValue - 1];

			if (ptr > 0)
				ctPtr = ptr;
			else {
				parenti = i;
				break;
			}
		}

		if (ptr > 0) {
			cPtr = ctPtr;
			counts[cPtr + nodeValue - 1]++;
		} else {
			// GrowBranch
			for (int i = parenti; i <= numberOfParents; i++) {
				parent = parents[node][i];
				parentValue = cases[casei][parent];

				if (i == numberOfParents)
					countsTree[ctPtr + parentValue - 1] = countsPtr;
				else {
					countsTree[ctPtr + parentValue - 1] = countsTreePtr;

					for (int j = countsTreePtr; j <= (countsTreePtr + nodeDimension[parents[node][i + 1]] - 1); j++)
						countsTree[j] = 0;

					ctPtr = countsTreePtr;
					countsTreePtr += nodeDimension[parents[node][i + 1]];

				}
			}

			for (int j = countsPtr; j <= (countsPtr + nodeDimension[node] - 1); j++)
				counts[j] = 0;

			cPtr = countsPtr;

			countsPtr += nodeDimension[node];

			// end of GrowBranch
			counts[cPtr + nodeValue - 1]++;
		}  // end of else statement
	}//fileCase

	/**
	 * Returns the children for a given node
	 *
	 * @param node
	 * @return

	protected int[] getChildren(int node){
		int[] children = new int[getNrOfChildren(node)];

		int index = 0;

		for(int i = 0; i < m_BayesNet.getNrOfNodes(); i++){
			if (m_BayesNet.getParentSet(i).contains(node)){
				children[index] = i;
				index++;
			}
		}

		return children;
	}*/

	/**
	 * Gets the Expected Parents of Target
	 *
	 * @return the Exp
	 */
	public int getExpectedParentsOfTarget() {
		return m_ExpectedParentsOfTarget;
	}



	/**
	 * Gets whether to init as naive bayes
	 *
	 * @return whether to init as naive bayes

	public boolean getInitAsNaiveBayes() {
		return m_bInitAsNaiveBayes;
	}*/

	/**
	 * Gets whether the network will be corrected for Markov Blanket
	 *
	 * @return true/false

	public boolean getMarkovBlanketClassifier() {
		return m_bMarkovBlanketClassifier;
	}*/

	/**
	 * Gets the Maximum number of Children
	 *
	 * @return the maxChildren
	 */
	public int getMaxNrOfChildren() {
		return m_MaxNrOfChildren;
	}



	/**
	 * Gets the Maximum number of parents
	 *
	 * @return the maxParents
	 */
	public int getMaxNrOfParents() {
		return m_MaxNrOfParents;
	}

	protected byte getMaxValue() {
		byte max = 0;

		for (byte value : nodeDimension)
			if (max < value)
				max = value;

		return max;
	}

	/**
	 * Returns the number of children for a given node
	 *
	 * @param node
	 * @return
	 */
	protected int getNrOfChildren(int node){
		int numberOfChildren = 0;

		for(int i = 0; i < m_BayesNet.getNrOfNodes(); i++){
			if (m_BayesNet.getParentSet(i).contains(node)){
				numberOfChildren++;
			}
		}

		return numberOfChildren;
	}

	/**
	 * Returns the number of parents for a given node
	 *
	 * @param node
	 * @return
	 */
	protected int getNrOfParents(int node){
		return m_BayesNet.getParentSet(node).getNrOfParents();
	}

	/**
	 * Gets the current settings of the search algorithm.
	 *
	 * @return an array of strings suitable for passing to setOptions
	 */
	public String [] getOptions() {
		Vector<String> result  = new Vector<String>();

		result.add("-T"); // expected parents of target
		result.add("" + getExpectedParentsOfTarget());

		result.add("-P"); // Maximum number of parents
		result.add("" + getMaxNrOfParents());

		result.add("-C"); // Maximum number of children
		result.add("" + getMaxNrOfChildren());

		result.add("-S"); // scoring metric

		switch (m_ScoreMetric) {
		case(0):
			result.add("K2");
		break;
		case(1):
			result.add("BDeu");
		result.add("-E"); // prior equivalent sample size
		result.add("" + getPriorEquivalentSampleSize());
		break;
		}

		return (String[]) result.toArray(new String[result.size()]);
	}

	/**
	 * Returns the i-th parent of a node
	 *
	 * @param node
	 * @return

	protected int getParent(int node, int index){
		return m_BayesNet.getParentSet(node).getParent(index);
	}*/


	/**
	 * Returns the parents for a given node
	 *
	 * @param node
	 * @return

	protected int[] getParents(int node){
		return m_BayesNet.getParentSet(node).getParents();
	}*/

	/**
	 * Returns the revision string.
	 * 
	 * @return		the revision
	 */
	public String getRevision() {
		return RevisionUtils.extract("$Revision: 8034 $");
	}

	/**
	 * get quality measure to be used in searching for networks.
	 * @return quality measure
	 */
	public SelectedTag getScoreMetric() {
		return new SelectedTag(m_ScoreMetric, SCORING_METRICS);
	}

	/**
	 * Returns an instance of a TechnicalInformation object, containing 
	 * detailed information about the technical background of this class,
	 * e.g., paper reference or book this class is based on.
	 * 
	 * @return the technical information about this class
	 */
	public TechnicalInformation getTechnicalInformation() {
		TechnicalInformation result;

		result = new TechnicalInformation(Type.INPROCEEDINGS);
		result.setValue(Field.AUTHOR, "G.F. Cooper and P. Hennings-Yeomans and "
				+ "S. Visweswaran and M. Barmada");
		result.setValue(Field.TITLE, "An efficient bayesian method for predicting "
				+ "clinical outcomes from genome-wide data");
		result.setValue(Field.YEAR, "2010");
		result.setValue(Field.PAGES, "127-131");
		result.setValue(Field.PUBLISHER, "AMIA Anual Symposium Proceedings");

		return result;
	}

	/**
	 * This will return a string describing the search algorithm.
	 * @return  a description of the data generator suitable for displaying in
	 *         the explorer/experimenter gui
	 */
	public String globalInfo() {

		return "This algorithm performs greedy search in a subspace of Bayesian "
				+ "Networks to find the one that best predicts a target node.\n\n"
				+ "For more information refer to:\n\n"
				+ getTechnicalInformation().toString();
	}

	/**
	 * Check if a given node has children
	 *
	 * @param node
	 * @return

	protected boolean hasChildren(BayesNet bn, int node){
		boolean children = false;
		int i=0;
		while( children == false & (i < bn.getNrOfNodes()) ){
			if (bn.getParentSet(i).contains(node)){
				children = true;
			}
			i++;
		}
		return children;
	}*/

	/**
	 * Check if a given node has parents
	 *
	 * @param node
	 * @param bn the model
	 * @return boolean state if the node has parents 

	protected boolean hasParents(BayesNet bn, int node){
		boolean parents = false;
		if (bn.getParentSet(node).getNrOfParents() > 0){
			parents = true;
		}
		return parents;
	}*/


	/**
	 * This function is similar to ScoreNode, except it just incorporates
	 * the contribution for node to lnChildrenProb.
	 *
	 * @param node
	 */
	protected void incorporateNodeScores(int node) {
		initializeFileCase(node);

		// this builds a frequency tree
		for (int casei = firstCase; casei <= lastCase; casei++)
			fileCase(node, casei);

		// zero the counts in the frequency tree, but keep the tree
		for (int i = 1; i <= countsPtr; i++)
			counts[i] = 0;

		double a;
		double b;
		//if (getScoreMetric().getSelectedTag().getIDStr().equals("0")) { //K2 is selected
		if (m_ScoreMetric==0) { //K2 is selected
			a = 1;
			b = (double) nodeDimension[node];
		} else {
			a = priorSampleSize / numberOfJointStates(node);
			b = a * (double) nodeDimension[node];
		}

		// derive the conditional probabilities of node
		for (int casei = firstCase; casei <= lastCase; casei++) {
			byte saveTargetValue = cases[casei][targetNode];
			int nodeValue_casei = cases[casei][node];

			for (int targetValue = 1; targetValue <= nodeDimension[targetNode]; targetValue++) {
				cases[casei][targetNode] = (byte) targetValue;
				deriveNodeProbs(casei, node, a, b);
				double lnProb = Math.log(nodeProb[nodeValue_casei]);

				lnChildrenProb[casei][targetValue] += lnProb;
				lnChildProb[casei][targetValue][map[node]] = lnProb;
			}

			cases[casei][targetNode] = saveTargetValue;
			fileCase(node, casei);
		}
	}

	/**
	 * @return a string to describe the InitAsNaiveBayes option.
	 */
	public String initAsNaiveBayesTipText() {
		return "Not used in this method, set as False.";
	}

	protected void initializeFileCase(int node) {
		if (parents[node][0] > 0) {
			int firstParentSize = nodeDimension[parents[node][1]];

			for (int i = 1; i <= firstParentSize; i++)
				countsTree[i] = 0;

			countsTreePtr = firstParentSize + 1;
			countsPtr = 1;
		} else {
			countsTreePtr = 1;
			countsPtr = nodeDimension[node] + 1;

			for (int i = 1; i <= nodeDimension[node]; i++)
				counts[i] = 0;
		}
	}



	protected void initVariables(){

		readNodeInfo(numberOfNodes);

		priorSampleSize = getPriorEquivalentSampleSize();
		maxParents = getMaxNrOfParents();
		maxChildren = getMaxNrOfChildren();

		maxValue = getMaxValue();
		maxCell = maxParents * maxValue * numberOfCases;
		map = new int[numberOfNodes + 1];
		parents = new int[numberOfNodes + 1][maxParents + 1];
		values = new int[numberOfNodes + 1];

		newChildren = new int[MAXNEWCHILDREN + 1];

		children = new int[maxChildren + 1];
		nodeProb = new double[maxValue + 1];
		targetCounts = new int[maxValue + 1];

		childPresent = new boolean[numberOfNodes + 1];
		parentPresent = new boolean[numberOfNodes + 1];
		parents = new int[numberOfNodes + 1][maxParents + 1];
		for (int i = 1; i <= numberOfNodes; i++) {
			childPresent[i] = false;
			parentPresent[i] = false;
			parents[i][0] = 0;
		}

		fileCaseCache = new FileCaseRecord[maxChildren + 1];
		for (int i = 1; i <= maxChildren; i++)
			fileCaseCache[i] = new FileCaseRecord(maxCell);

		counts = new int[maxCell + 1];
		countsTree = new int[maxCell + 1];
		lnChildProb = new double[numberOfCases + 1][maxValue + 1][maxChildren + 1];

		numberOfModelsScored = 0;
		numberOfChildren = 0;

		childPresent = new boolean[numberOfNodes + 1];
		newChildPresent = new boolean[numberOfNodes + 1];
		parents = new int[numberOfNodes + 1][maxParents + 1];
		for (int i = 1; i <= numberOfNodes; i++) {
			childPresent[i] = false;
			newChildPresent[i] = false;
			parents[i][0] = 0;
		}

		firstCase = lowerBound = 1;
		lastCase = upperBound = numberOfCases;

		targetNode = numberOfNodes;

		lnChildrenProb = new double[numberOfCases + 1][maxValue + 1];
		for (int casei = 1; casei <= lastCase; casei++)
			for (int j = 1; j <= nodeDimension[targetNode]; j++)
				lnChildrenProb[casei][j] = 0;
	}




	/**
	 * Check if a given node has parents
	 *
	 * @param node
	 * @return

	protected boolean isChildrenXofY(int xNode, int yNode){
		boolean isChildren = false;
		if (m_BayesNet.getParentSet(xNode).contains(yNode)){
			isChildren = true;
		}
		return isChildren;
	}*/

	/**
	 * Check if a given node has parents
	 *
	 * @param node
	 * @return

	protected boolean isParentXofY(int xNode, int yNode){
		boolean isParent = false;
		if(m_BayesNet.getParentSet(yNode).contains(xNode)){
			isParent = true;
		}
		return isParent;
	}*/


	/**
	 * Returns an enumeration describing the available options.
	 *
	 * @return an enumeration of all the available options.
	 */
	public Enumeration<Option> listOptions() {
		Vector<Option> newVector = new Vector<Option>(0);

		newVector.addElement(new Option("\tExpected parents of target", "T", 2, 
				"-P <nr of parents>"));
		newVector.addElement(new Option("\tMaximum number of parents", "P", 2, 
				"-P <nr of parents>"));
		newVector.addElement(new Option("\tMaximum number of children", "C", 2, 
				"-C <nr of children>"));
		newVector.addElement(new Option("\tPrior Equivalent Sample Size", "E", 0, 
				"-E <pess>"));
		newVector.addElement(new Option("\tScoring Metric", "S", 0, 
				"-S <scoring metric>"));

		Enumeration<Option> enu = listOptions();
		while (enu.hasMoreElements()) {
			newVector.addElement(enu.nextElement());
		}
		return newVector.elements();

	}

	/**
	 * Takes ln(x) and ln(y) as input, and returns ln(x + y)
	 *
	 * @param lnX is natural log of x
	 * @param lnY is natural log of y
	 * @return natural log of x plus y
	 */
	protected double lnXpluslnY(double lnX, double lnY) {
		double lnYminusLnX, temp;

		if (lnY > lnX) {
			temp = lnX;
			lnX = lnY;
			lnY = temp;
		}

		lnYminusLnX = lnY - lnX;

		if (lnYminusLnX < MININUM_EXPONENT)
			return lnX;
		else
			return Math.log1p(Math.exp(lnYminusLnX)) + lnX;
	}


	/**
	 * @return a string to describe the MarkovBlanketClassifier option.
	 */
	public String markovBlanketClassifierTipText() {
		return "Not used in this method, set as False.";
	}



	/**
	 * @return a string to describe the MaxNrOfChildren option.
	 */
	public String maxNrOfChildrenTipText() {
		return "Set the maximum number of children that each node can have.";
	} // maxNrOfChildrenTipText

	/**
	 * @return a string to describe the MaxNrOfParents option.
	 */
	public String maxNrOfParentsTipText() {
		return "Set the maximum number of parents that each node can have.";
	} // maxNrOfParentsTipText


	/**
	 * @return a string to describe the MaxNrOfParents option.
	 */
	public String priorEquivalentSampleSizeTipText() {
		return "Prior equivalent sample size used in the BDeu scoring measure only. This value is not needed in the K2 scoring.";
	} // maxNrOfParentsTipText

	/**
	 * Moves the parents of the target node to be children of the target node
	 * and adds appropriate links among the children.
	 * There may already be other children as well. If so, the new children
	 * are integrated appropriately with the existing children.
	 *
	 * @return true

	protected boolean moveParentsToChildren() {

		boolean structureGrew = false;

		for (int i = 0; i < m_BayesNet.getNrOfParents(targetNode); i++) {
			int node = m_BayesNet.getParent(targetNode, i);

			if (childPresent[node]) {
				System.out.println("already there");
				if (!redundant(node)) {
					//updateChild(node);
					structureGrew = true;;
				}
			} else {
				System.out.println("parent added: "+node);
				addChild(node); // addChild(node);
				structureGrew = true;
			}
		}
		return structureGrew;
	}
	 */


	protected double numberOfJointStates(int node) {
		double x = (double) nodeDimension[node];

		for (int i = 1; i <= parents[node][0]; i++)
			x *= (double) nodeDimension[parents[node][i]];

		return x;
	}



	/*
	 * protected String print(BayesNet bn){
	 *
		String model = "";
		//Initialize BayesNet to single variable
		for(int iNode = 0; iNode < bn.getNrOfNodes(); iNode++) {
			if(hasParents(bn, iNode) || hasChildren(bn, iNode) || iNode == targetNode){
				model += bn.getNodeName(iNode) + " ("+bn.getCardinality(iNode)+")";
				if(bn.getParentSet(iNode).getNrOfParents() > 0){
					model += " --> ";
				}
				for(int jNode = 0; jNode < bn.getParentSet(iNode).getNrOfParents(); jNode++){
					int pNode = bn.getParentSet(iNode).getParent(jNode);

					model += bn.getNodeName(pNode);
					if((jNode+1) < bn.getParentSet(iNode).getNrOfParents()){
						model += ", ";
					}
				}

				model += "\n";
			}
			else{
				model += bn.getNodeName(iNode) + " ("+bn.getCardinality(iNode)+")\n";
			}
		}
		return model;
		//System.out.println("Score: " + scoreNode(targetNode));
		//System.out.println("----");
	}*/


	/*protected void printFull(BayesNet bn){
		//Initialize BayesNet to single variable
		for(int iNode = 0; iNode < bn.getNrOfNodes(); iNode++) {
			System.out.print("Node["+iNode+"]: "+bn.getNodeName(iNode) + " ("+bn.getCardinality(iNode)+")");
			if(bn.getParentSet(iNode).getNrOfParents() > 0){
				System.out.print(" --> ");
			}
			for(int jNode = 0; jNode < bn.getParentSet(iNode).getNrOfParents(); jNode++){
				int pNode = bn.getParentSet(iNode).getParent(jNode);

				System.out.print(bn.getNodeName(pNode));
				if((jNode+1) < bn.getParentSet(iNode).getNrOfParents()){
					System.out.print(", ");
				}
			}

			System.out.println("");
		}

		System.out.println("----");
	}*/

	protected double prior() {
		double nc = (double) numberOfChildren;
		double nn = (double) (numberOfNodes - 1);
		double p = ((double) getExpectedParentsOfTarget()) / nn;  // nn equals the number of potential predictors of the target

		return (nc * Math.log(p) + (nn - nc) * Math.log(1 - p));
	}


	/*protected double prior(int node) {
		double np = (double) m_BayesNet.getNrOfParents(node);
		double nc = (double) getNrOfChildren(node);
		double nn = (double) (m_BayesNet.getNrOfNodes() - 1);
		double p = ((double) m_ExpectedParentsOfTarget) / nn;  // nn equals the number of potential predictors of the target

		return ((np + nc) * Math.log(p) + (nn - nc - np) * Math.log(1 - p));
	}*/

	protected void pruneParents(int node) {
		double modelScore = tallyModelScore();
		double bestScore = 1;
		double score = 0;

		while (parents[node][0] > 0) {
			int numberOfParents = parents[node][0];
			int besti = 0;

			// remove i as a parent of node
			for (int i = 1; i <= numberOfParents; i++) {
				int parent = parents[node][i];

				for (int j = (i + 1); j <= numberOfParents; j++)
					parents[node][j - 1] = parents[node][j];

				parents[node][0]--;

				score = scoreNode2(node);
				if ((score > bestScore) || (bestScore == 1)) {
					besti = i;
					bestScore = score;
				}

				// add i back as a parent of node
				for (int j = numberOfParents; j >= (i + 1); j--)
					parents[node][j] = parents[node][j - 1];

				parents[node][i] = parent;
				parents[node][0]++;
			}

			if (bestScore > modelScore) {
				//if(m_BayesNet.getDebug() == true){
					System.out.println("Pruning node "+nodeInfo[parents[node][besti]].name+" away from node "+nodeInfo[node].name);
				//}
				// remove the parent that in doing so most increases the score
				for (int j = (besti + 1 ); j <= numberOfParents; j++)
					parents[node][j - 1] = parents[node][j];

				parents[node][0]--;
				modelScore = bestScore;
			} else
				break;
		}  // end of while-loop

		score = scoreNode2(node);
	}

	/**
	 * Extract data from a particular ARFF file.
	 *
	 */
	protected void readCases(){

		numberOfNodes = m_BayesNet.getNrOfNodes();

		nodeDimension = new byte[numberOfNodes + 1];
		int index = 1;
		for(int iNode = 0; iNode < numberOfNodes; iNode++){
			nodeDimension[index++] = (byte) m_BayesNet.getCardinality(iNode);
		}

		numberOfCases = m_BayesNet.m_Instances.numInstances();

		cases = new byte[numberOfCases + 1][numberOfNodes + 1];
		for (int row = 1; row <= numberOfCases; row++) {
			for(int col = 1; col <= numberOfNodes; col++){
				cases[row][col] = toByte(row-1, col-1);
				//System.out.print(cases[row][col]+" ");
			}
			//System.out.println();
		}

	}


	protected void readNodeInfo(int numOfNodes){

		nodeInfo = new NodeInfoRecord[numberOfNodes + 1];
		for (int i = 1; i <= numberOfNodes; i++) {
			String name = m_BayesNet.getNodeName(i-1);
			int numberOfValues = m_BayesNet.getCardinality(i-1);
			nodeInfo[i] = new NodeInfoRecord(name, numberOfValues);

			//String[] myValues = nodeInfo[i].value;
			//for (int j = 1; j <= numberOfValues; j++)
			//	myValues[j] = m_BayesNet.
		}

	}

	/*protected boolean redundant(int newChild) {

		boolean allSame = true;

		for (int i = 0; i < m_BayesNet.getNrOfParents(targetNode); i++) {
			int parent = m_BayesNet.getParent(targetNode, i);

			if (parent == newChild)
				break;

			if (!hasParents(m_BayesNet, parent)) {
				allSame = false;
				break;
			}
		}

		return allSame;
	}*/



	protected void removeChild(int child) {
		newChildPresent[child] = false;

		if (childJustAddedFlag) {
			numberOfChildren--;
			childPresent[child] = false;
			parents[child][0] = 0;
			newChildren[0]--;
			removeNodeScores(child);
			map[child] = 0;
		} else {
			for (int i = 1; i <= newChildren[0]; i++) {
				int existingNewChild = newChildren[i];
				removeNodeScores(existingNewChild);
				parents[existingNewChild][0]--;
				incorporateNodeScores(existingNewChild);
			}
		}
	}

	protected void removeNodeScores(int node) {
		int mapNode = map[node];

		for (int casei = firstCase; casei <= lastCase; casei++) {
			for (int targetValue = 1; targetValue <= nodeDimension[targetNode]; targetValue++) {
				double lnProb = lnChildProb[casei][targetValue][mapNode];
				lnChildrenProb[casei][targetValue] -= lnProb;
			}
		}
	}

	/**
	 * Remove the nodes that where not used after running the algorithm
	 * @throws Exception 
	 *

	protected void removeUnusedNodes() throws Exception{

		//Create and EditableBayesNet
		Instances instSmall = new Instances(inst);

		//Remove Unused Attributes
		for(int iNode = m_BayesNet.getNrOfNodes()-1; iNode >= 0; iNode--) {
			if((!hasParents(m_BayesNet, iNode) || !hasChildren(m_BayesNet, iNode)) && iNode != targetNode){
				//System.out.println("instSmall.deleteAttribute("+iNode+")");
				instSmall.deleteAttributeAt(iNode);
			}
		}

		EditableBayesNet bn = new EditableBayesNet(instSmall);

		//Add all arcs of parent nodes
		for(int iNode = 0; iNode < bn.getNrOfNodes(); iNode++) {
			int nodeIndex = 0;
			//get node Index
			for(int i = 0; i < m_BayesNet.getNrOfNodes(); i++){
				if(m_BayesNet.getNodeName(i) == bn.getNodeName(iNode)){
					nodeIndex = i;
					break;
				}
			}
			if(hasParents(m_BayesNet, nodeIndex)){
				int[] parents = m_BayesNet.getParentSet(nodeIndex).getParents();
				for(int jNode = 0; jNode < m_BayesNet.getParentSet(nodeIndex).getNrOfParents(); jNode++){
					int newNodeIndex = 0;
					for(int i = 0; i < bn.getNrOfNodes(); i++){
						if(bn.getNodeName(i) == m_BayesNet.getNodeName(parents[jNode])){
							newNodeIndex = i;
							break;
						}
					}
					bn.addArc(newNodeIndex, iNode); // addArc(parent, child)
				}
			}
		}

		//printFull(bn);
		m_BayesNet = bn;
	}*/

	/**
	 * Assigns a so-called "prequential" Bayesian score to the model.
	 * This scores captures how well the model predicts the target node.
	 * In particular, it represents the probability of the target node data, given
	 * the model and training data. It is a conditional, marginal likelihood.
	 *
	 * @param node is the target node
	 * @return score
	 */
	protected double scoreNode(int node) {
		double lnTotalScore = 0;

		if (parents[node][0] > 0) {
			byte firstParentSize = nodeDimension[parents[node][1]];

			for (int i = 1; i <= firstParentSize; i++)
				countsTree[i] = 0;

			countsTreePtr = firstParentSize + 1;
			countsPtr = 1;
		} else {
			countsTreePtr = 1;
			countsPtr = nodeDimension[node] + 1;

			for (int i = 1; i <= nodeDimension[node]; i++)
				counts[i] = 0;
		}


		// this builds a frequency tree
		for (int casei = firstCase; casei <= lastCase; casei++)
			fileCase(node, casei);

		// zero the counts in the frequency tree, but keep the tree
		for (int i = 1; i <= countsPtr; i++)
			counts[i] = 0;

		double a;
		double b;

		//if (getScoreMetric().getSelectedTag().getIDStr().equals("0")) {  // K2 scoring measure
		if (m_ScoreMetric==0) {  // K2 scoring measure
			a = 1;
			b = (double) nodeDimension[node];
		} else {  // BDeu scoring measure
			a = priorSampleSize / numberOfJointStates(node);
			b = a * (double) nodeDimension[node];
		}

		//System.out.println("a: "+a+"\tb: "+b);

		// derive the conditional prequential score
		for (int casei = firstCase; casei <= lastCase; casei++) {
			deriveNodeProbs(casei, node, a, b);
			byte nodeValue_casei = cases[casei][node];
			lnTotalProb = MINUS_INFINITY;
			//System.out.println("node: "+node);
			//System.out.println("lnTotalProb = "+lnTotalProb+"\tnodeValue_casei = "+nodeValue_casei+"\tnodeProb["+node+"] = "+nodeProb[node-1]);

			double lnNumer = 0;
			for (int j = 1; j <= nodeDimension[node]; j++) {
				//System.out.println("nodeProb["+j+"] = "+ nodeProb[j] + "  --> lnChildrenProb["+casei+"]["+j+"] = "+lnChildrenProb[casei][j]);
				double lnMarginal = Math.log(nodeProb[j]) + lnChildrenProb[casei][j];
				lnTotalProb = lnXpluslnY(lnTotalProb, lnMarginal);

				if (j == nodeValue_casei)
					lnNumer = lnMarginal;
			}

			double lnNodeProbAll = lnNumer - lnTotalProb;

			// lnNodeProbAll is the prob of the node value of case_i give all the predictors,
			// both parent predictors and children predictors.
			lnTotalScore += lnNodeProbAll;

			fileCase(node, casei);
		}

		return lnTotalScore;
	}

	/**
	 * Derives overall model score, using in the model "node" and its parents.
	 * It differs from ScoreNode in its use in scoring only models with children of the target,
	 * whereas ScoreNode scores parents of the target.
	 *
	 * @param node
	 * @return
	 */
	protected double scoreNode2(int node) {
		removeNodeScores(node);
		incorporateNodeScores(node);

		return tallyModelScore();
	}

	/**
	 * search determines the network structure/graph of the network
	 * based on the EBMC search algorithm
	 * 
	 * @param bayesNet the network
	 * @param instances the data to work with
	 * @throws Exception if something goes wrong
	 */
	public void buildStructure(BayesNet bayesNet, Instances instances) throws Exception {

		//m_BayesNet = new BayesNet();
		//m_BayesNet.m_Instances = instances;
		//m_BayesNet.initStructure();
		//copyParentSets(m_BayesNet, bayesNet);

		//System.out.println("Expected parents of target: "+m_ExpectedParentsOfTarget);
		//System.out.println("Max nr of parents: "+m_MaxNrOfParents);
		//System.out.println("Max nr of children: "+m_MaxNrOfChildren);

		m_BayesNet = bayesNet;

		//inst = instances;
		targetNode = instances.classIndex();
		readCases();
		initVariables();


		//System.out.println("ScoreMetric: "+m_ScoreMetric);

		/*System.out.println("BEFORE:");
		System.out.println(print(m_BayesNet));
		m_BayesNet.initStructure();
		System.out.println("\nAFTER:");
		System.out.println(print(m_BayesNet));

		//Initialize variables
		for(int iNode = 0; iNode < m_BayesNet.getNrOfNodes(); iNode++) {
			deleteAllParents(iNode);
		}*/

		//EBMC Algorithm
		boolean hopeRemains = true;
		while (hopeRemains) {
			hopeRemains = searchChildren(targetNode);
		}

		// Removing one or more of the arcs would improve the prediction of T
		weedoutWeakArcs();


		// Updating the weka model and printingT

		//if(m_BayesNet.getDebug() == true){
			System.out.println("\n\n==========================");
			System.out.println("MODEL: ");
		//}
		updateBayesNet();
		//copyParentSets(bayesNet, m_BayesNet);

		//if(m_BayesNet.getDebug() == true){
			System.out.println(print(bayesNet));
		//}
		//my_model += print(bayesNet);

	} // buildStructure 

	public void updateBayesNet(){
		for(int i = 1; i < parents.length-1;i++){
			//System.out.print("A"+i+" ("+parents[i][0]+") : ");
			if(parents[i][0] > 0){
				for(int j = 1; j<=parents[i][0]; j++){
					//System.out.print("A"+parents[i][j]+" ");
					m_BayesNet.getParentSet(i-1).addParent(parents[i][j]-1, m_BayesNet.m_Instances);
				}
			}
			//System.out.println();
		}
	}

	/** CopyParentSets copies parent sets of source to dest BayesNet
	 * @param dest destination network
	 * @param source source network

	void copyParentSets(BayesNet dest, BayesNet source){
		int nNodes = source.getNrOfNodes();
		// clear parent set first
		for (int iNode = 0; iNode < nNodes; iNode++) {
			dest.getParentSet(iNode).copy(source.getParentSet(iNode));
		}		
	}*/ // CopyParentSets

	/**
	 * buildStructure determines the network structure/graph of the network
	 * with the EBMC algorithm
	 * 
	 * @param bayesNet the network
	 * @param instances the data to use
	 * @throws Exception if something goes wrong

	public void buildStructure (BayesNet bayesNet, Instances instances) throws Exception {
		m_BayesNet = bayesNet;
		search(bayesNet, instances);
	} // buildStructure
	 */



	/**
	 * Searches for an additional, new set of children that improve the prediction of the target node
	 * in light of the existing children that have already been "locked in".
	 *
	 * @param targetNode
	 * @return
	 */
	protected boolean searchChildren(int targetNode) {
		boolean allTrue, improvement;
		double bestScore, score;
		int bestChild;

		parents[targetNode][0] = 0;
		//System.out.println("prior: "+prior());
		nodeScore = scoreNode(targetNode) + prior();  // Score the targetnode with current children (if any)
		//if(m_BayesNet.getDebug() == true){
			System.out.println("score: "+nodeScore);
		//}
		numberOfModelsScored++;

		newChildren[0] = 0;  // newChildren is a list of new children being added to the model.

		// It can be considered as a kind of generalized rule that predicts the target.
		improvement = false;
		//if(m_BayesNet.getDebug() == true){
			System.out.println("starting search for an additional predictor");
		//}
		while (newChildren[0] < MAXNEWCHILDREN) {
			bestScore = 1;
			bestChild = 0;
			allTrue = true;

			for (int child = 1; child <= numberOfNodes; child++) {
				if (child != targetNode && (!childPresent[child] || (childPresent[child] && !newChildPresent[child] && newChildren[0] > 0))) {
					allTrue = false;
					//System.out.println("numberOfChildren: "+numberOfChildren);

					if(numberOfChildren < maxChildren){ //added by Arturo!!
						addChild(child);  // adds child as a child in the newChildren "rule" being constructed and tested
					}
					score = tallyModelScore();  // scores the new "rule" consisting of the newChildren
					numberOfModelsScored++;

					if (score > bestScore || bestScore == 1) {
						bestChild = child;
						bestScore = score;
					}

					removeChild(child);  // removes child as a child of the newChildren "rule"
				}
			}

			// break out of While, because no children can be added to the rule
			if (allTrue)
				break;

			if (bestScore > nodeScore) {
				nodeScore = bestScore;
				addChild(bestChild);
				improvement = true;

				//if(m_BayesNet.getDebug() == true){
					System.out.print("   current new predictors: ");

					for (int j = 1; j <= newChildren[0]; j++)
						System.out.print(nodeInfo[newChildren[j]].name + " ");
					System.out.println("   score:"+nodeScore);
				//}

			} else
				break;
		}  // end of while

		if (improvement && (numberOfChildren < maxChildren))
			return true;
		else
			return false;
	} //searchChildren

	/**
	 * Performs a greedy, forward stepping search for the highest scoring model,
	 * according to the score that is returned by ScoreNode.
	 *
	 * @param node is the target node

	protected void searchParents(int node) {
		boolean allTrue;
		double bestScore, score, nodeScore;
		int bestParent, parent;

		System.out.println("Score with no parents:");
		nodeScore = scoreNode(node) + prior(node);  // Score the node with no parents.
		System.out.println("score: "+nodeScore+"\n----");

		while (getNrOfParents(node) < getMaxNrOfParents()) {
			bestScore = 1;
			bestParent = 0;
			allTrue = true;

			// The last conjunct above insures that the first parent of the target will not already be a child of the target.
			// This condition guarantees that the predictors (parents) of the target will be a unique set.
			// Otherwise, the search could get stuck in an infinite loop.
			for(int i = 0; i < inst.numAttributes(); i++) {
				if ((i != node) && !isParentXofY(i, node) && !isChildrenXofY(i, node)) {
					allTrue = false;
					parent = i;
					score = calcScoreWithExtraParent(node, parent);

					if ((score > bestScore) || (bestScore == 1)) {
						bestParent = parent;
						bestScore = score;
					}
				}
			}

			if (allTrue)
				break;

			if (bestScore > nodeScore) {
				nodeScore = bestScore;
				addParent(node, bestParent);
			} else
				break;
		}
	}
	 */

	/**
	 * Sets the Expected Parents of Target
	 *
	 * @param exp the expected parents of target
	 */
	public void setExpectedParentsOfTarget(int exp) {
		this.m_ExpectedParentsOfTarget = exp;
	}


	/**
	 * Sets the Maximum number of children
	 *
	 * @param setMaxChildren the children number of parents that any node is allowed to have
	 */
	public void setMaxNrOfChildren(int maxChildren) {
		this.m_MaxNrOfChildren = maxChildren;
	}


	/**
	 * Sets the Maximum number of parents
	 *
	 * @param setMaxParents the maximum number of parents that any node is allowed to have
	 */
	public void setMaxNrOfParents(int maxParents) {
		this.m_MaxNrOfParents = maxParents;
	}

	/**
	 * Parses a given list of options. <p/>
	 *
	 <!-- options-start -->
	 * Valid options are: <p/>
	 * 
	 * <pre> -T &lt;expected parents of target&gt;
	 *  Maximum number of parents</pre>
	 * 
	 * * <pre> -P &lt;nr of parents&gt;
	 *  Maximum number of parents</pre>
	 *  
	 * <pre> -C &lt;nr of children&gt;
	 *  Maximum number of parents</pre>
	 * 
	 * <pre> -mbc
	 *  Applies a Markov Blanket correction to the network structure, 
	 *  after a network structure is learned. This ensures that all 
	 *  nodes in the network are part of the Markov blanket of the 
	 *  classifier node.</pre>
	 * 
	 * <pre> -S [K2|BDeu]
	 *  Score type (BAYES, BDeu)</pre>
	 *  
	 *  * <pre> -E &lt;prior equivalent sample size;
	 *  Pess</pre>
	 * 
	 <!-- options-end -->
	 *
	 * @param options the list of options as an array of strings
	 * @throws Exception if an option is not supported
	 */
	public void setOptions(String[] options) throws Exception {

		//m_bInitAsNaiveBayes = !(Utils.getFlag('N', options));

		setExpectedParentsOfTarget(Integer.parseInt(Utils.getOption('T', options)));

		setMaxNrOfParents(Integer.parseInt(Utils.getOption('P', options)));

		setMaxNrOfChildren(Integer.parseInt(Utils.getOption('C', options)));

		String sScore = Utils.getOption('S', options);

		if (sScore.compareTo("K2") == 0) {
			setScoreMetric(new SelectedTag(0, SCORING_METRICS));
		}
		if (sScore.compareTo("BDeu") == 0) {
			setScoreMetric(new SelectedTag(1, SCORING_METRICS));
			setPriorEquivalentSampleSize(Integer.parseInt(Utils.getOption('E', options)));
		}



		//setOptions(options);
	}

	public void setPriorSampleSize(int priorSampleSize) {
		this.priorSampleSize = priorSampleSize;
	}

	/**
	 * set quality measure to be used in searching for networks.
	 * 
	 * @param newScoreMetric the new score type
	 */
	public void setScoreMetric(SelectedTag newScoreMetric) {
		if (newScoreMetric.getTags() == SCORING_METRICS) {
			m_ScoreMetric = newScoreMetric.getSelectedTag().getID();
		}
	}


	protected double tallyModelScore() {
		parents[targetNode][0] = 0;

		return (scoreNode(targetNode) + prior());
	}

	protected byte toByte(int instance, int attribute){
		byte bit = 0;

		String value = m_BayesNet.m_Instances.instance(instance).stringValue(attribute);
		bit = (byte) (m_BayesNet.m_Instances.attribute(attribute).indexOfValue(value) + 1);

		return bit;

	}

	/**
	 * Removes arcs among children (of the target) that contribute negatively to the overall model score.
	 * Greedily removes arcs of each child, one child at a time. Thus, it is quite myopic, yet fairly efficient.
	 */
	protected void weedoutWeakArcs() {
		parents[targetNode][0] = 0;  // At this point, the model consists only of the target, children of the target, and selected arcs among the children.

		for (int i = 1; i <= numberOfChildren; i++)
			pruneParents(children[i]);
	}

	protected static String print(BayesNet bn){

		String model = "";
		//Initialize BayesNet to single variable
		for(int iNode = 0; iNode < bn.getNrOfNodes(); iNode++) {
			if(hasParents(bn, iNode) || hasChildren(bn, iNode) || iNode == (bn.getNrOfNodes()-1)){
				model += bn.getNodeName(iNode) + " ("+bn.getCardinality(iNode)+")";
				if(bn.getParentSet(iNode).getNrOfParents() > 0){
					model += " --> ";
				}
				for(int jNode = 0; jNode < bn.getParentSet(iNode).getNrOfParents(); jNode++){
					int pNode = bn.getParentSet(iNode).getParent(jNode);

					model += bn.getNodeName(pNode);
					if((jNode+1) < bn.getParentSet(iNode).getNrOfParents()){
						model += ", ";
					}
				}

				model += "\n";
			}
			else{
				model += bn.getNodeName(iNode) + " ("+bn.getCardinality(iNode)+")\n";
			}
		}
		return model;
		//System.out.println("Score: " + scoreNode(targetNode));
		//System.out.println("----");
	}

	/**
	 * Check if a given node has children
	 *
	 * @param node
	 * @return
	 */
	protected static boolean hasChildren(BayesNet bn, int node){
		boolean children = false;
		int i=0;
		while( children == false & (i < bn.getNrOfNodes()) ){
			if (bn.getParentSet(i).contains(node)){
				children = true;
			}
			i++;
		}
		return children;
	}

	/**
	 * Check if a given node has parents
	 *
	 * @param node
	 * @param bn the model
	 * @return boolean state if the node has parents 
	 */
	protected static boolean hasParents(BayesNet bn, int node){
		boolean parents = false;
		if (bn.getParentSet(node).getNrOfParents() > 0){
			parents = true;
		}
		return parents;
	}

	public int getPriorEquivalentSampleSize() {
		return m_PriorEquivalentSampleSize;
	}

	public void setPriorEquivalentSampleSize(int m_PriorEquivalentSampleSize) {
		this.m_PriorEquivalentSampleSize = m_PriorEquivalentSampleSize;
	}

}
