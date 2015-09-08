package weka.classifiers.bayes.net.search.local.data;

import java.io.Serializable;

/**
 *
 * Feb 17, 2011
 * 1:46:31 PM
 * @author Kevin V. Bui (kvb2@pitt.edu)
 */
public class FileCaseRecord implements Serializable{
	
	static final long serialVersionUID = 6176545934752116631L;

    public int countsTreePtr;
    public int countsPtr;
    public int[] countsTree;
    public int[] counts;


    /**
     * Class constructor.
     *
     * @param size the size of array to hold records
     */
    public FileCaseRecord(int size) {
        countsTreePtr = 0;
        countsPtr = 0;
        countsTree = new int[size];
        counts = new int[size];

        for (int i = 0; i < countsTree.length; i++)
            countsTree[i] = 0;
        for (int i = 0; i < counts.length; i++)
            counts[i] = 0;
    }

}
