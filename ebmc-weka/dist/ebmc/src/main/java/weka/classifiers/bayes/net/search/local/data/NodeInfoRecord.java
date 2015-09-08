package weka.classifiers.bayes.net.search.local.data;

import java.io.Serializable;

/**
 *
 * Feb 17, 2011
 * 1:53:17 PM
 * @author Kevin V. Bui (kvb2@pitt.edu)
 */
public class NodeInfoRecord implements Serializable{

    public String name;
    public int numberOfValues;
    public String[] value;
    static final long serialVersionUID = 6176545934752116631L;


    /**
     * Class constructor.
     *
     * @param name the name of the node(variable)
     * @param numberOfValues the number of attribute the node has
     */
    public NodeInfoRecord(String name, int numberOfValues) {
        this.name = name;
        this.numberOfValues = numberOfValues;
        this.value = new String[numberOfValues];
    }
    
}
