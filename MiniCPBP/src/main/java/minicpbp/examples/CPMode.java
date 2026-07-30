// CPMode.java
package minicpbp.examples;

import minicpbp.engine.core.Solver;
import minicpbp.engine.core.IntVar;
import java.util.Set;

public interface CPMode {
    void applyConstraints(Solver cp, IntVar[] action, IntVar totalReward, int goalReward, int holeReward);

    int getNbActions();

    void fillTransitions(double[][][] P, int nbStates, int squareSize, Set<Integer> holeSet, int goalStateIdx, double noSlipProba, double sideSlipProba);
}
