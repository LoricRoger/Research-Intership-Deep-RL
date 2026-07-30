package minicpbp.examples;

/**
 * Per-cell transition and reward matrices for the GameOfLife MDP.
 *
 * State encoding per cell: 0 = dead, 1 = alive.
 *
 * Action encoding per cell i:
 *   a = n * 2 + set_i
 *   where n ∈ {0..deg(i)} is the number of alive neighbors
 *         set_i ∈ {0,1}   is the set action for this cell
 *   nbActions(i) = 2 * (deg(i) + 1)
 *
 * RDDL transition (game_of_life_mdp):
 *   alive'(x,y) = Bernoulli(1 - NOISE-PROB(x,y))
 *     if (alive ∧ n ∈ {2,3}) ∨ (¬alive ∧ n == 3) ∨ set(x,y)
 *               = Bernoulli(NOISE-PROB(x,y))
 *     otherwise
 *
 * Reward per step (RDDL): sum_{x,y} [alive(x,y) - set(x,y)]
 *   Penalty for set = 1 (integer exact, no scaling needed).
 *
 * Init state: variable per instance (bitmask provided in JSON config).
 */
public class GameOfLifeNodeDecomposition {

    // -------------------------------------------------------------------------
    // Result container
    // -------------------------------------------------------------------------

    public static class GameOfLifeMatrices {
        /** P[i][s][a][ns] — transition probability for cell i */
        public final double[][][][] P_per_cell;
        /** R[i][s][a][ns] — integer reward for cell i */
        public final int[][][][] R_per_cell;
        /** nbActions[i] = 2*(deg(i)+1) */
        public final int[] nbActionsPerCell;
        /** number of cells */
        public final int N;

        public GameOfLifeMatrices(
                double[][][][] P_per_cell,
                int[][][][] R_per_cell,
                int[] nbActionsPerCell,
                int N) {
            this.P_per_cell       = P_per_cell;
            this.R_per_cell       = R_per_cell;
            this.nbActionsPerCell = nbActionsPerCell;
            this.N                = N;
        }
    }

    // -------------------------------------------------------------------------
    // Public API
    // -------------------------------------------------------------------------

    /**
     * Builds all per-cell P/R matrices for a GameOfLife instance.
     *
     * @param N          number of cells
     * @param neighbors  neighbors[i] = array of neighbor indices for cell i
     * @param noiseProbs noiseProbs[i] = NOISE-PROB for cell i
     */
    public static GameOfLifeMatrices buildAllMatrices(
            int N, int[][] neighbors, double[] noiseProbs) {

        double[][][][] P = new double[N][][][];
        int[][][][] R    = new int[N][][][];
        int[] nbActions  = new int[N];

        for (int i = 0; i < N; i++) {
            int deg      = neighbors[i].length;
            nbActions[i] = 2 * (deg + 1);
            P[i]         = buildTransition(deg, noiseProbs[i]);
            R[i]         = buildReward(deg);
        }

        return new GameOfLifeMatrices(P, R, nbActions, N);
    }

    /**
     * Returns the table tuples for the action-encoding constraint of a cell
     * with the given degree.
     *
     * Each tuple is [n, set_i, a] where a = n*2 + set_i.
     */
    public static int[][] buildActionEncodingTable(int deg) {
        int nTuples    = 2 * (deg + 1);
        int[][] tuples = new int[nTuples][3];
        int row = 0;
        for (int n = 0; n <= deg; n++) {
            for (int sv = 0; sv <= 1; sv++) {
                tuples[row][0] = n;
                tuples[row][1] = sv;
                tuples[row][2] = n * 2 + sv;
                row++;
            }
        }
        return tuples;
    }

    // -------------------------------------------------------------------------
    // Private: per-cell P and R matrices
    // -------------------------------------------------------------------------

    private static double[][][] buildTransition(int deg, double noiseProb) {
        int nbAct      = 2 * (deg + 1);
        double[][][] P = new double[2][nbAct][2];

        for (int s = 0; s < 2; s++) {
            for (int n = 0; n <= deg; n++) {
                for (int sv = 0; sv <= 1; sv++) {
                    int a = n * 2 + sv;
                    boolean conwayRule = (s == 1 && (n == 2 || n == 3))
                                     || (s == 0 && n == 3);
                    double pAlive;
                    if (sv == 1 || conwayRule) {
                        pAlive = 1.0 - noiseProb;
                    } else {
                        pAlive = noiseProb;
                    }
                    P[s][a][1] = pAlive;
                    P[s][a][0] = 1.0 - pAlive;
                }
            }
        }
        return P;
    }

    // R[s][a][ns] = ns - set_i   where set_i = a % 2
    // alive reward = +1, set penalty = -1 (exact integers, no scaling).
    private static int[][][] buildReward(int deg) {
        int nbAct   = 2 * (deg + 1);
        int[][][] R = new int[2][nbAct][2];
        for (int s = 0; s < 2; s++) {
            for (int a = 0; a < nbAct; a++) {
                int sv = a % 2;
                for (int ns = 0; ns < 2; ns++)
                    R[s][a][ns] = ns - sv;
            }
        }
        return R;
    }

    // -------------------------------------------------------------------------
    // main() — validation
    // -------------------------------------------------------------------------

    public static void main(String[] args) {
        System.out.println("=== GameOfLifeNodeDecomposition validation ===");

        // Simple 3-cell line: cell 0 (deg=1), cell 1 (deg=2), cell 2 (deg=1)
        int N        = 3;
        int[][] nbrs = {{1}, {0, 2}, {1}};
        double[] nps = {0.02, 0.03, 0.025};

        GameOfLifeMatrices m = buildAllMatrices(N, nbrs, nps);

        boolean ok = true;
        for (int i = 0; i < N; i++) {
            int nbAct = m.nbActionsPerCell[i];
            for (int s = 0; s < 2; s++) {
                for (int a = 0; a < nbAct; a++) {
                    double sum = m.P_per_cell[i][s][a][0] + m.P_per_cell[i][s][a][1];
                    if (Math.abs(sum - 1.0) > 1e-9) {
                        System.out.println("  FAIL cell=" + i + " s=" + s + " a=" + a + " sum=" + sum);
                        ok = false;
                    }
                }
            }
        }
        System.out.println("Probability sums: " + (ok ? "PASS" : "FAIL"));

        System.out.println("Encoding table for deg=2:");
        for (int[] t : buildActionEncodingTable(2))
            System.out.println("  n=" + t[0] + " set=" + t[1] + " a=" + t[2]);

        System.out.println("Transition table for deg=2, noiseProb=0.03:");
        double[][][] P2 = buildTransition(2, 0.03);
        for (int s = 0; s < 2; s++)
            for (int n = 0; n <= 2; n++)
                for (int sv = 0; sv <= 1; sv++) {
                    int a = n * 2 + sv;
                    System.out.printf("  s=%d n=%d set=%d -> P(alive)=%.4f%n", s, n, sv, P2[s][a][1]);
                }
    }
}
