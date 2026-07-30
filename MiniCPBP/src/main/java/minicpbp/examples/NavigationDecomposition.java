package minicpbp.examples;

/**
 * Transition and reward matrices for the Navigation MDP.
 *
 * State encoding: state_idx = xi * nY + yj  (column-major, 0-based)
 *   xi ∈ {0..nX-1}  west-to-east
 *   yj ∈ {0..nY-1}  south-to-north
 *   dead_state = nX * nY
 *   goal_state = goal_pos_idx  (the physical goal cell — absorbing, reward 0)
 *   nbStates   = nX * nY + 1
 *
 * Actions: 0=north (yj+1), 1=south (yj-1), 2=east (xi+1), 3=west (xi-1), 4=noop
 *
 * Transitions:
 *   DEAD and GOAL are absorbing (self-loop with prob 1).
 *   From any other (xi, yj) with action a:
 *     - Compute destination (dxi, dyj); if out of bounds → stay at (xi, yj).
 *     - If destination is the goal cell → go to goal_pos_idx with prob 1.
 *     - Otherwise → destination with prob (1 - P[dxi][dyj]),
 *                   dead_state  with prob P[dxi][dyj].
 *
 * Rewards:
 *   R[s][a][s'] = 0 if s' == goal_pos_idx, else -1.
 */
public class NavigationDecomposition {

    public static final int NB_ACTIONS = 5;

    // -------------------------------------------------------------------------
    // Result container
    // -------------------------------------------------------------------------

    public static class NavigationMatrices {
        /** P[s][a][s'] — transition probability */
        public final double[][][] P;
        /** R[s][a][s'] — integer reward */
        public final int[][][] R;
        public final int nbStates;
        public final int deadStateIdx;
        public final int goalStateIdx;  // == goal_pos_idx

        public NavigationMatrices(double[][][] P, int[][][] R,
                                  int nbStates, int deadStateIdx, int goalStateIdx) {
            this.P            = P;
            this.R            = R;
            this.nbStates     = nbStates;
            this.deadStateIdx = deadStateIdx;
            this.goalStateIdx = goalStateIdx;
        }
    }

    // -------------------------------------------------------------------------
    // Public API
    // -------------------------------------------------------------------------

    /**
     * Builds P and R matrices for a Navigation instance.
     *
     * @param nX         number of x positions (west-to-east)
     * @param nY         number of y positions (south-to-north)
     * @param goalPosIdx grid index of the goal cell (xi*nY + yj) — used directly as absorbing goal state
     * @param pGrid      pGrid[xi][yj] = disappearance probability at cell (xi,yj)
     */
    public static NavigationMatrices buildMatrices(int nX, int nY,
                                                   int goalPosIdx,
                                                   double[][] pGrid) {
        int deadStateIdx = nX * nY;
        int nbStates     = nX * nY + 1;

        double[][][] P = buildTransitionMatrix(nX, nY, nbStates, goalPosIdx, deadStateIdx, pGrid);
        int[][][] R    = buildRewardMatrix(nbStates, goalPosIdx);

        return new NavigationMatrices(P, R, nbStates, deadStateIdx, goalPosIdx);
    }

    // -------------------------------------------------------------------------
    // Private helpers
    // -------------------------------------------------------------------------

    private static double[][][] buildTransitionMatrix(int nX, int nY, int nbStates,
                                                      int goalPosIdx, int deadStateIdx,
                                                      double[][] pGrid) {
        double[][][] P = new double[nbStates][NB_ACTIONS][nbStates];

        // Absorbing: DEAD and GOAL self-loop
        for (int a = 0; a < NB_ACTIONS; a++) {
            P[deadStateIdx][a][deadStateIdx] = 1.0;
            P[goalPosIdx][a][goalPosIdx]     = 1.0;
        }

        // Normal grid cells (skip goal and dead)
        for (int xi = 0; xi < nX; xi++) {
            for (int yj = 0; yj < nY; yj++) {
                int src = xi * nY + yj;
                if (src == goalPosIdx) continue;

                for (int a = 0; a < NB_ACTIONS; a++) {
                    int dxi = xi, dyj = yj;
                    switch (a) {
                        case 0: dyj++; break;  // north
                        case 1: dyj--; break;  // south
                        case 2: dxi++; break;  // east
                        case 3: dxi--; break;  // west
                        case 4: break;          // noop
                    }
                    // Out of bounds → stay
                    if (dxi < 0 || dxi >= nX || dyj < 0 || dyj >= nY) {
                        dxi = xi; dyj = yj;
                    }

                    int dest = dxi * nY + dyj;
                    if (dest == goalPosIdx) {
                        P[src][a][goalPosIdx] = 1.0;
                    } else {
                        double pDie = pGrid[dxi][dyj];
                        P[src][a][deadStateIdx] += pDie;
                        P[src][a][dest]         += 1.0 - pDie;
                    }
                }
            }
        }
        return P;
    }

    private static int[][][] buildRewardMatrix(int nbStates, int goalPosIdx) {
        int[][][] R = new int[nbStates][NB_ACTIONS][nbStates];
        for (int s = 0; s < nbStates; s++) {
            for (int a = 0; a < NB_ACTIONS; a++) {
                for (int ns = 0; ns < nbStates; ns++) {
                    if (ns != goalPosIdx) R[s][a][ns] = -1;
                }
            }
        }
        return R;
    }

    // -------------------------------------------------------------------------
    // main() — validation
    // -------------------------------------------------------------------------

    public static void main(String[] args) {
        System.out.println("=== NavigationDecomposition validation (instance 1: 4x3) ===");

        int nX = 4, nY = 3;
        double[][] pGrid = new double[nX][nY];
        pGrid[0][1] = 0.048967;
        pGrid[1][1] = 0.345437;
        pGrid[2][1] = 0.636995;
        pGrid[3][1] = 0.928158;

        // goal = (x21, y20) → xi=3, yj=2 → goalPosIdx=11
        int goalPosIdx = 3 * nY + 2;

        NavigationMatrices m = buildMatrices(nX, nY, goalPosIdx, pGrid);

        System.out.println("nbStates=" + m.nbStates
            + "  dead=" + m.deadStateIdx
            + "  goal=" + m.goalStateIdx);

        // Verify all rows sum to 1
        boolean ok = true;
        for (int s = 0; s < m.nbStates; s++) {
            for (int a = 0; a < NB_ACTIONS; a++) {
                double sum = 0;
                for (int ns = 0; ns < m.nbStates; ns++) sum += m.P[s][a][ns];
                if (Math.abs(sum - 1.0) > 1e-9) {
                    System.out.println("  FAIL s=" + s + " a=" + a + " sum=" + sum);
                    ok = false;
                }
            }
        }
        System.out.println("Probability sums: " + (ok ? "PASS" : "FAIL"));

        // init=(x21,y12)=9, north → dest=(x21,y15)=10, pDie=0.928158
        System.out.printf("P[9][north=0][10]=%.6f  (expect ~0.071842)%n", m.P[9][0][10]);
        System.out.printf("P[9][north=0][dead=%d]=%.6f  (expect ~0.928158)%n",
                          m.deadStateIdx, m.P[9][0][m.deadStateIdx]);
        // (x21,y15)=10, north → goal=11
        System.out.printf("P[10][north=0][goal=%d]=%.6f  (expect 1.0)%n",
                          m.goalStateIdx, m.P[10][0][m.goalStateIdx]);
        // goal is absorbing
        System.out.printf("P[goal=%d][noop=4][goal=%d]=%.6f  (expect 1.0)%n",
                          m.goalStateIdx, m.goalStateIdx, m.P[m.goalStateIdx][4][m.goalStateIdx]);
    }
}
