package minicpbp.examples;

/**
 * CP model construction for TriangleTireworld MDP.
 *
 * State encoding : s = 4*p + 2*tau + h
 *   p   ∈ {0..N-1}  current location index
 *   tau ∈ {0,1}     tire intact (1) or flat (0)
 *   h   ∈ {0,1}     hasspare
 *   D(S_k) = {0..4N-1}
 *
 * Action space CP : Discrete(6)  (fixed, not grounded)
 *   0 = noop
 *   1 = move_1  (successors[p][0], or noop if absent)
 *   2 = move_2  (successors[p][1], or noop if absent)
 *   3 = move_3  (successors[p][2], or noop if absent)
 *   4 = loadtire
 *   5 = changetire
 *
 * Auxiliary variable sigma_k : spare present under the vehicle at step k.
 *   sigma_k = A^k_{p_k}   linked by ShortTableCT.
 *
 * Auxiliary variable a_k : encoded action read by Markov.
 *   a_k = m_k + 6*sigma_k,  D(a_k) = {0..11}
 *   a_k in {0..5}  -> sigma_k = 0 (no spare under vehicle)
 *   a_k in {6..11} -> sigma_k = 1 (spare present)
 *   loadtire with spare (a_k=10) sets hasspare=1; without spare (a_k=4) stays.
 */
public class TriangleTireworldDecomposition {

    // -------------------------------------------------------------------------
    // Constants
    // -------------------------------------------------------------------------

    static final int N_ACTIONS     = 6;
    static final int N_ENC_ACTIONS = 12;  // after sigma encoding
    static final int STAR          = -1;  // wildcard for ShortTableCT

    // -------------------------------------------------------------------------
    // State encoding helpers
    // -------------------------------------------------------------------------

    public static int encodeState(int p, int tau, int h) {
        return 4 * p + 2 * tau + h;
    }

    public static int decodePos(int s) {
        return s / 4;
    }

    public static int decodeTire(int s) {
        return (s / 2) % 2;
    }

    public static int decodeHasSpare(int s) {
        return s % 2;
    }

    // -------------------------------------------------------------------------
    // Transition matrix  P[s][a][s']
    // s  in {0..4N-1},  a in {0..11},  s' in {0..4N-1}
    // -------------------------------------------------------------------------

    public static double[][][] buildTransition(
            int N, int[][] successors, int goalLocation, double flatProb) {

        int nbStates = 4 * N;
        double[][][] P = new double[nbStates][N_ENC_ACTIONS][nbStates];

        for (int s = 0; s < nbStates; s++) {
            int p   = decodePos(s);
            int tau = decodeTire(s);
            int h   = decodeHasSpare(s);

            // Goal is absorbing for all actions
            if (p == goalLocation) {
                for (int a = 0; a < N_ENC_ACTIONS; a++)
                    P[s][a][s] = 1.0;
                continue;
            }

            for (int a = 0; a < N_ENC_ACTIONS; a++) {
                int m     = a % N_ACTIONS;  // raw move in {0..5}
                int sigma = a / N_ACTIONS;  // sigma in {0,1}

                if (m >= 1 && m <= 3) {
                    // move_i : go to successors[p][m-1] if it exists, else noop
                    int succIdx = m - 1;
                    int dest    = (succIdx < successors[p].length)
                                  ? successors[p][succIdx]
                                  : p;

                    if (tau == 1 && dest != p) {
                        // Intact tire, actually moving: flat_prob of getting flat
                        P[s][a][encodeState(dest, 1, h)] += 1.0 - flatProb;
                        P[s][a][encodeState(dest, 0, h)] += flatProb;
                    } else {
                        // Flat tire or noop move: stay in place
                        P[s][a][s] = 1.0;
                    }

                } else if (m == 4) {
                    // loadtire : pick up spare from ground
                    // sigma_k encodes whether the spare is at p_k — no need to re-check p
                    if (sigma == 1) {
                        // spare present under vehicle → hasspare becomes 1
                        P[s][a][encodeState(p, tau, 1)] = 1.0;
                    } else {
                        // no spare at this location → no effect
                        P[s][a][s] = 1.0;
                    }

                } else if (m == 5) {
                    // changetire : replace flat with carried spare
                    if (h == 1) {
                        // have spare → tire becomes intact, hasspare becomes 0
                        P[s][a][encodeState(p, 1, 0)] = 1.0;
                    } else {
                        // no spare in vehicle → no effect
                        P[s][a][s] = 1.0;
                    }

                } else {
                    // noop (m == 0)
                    P[s][a][s] = 1.0;
                }
            }
        }
        return P;
    }

    // -------------------------------------------------------------------------
    // Reward matrix  R[s][a][s']
    // +100 on reaching goal, 0 while at goal, -1 otherwise
    // -------------------------------------------------------------------------

    public static int[][][] buildReward(int N, int goalLocation) {
        int nbStates = 4 * N;
        int[][][] R  = new int[nbStates][N_ENC_ACTIONS][nbStates];

        for (int s = 0; s < nbStates; s++) {
            int p = decodePos(s);
            for (int a = 0; a < N_ENC_ACTIONS; a++) {
                for (int ns = 0; ns < nbStates; ns++) {
                    if (p == goalLocation) {
                        R[s][a][ns] = 0;                            // already at goal
                    } else if (decodePos(ns) == goalLocation) {
                        R[s][a][ns] = 100;                          // just reached goal
                    } else {
                        R[s][a][ns] = -1;
                    }
                }
            }
        }
        return R;
    }

    // -------------------------------------------------------------------------
    // Action-encoding table  (m_k, sigma_k, a_k)  — 12 tuples
    // a_k = m_k + 6*sigma_k
    // -------------------------------------------------------------------------

    public static int[][] buildActionEncodingTable() {
        int[][] tuples = new int[N_ENC_ACTIONS][3];
        int row = 0;
        for (int sigma = 0; sigma <= 1; sigma++) {
            for (int m = 0; m < N_ACTIONS; m++) {
                tuples[row][0] = m;
                tuples[row][1] = sigma;
                tuples[row][2] = m + N_ACTIONS * sigma;
                row++;
            }
        }
        return tuples;
    }

    // -------------------------------------------------------------------------
    // ShortTableCT table for sigma_k
    //
    // Scope : (p_k, A^k_0, A^k_1, ..., A^k_{N-1}, sigma_k)
    // Width : N + 2
    //
    // For each location i in {0..N-1} :
    //   [i, *, ..., 0 (col 1+i), ..., *, 0]   A^k_i=0 → sigma=0
    //   [i, *, ..., 1 (col 1+i), ..., *, 1]   A^k_i=1 → sigma=1
    // Total : 2N rows.
    //
    // Column layout : col 0 = p_k,  cols 1..N = A^k_0..A^k_{N-1},  col N+1 = sigma_k
    // -------------------------------------------------------------------------

    public static int[][] buildSigmaTable(int N) {
        int width    = N + 2;
        int[][] tbl  = new int[2 * N][width];

        for (int i = 0; i < N; i++) {
            for (int val = 0; val <= 1; val++) {
                int row = 2 * i + val;
                // Fill entire row with STAR first
                for (int c = 0; c < width; c++) tbl[row][c] = STAR;
                tbl[row][0]       = i;    // p_k = location i
                tbl[row][1 + i]   = val;  // A^k_i = val
                tbl[row][width-1] = val;  // sigma_k = val
            }
        }
        return tbl;
    }

    // -------------------------------------------------------------------------
    // Table for spare dynamics  (p_k, m_k, A^k_l, A^{k+1}_l)
    //
    // A^{k+1}_l = 0   if p_k=l ∧ m_k=4 ∧ A^k_l=1   (loadtire consumes spare)
    // A^{k+1}_l = A^k_l  otherwise  (persistence)
    //
    // We use a regular table (not short) — compact set of tuples.
    // For each state of (p_k, m_k, A^k_l, A^{k+1}_l) enumerate valid rows:
    //   p ≠ l, any m, any A:  A^{k+1}_l = A^k_l  → 2*(N-1)*6 rows
    //   p = l, m ≠ 4, any A:  A^{k+1}_l = A^k_l  → 2*5 rows
    //   p = l, m = 4, A=0:    A^{k+1}_l = 0       → 1 row
    //   p = l, m = 4, A=1:    A^{k+1}_l = 0       → 1 row (consumed)
    // -------------------------------------------------------------------------

    public static int[][] buildSpareDynamicsTable(int N, int location) {
        // Max rows: N*6*2 (all combinations that are valid)
        // Easier: enumerate all (p,m,A) and derive A_next
        int maxRows  = N * N_ACTIONS * 2;
        int[][] tbl  = new int[maxRows][4];
        int row = 0;

        for (int p = 0; p < N; p++) {
            for (int m = 0; m < N_ACTIONS; m++) {
                for (int Ak = 0; Ak <= 1; Ak++) {
                    int Ak1;
                    if (p == location && m == 4 && Ak == 1) {
                        Ak1 = 0;  // loadtire consumes the spare
                    } else {
                        Ak1 = Ak; // spare persists
                    }
                    tbl[row][0] = p;
                    tbl[row][1] = m;
                    tbl[row][2] = Ak;
                    tbl[row][3] = Ak1;
                    row++;
                }
            }
        }
        // Trim to actual size (should be maxRows, all rows filled)
        if (row < maxRows) {
            int[][] trimmed = new int[row][4];
            System.arraycopy(tbl, 0, trimmed, 0, row);
            return trimmed;
        }
        return tbl;
    }

    // -------------------------------------------------------------------------
    // Probability sanity check (for validation / tests)
    // -------------------------------------------------------------------------

    public static boolean validateTransition(double[][][] P) {
        int nbStates = P.length;
        boolean ok   = true;
        for (int s = 0; s < nbStates; s++) {
            for (int a = 0; a < P[s].length; a++) {
                double sum = 0.0;
                for (int ns = 0; ns < P[s][a].length; ns++) sum += P[s][a][ns];
                if (Math.abs(sum - 1.0) > 1e-9) {
                    System.err.println("P[" + s + "][" + a + "] sums to " + sum);
                    ok = false;
                }
            }
        }
        return ok;
    }
}
