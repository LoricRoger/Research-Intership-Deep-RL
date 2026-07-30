package minicpbp.examples;

/**
 * Per-computer transition and reward matrices for the SysAdmin MDP.
 *
 * State encoding per computer: 0 = dead, 1 = alive.
 *
 * Action encoding per computer i (used as the Markov action variable):
 *   a = n * 2 + r
 *   where n ∈ {0..deg(i)} is the number of alive neighbors
 *         r ∈ {0,1}       is the reboot action
 *   nbActions(i) = 2 * (deg(i) + 1)
 *
 * RDDL transition (sysadmin_mdp):
 *   running'(x) = KronDelta(true)                         if reboot(x)
 *               = Bernoulli(0.45 + 0.5*(1+n)/(1+deg(x))) if running(x) ∧ ¬reboot(x)
 *               = Bernoulli(REBOOT_PROB)                   otherwise
 *
 * Reward per step (RDDL): sum_x [ running'(x) - REBOOT_PENALTY * reboot(x) ]
 *   REBOOT_PENALTY = 0.75 = 3/4
 *
 * Rewards are scaled by 4:
 * The ETR returned by the service is therefore scaled by 4; divide by 4 on the Python side.
 *
 * All instances start with every computer alive, so
 * initState = 1 for all computers
 */
public class SysAdminNodeDecomposition {

    // -------------------------------------------------------------------------
    // Result container
    // -------------------------------------------------------------------------

    public static class SysAdminMatrices {
        /** P[i][s][a][ns] — transition probability for computer i */
        public final double[][][][] P_per_comp;
        /** R[i][s][a][ns] — integer reward for computer i */
        public final int[][][][] R_per_comp;
        /** nbActions[i] = 2*(deg(i)+1) */
        public final int[] nbActionsPerComp;
        /** number of computers */
        public final int N;

        public SysAdminMatrices(
                double[][][][] P_per_comp,
                int[][][][] R_per_comp,
                int[] nbActionsPerComp,
                int N) {
            this.P_per_comp       = P_per_comp;
            this.R_per_comp       = R_per_comp;
            this.nbActionsPerComp = nbActionsPerComp;
            this.N                = N;
        }
    }

    // -------------------------------------------------------------------------
    // Public API
    // -------------------------------------------------------------------------

    /**
     * Builds all per-computer P/R matrices for a SysAdmin instance.
     *
     * @param N          number of computers
     * @param neighbors  neighbors[i] = array of neighbor indices for computer i
     * @param rebootProb RDDL REBOOT-PROB (spontaneous recovery probability)
     */
    public static SysAdminMatrices buildAllMatrices(
            int N, int[][] neighbors, double rebootProb) {

        double[][][][] P = new double[N][][][];
        int[][][][] R    = new int[N][][][];
        int[] nbActions  = new int[N];

        for (int i = 0; i < N; i++) {
            int deg      = neighbors[i].length;
            nbActions[i] = 2 * (deg + 1);
            P[i]         = buildTransition(deg, rebootProb);
            R[i]         = buildReward(deg);
        }

        return new SysAdminMatrices(P, R, nbActions, N);
    }

    /**
     * Returns the table tuples for the action-encoding constraint of a computer
     * with the given degree.
     *
     * Each tuple is [n, r, a] where a = n*2 + r, n ∈ {0..deg}, r ∈ {0,1}.
     */
    public static int[][] buildActionEncodingTable(int deg) {
        int nTuples    = 2 * (deg + 1);  // Nombre d'actions possibles
        int[][] tuples = new int[nTuples][3];
        int row = 0;
        for (int n = 0; n <= deg; n++) {  // Nombre de voisins vivants
            for (int r = 0; r <= 1; r++) { // Action de reboot ou pas
                tuples[row][0] = n;
                tuples[row][1] = r;
                tuples[row][2] = n * 2 + r; // Valeur encodée du tuple (n,r) (donc action pour la contrainte markov)
                row++;  // row va donc de 0 à 2*(deg+1)-1 : ce qui représente le nombre d'actions
            }
        }
        return tuples;
    }

    // -------------------------------------------------------------------------
    // Private: per-computer P and R matrices
    // -------------------------------------------------------------------------

    private static double[][][] buildTransition(int deg, double rebootProb) {
        int nbAct      = 2 * (deg + 1);
        double[][][] P = new double[2][nbAct][2];

        for (int s = 0; s < 2; s++) {  // Etat de l'ordi considéré
            for (int n = 0; n <= deg; n++) { // Nombre de voisins vivants
                for (int r = 0; r <= 1; r++) {     // Action de reboot ou pas
                    int a = n * 2 + r;      // Encode l'action
                    double pAlive;
                    if (r == 1) {   // On reboot
                        pAlive = 1.0;
                    } else if (s == 1) {   // Il est vivant -> proba de crash
                        pAlive = 0.45 + 0.5 * (1.0 + n) / (1.0 + deg);
                    } else {   // Il est mort -> proba de revivre
                        pAlive = rebootProb;
                    }
                    P[s][a][1] = pAlive;
                    P[s][a][0] = 1.0 - pAlive;
                }
            }
        }
        return P;
    }

    // R[s][a][ns] = 4*ns - 3*r  where r = a % 2  (scaled by 4 to keep integers exact).
    // REBOOT_PENALTY = 0.75 = 3/4; multiply whole reward by 4: running→4, penalty→3.
    private static int[][][] buildReward(int deg) {
        int nbAct   = 2 * (deg + 1);
        int[][][] R = new int[2][nbAct][2];
        for (int s = 0; s < 2; s++) {
            for (int a = 0; a < nbAct; a++) {
                int r = a % 2;   // 'est ce que c'est un reboot ?'
                for (int ns = 0; ns < 2; ns++)
                    R[s][a][ns] = 4 * ns - 3 * r;
            }
        }
        return R;
    }

    // -------------------------------------------------------------------------
    // main() — validation
    // -------------------------------------------------------------------------

    public static void main(String[] args) {
        System.out.println("=== SysAdminLineDecomposition validation ===");

        int N        = 3;
        int[][] nbrs = {{1}, {0, 2}, {1}};
        double rp    = 0.1;

        SysAdminMatrices m = buildAllMatrices(N, nbrs, rp);

        boolean ok = true;
        for (int i = 0; i < N; i++) {
            int nbAct = m.nbActionsPerComp[i];
            for (int s = 0; s < 2; s++) {
                for (int a = 0; a < nbAct; a++) {
                    double sum = m.P_per_comp[i][s][a][0] + m.P_per_comp[i][s][a][1];
                    if (Math.abs(sum - 1.0) > 1e-9) {
                        System.out.println("  FAIL comp=" + i + " s=" + s + " a=" + a + " sum=" + sum);
                        ok = false;
                    }
                }
            }
        }
        System.out.println("Probability sums: " + (ok ? "PASS" : "FAIL"));

        System.out.println("Encoding table for deg=1:");
        for (int[] t : buildActionEncodingTable(1))
            System.out.println("  n=" + t[0] + " r=" + t[1] + " a=" + t[2]);
    }
}
