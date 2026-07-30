package minicpbp.examples;

/**
 * CrossingTraffic decomposition for the costRegular-based CP model.
 *
 * Grid is W columns by H rows, 0-indexed:
 *   x ∈ [0, W-1]   (x = W-1 is the input border where cars appear)
 *   y ∈ [0, H-1]   (y = 0 and y = H-1 are safe rows: never hold cars)
 * Convention: y grows upward (y = 0 is the bottom row), aligned with the
 * existing TrafficCrossingLineDecomposition and IPPC CrossingTraffic.
 * Standard IPPC layout: init = (0, 0) bottom-left, goal = (W-1, H-1) top-right.
 *
 * Cell linearization: idx(x, y) = x + W * y, ∈ [0, W*H).
 * Robot state space: positions [0, W*H) ∪ {DEAD = W*H}. So nbStates = W*H + 1.
 *
 * Action encoding (aggregated for costRegular alphabet):
 *   m ∈ {0..4} (NORTH:y+1, SOUTH:y-1, EAST:x+1, WEST:x-1, NOOP)
 *   c ∈ {0,1}  = collision under the robot at this step
 *   a = m + 5*c, a ∈ {0..9}
 * The m → index mapping below matches the RDDL action ordering used by
 * the Python RL client (action_north=0, action_south=1, action_east=2,
 * action_west=3, noop=4).
 *
 * costRegular costs (2D, c[state][value]):
 *   c[goal][·] = 0  (absorbing, goal reached)
 *   c[s][·]    = -1 everywhere else, INCLUDING DEAD
 * Dying at step k therefore keeps paying -1 for every remaining step until T,
 * matching the IPPC CrossingTraffic reward (-1 per step while robot-at-goal
 * is false), so dying early is strictly worse than dying late.
 * Sum over k=0..T-1 of c[R_k][a_k] = totalCost, which equals the expected
 * total reward returned by QUERY_ETR.
 *
 * Stochasticity is fully exogenous: it lives only in the oracle marginals
 * posted on the input-border obstacles. The transition function below is
 * deterministic in (state, action).
 */
public class TrafficCrossingDecomposition {

    // -------------------------------------------------------------------------
    // Constants
    // -------------------------------------------------------------------------

    // Matches TrafficCrossingLineDecomposition / RDDL action ordering.
    public static final int NORTH = 0;
    public static final int SOUTH = 1;
    public static final int EAST  = 2;
    public static final int WEST  = 3;
    public static final int NOOP  = 4;

    public static final int NB_MOVES   = 5;
    public static final int NB_ACTIONS = 10;   // (m, c) → m + 5*c
    public static final int STAR       = -1;   // joker value for ShortTableCT

    // -------------------------------------------------------------------------
    // Indexing helpers — every grid <-> index conversion goes through these
    // -------------------------------------------------------------------------

    public static int idx(int x, int y, int W) {
        return x + W * y;
    }

    public static int xOf(int i, int W) { return i % W; }
    public static int yOf(int i, int W) { return i / W; }

    public static int dead(int W, int H) { return W * H; }
    public static int nbStates(int W, int H) { return W * H + 1; }

    // -------------------------------------------------------------------------
    // Result container
    // -------------------------------------------------------------------------

    public static class CrossingTrafficModel {
        public final int W, H;
        public final int goalX, goalY;
        public final int initState;     // idx(initX, initY)
        public final int goalState;     // idx(goalX, goalY)
        public final int deadState;     // W*H
        public final int nbStates;      // W*H + 1
        public final int[][] A;         // A[state][action 0..9] → next state
        public final int[][] cost;      // cost[state][action 0..9]

        CrossingTrafficModel(int W, int H, int goalX, int goalY,
                             int initState, int goalState, int deadState,
                             int nbStates, int[][] A, int[][] cost) {
            this.W = W; this.H = H;
            this.goalX = goalX; this.goalY = goalY;
            this.initState = initState;
            this.goalState = goalState;
            this.deadState = deadState;
            this.nbStates  = nbStates;
            this.A         = A;
            this.cost      = cost;
        }
    }

    // -------------------------------------------------------------------------
    // Model build: automaton transitions + per-state cost matrix
    // -------------------------------------------------------------------------

    public static CrossingTrafficModel build(int W, int H,
                                             int goalX, int goalY,
                                             int initX, int initY) {
        int nbStates  = nbStates(W, H);
        int deadState = dead(W, H);
        int goalState = idx(goalX, goalY, W);
        int initState = idx(initX, initY, W);

        int[][] A    = new int[nbStates][NB_ACTIONS];
        int[][] cost = new int[nbStates][NB_ACTIONS];

        // DEAD: absorbing, -1 cost per step (matches IPPC reward semantics:
        // dying does not stop the per-step penalty until the horizon).
        for (int a = 0; a < NB_ACTIONS; a++) {
            A[deadState][a]    = deadState;
            cost[deadState][a] = -1;
        }

        // Grid states
        for (int s = 0; s < W * H; s++) {
            int x = xOf(s, W);
            int y = yOf(s, W);
            boolean atGoal = (s == goalState);

            for (int a = 0; a < NB_ACTIONS; a++) {
                int m = a % 5;        // 0..4
                int c = a / 5;        // 0..1

                int next;
                if (atGoal) {
                    next = goalState;            // absorbing once reached
                } else if (c == 1) {
                    next = deadState;            // collision dominates the move
                } else {
                    next = moveDest(s, m, W, H); // deterministic move; illegal → stay
                }
                A[s][a] = next;
                // Cost depends on the CURRENT state: 0 only on goal, -1 elsewhere.
                cost[s][a] = (atGoal ? 0 : -1);
            }
        }

        return new CrossingTrafficModel(W, H, goalX, goalY,
                                        initState, goalState, deadState,
                                        nbStates, A, cost);
    }

    // Illegal moves (off-grid) leave the robot in place. NOOP is also a stay.
    private static int moveDest(int s, int m, int W, int H) {
        int x = xOf(s, W);
        int y = yOf(s, W);
        int nx = x, ny = y;
        switch (m) {
            case NORTH: ny = y + 1; break;
            case SOUTH: ny = y - 1; break;
            case EAST:  nx = x + 1; break;
            case WEST:  nx = x - 1; break;
            default:    break;        // NOOP
        }
        if (nx < 0 || nx >= W || ny < 0 || ny >= H) return s;
        return idx(nx, ny, W);
    }

    // -------------------------------------------------------------------------
    // Action encoding table: tuples (m, c, a) with a = m + 5*c
    // -------------------------------------------------------------------------

    public static int[][] buildActionEncodingTable() {
        int[][] t = new int[NB_ACTIONS][3];
        int row = 0;
        for (int c = 0; c <= 1; c++) {
            for (int m = 0; m < NB_MOVES; m++) {
                t[row][0] = m;
                t[row][1] = c;
                t[row][2] = m + 5 * c;
                row++;
            }
        }
        return t;
    }

    // -------------------------------------------------------------------------
    // Collision short table: scope [R, O_0, ..., O_{WH-1}, c]  (W*H + 2 cols)
    //
    //   R = i, O_i = b, others = *  →  c = b      (2 tuples per cell)
    //   R = DEAD, others = *        →  c = 0      (1 tuple)
    // Total: 2*W*H + 1 tuples.  Joker = STAR (= -1).
    // -------------------------------------------------------------------------

    public static int[][] buildCollisionTableTuples(int W, int H) {
        int cells   = W * H;
        int arity   = cells + 2;          // [R, O_0..O_{cells-1}, c]
        int nTuples = 2 * cells + 1;     // Nombre de lignes
        int[][] t   = new int[nTuples][arity];

        // Initialize all entries to STAR
        for (int r = 0; r < nTuples; r++) {
            for (int col = 0; col < arity; col++) t[r][col] = STAR;
        }

        int row = 0;
        for (int i = 0; i < cells; i++) {
            for (int b = 0; b <= 1; b++) {
                t[row][0]         = i;             // R = i
                t[row][1 + i]     = b;             // O_i = b
                t[row][arity - 1] = b;             // c = b
                row++;
            }
        }
        // DEAD tuple: R = DEAD, c = 0, all O_* = *
        t[row][0]         = dead(W, H);
        t[row][arity - 1] = 0;

        return t;
    }

    // -------------------------------------------------------------------------
    // Validation main()
    // -------------------------------------------------------------------------

    public static void main(String[] args) {
        System.out.println("=== TrafficCrossingDecomposition validation ===");

        int W = 3, H = 3;
        int gX = 2, gY = 2;            // top-right corner (IPPC layout, y up)
        int iX = 0, iY = 0;            // bottom-left corner
        CrossingTrafficModel m = build(W, H, gX, gY, iX, iY);

        int DEAD = m.deadState;
        int GOAL = m.goalState;
        int INIT = m.initState;

        boolean ok = true;

        // DEAD absorbing; -1 cost (matches IPPC: dying keeps paying -1 per step).
        for (int a = 0; a < NB_ACTIONS; a++) {
            if (m.A[DEAD][a] != DEAD) {
                System.out.println("  FAIL DEAD not absorbing on a=" + a);
                ok = false;
            }
            if (m.cost[DEAD][a] != -1) {
                System.out.println("  FAIL cost[DEAD][a=" + a + "] = " + m.cost[DEAD][a]);
                ok = false;
            }
        }

        // GOAL absorbing, zero cost
        for (int a = 0; a < NB_ACTIONS; a++) {
            if (m.A[GOAL][a] != GOAL) {
                System.out.println("  FAIL GOAL not absorbing on a=" + a);
                ok = false;
            }
            if (m.cost[GOAL][a] != 0) {
                System.out.println("  FAIL cost[GOAL][a=" + a + "] = " + m.cost[GOAL][a]);
                ok = false;
            }
        }

        // INIT non-zero cost
        if (m.cost[INIT][NOOP] != -1) {
            System.out.println("  FAIL cost[INIT][NOOP] = " + m.cost[INIT][NOOP]);
            ok = false;
        }

        // Any action with c=1 from a live, non-goal state goes to DEAD
        if (m.A[INIT][NOOP + 5] != DEAD) {
            System.out.println("  FAIL collision did not send INIT to DEAD");
            ok = false;
        }

        // Move dynamics from INIT (0,0): NORTH(y+1) → (0,1), EAST → (1,0),
        // WEST illegal → stay, SOUTH illegal → stay.
        if (m.A[INIT][NORTH] != idx(0, 1, W)) { System.out.println("  FAIL INIT NORTH"); ok = false; }
        if (m.A[INIT][EAST]  != idx(1, 0, W)) { System.out.println("  FAIL INIT EAST");  ok = false; }
        if (m.A[INIT][WEST]  != INIT)         { System.out.println("  FAIL INIT WEST stay"); ok = false; }
        if (m.A[INIT][SOUTH] != INIT)         { System.out.println("  FAIL INIT SOUTH stay"); ok = false; }

        // Action encoding table
        int[][] enc = buildActionEncodingTable();
        if (enc.length != NB_ACTIONS) { System.out.println("  FAIL enc length"); ok = false; }
        for (int[] row : enc) {
            if (row[2] != row[0] + 5 * row[1]) { System.out.println("  FAIL enc row"); ok = false; }
        }

        // Collision short table
        int[][] coll = buildCollisionTableTuples(W, H);
        int expected = 2 * W * H + 1;
        if (coll.length != expected) {
            System.out.println("  FAIL coll length: " + coll.length + " vs " + expected);
            ok = false;
        }
        int arity = W * H + 2;
        // sample: tuple for R=INIT, O_INIT=1 → c=1
        boolean found = false;
        for (int[] row : coll) {
            if (row[0] == INIT && row[1 + INIT] == 1 && row[arity - 1] == 1) {
                found = true;
                // check the rest is STAR
                for (int col = 1; col < arity - 1; col++) {
                    if (col == 1 + INIT) continue;
                    if (row[col] != STAR) { System.out.println("  FAIL non-star in coll row"); ok = false; }
                }
                break;
            }
        }
        if (!found) { System.out.println("  FAIL missing coll tuple R=INIT, O_INIT=1"); ok = false; }

        System.out.println(ok ? "PASS" : "FAIL");
        System.out.println("  W=" + W + " H=" + H + " nbStates=" + m.nbStates
            + " INIT=" + INIT + " GOAL=" + GOAL + " DEAD=" + DEAD);
        System.out.println("  action table rows: " + enc.length
            + ", collision tuples: " + coll.length + " (arity " + arity + ")");
    }
}
