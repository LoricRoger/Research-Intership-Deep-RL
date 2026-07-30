package minicpbp.examples;

import minicpbp.engine.core.IntVar;
import minicpbp.engine.core.Solver;
import minicpbp.util.exception.InconsistencyException;
import static minicpbp.cp.Factory.*;

import java.io.*;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

import org.json.JSONArray;
import org.json.JSONObject;
import org.json.JSONException;

/**
 * TCP server exposing a CP model for the TriangleTireworld MDP (ETR mode).
 *
 * Wire protocol (same as other CP services):
 *   INIT <instance_id>          -> OK INIT successful for <id>
 *   RESET                       -> OK RESET successful
 *   STEP <step_idx> <act> <s>   -> OK STEP processed
 *       act in {0..5}  (CP action index, see Decomposition)
 *       s   in {0..4N-1}  (encoded vehicle state)
 *   QUERY_ETR                   -> ETR_VALUE <float>
 *   QUIT                        -> OK Goodbye
 *
 * State encoding : s = 4*p + 2*tau + h  (see TriangleTireworldDecomposition)
 * Action encoding: 0=noop, 1-3=move_i, 4=loadtire, 5=changetire
 *
 * Config file: triangletireworld_mdp_ippc2014.json  (in working directory)
 *
 * Default port: 12351.
 */
public class TriangleTireworldService {

    private static int    PORT           = 12351;
    private static final String INSTANCES_JSON_FILE =
        "triangletireworld_mdp_ippc2014.json";
    private static int    BP_ITERATIONS  = 1;

    // -------------------------------------------------------------------------
    // Instance parameters (loaded from JSON on INIT)
    // -------------------------------------------------------------------------

    private static int      N               = -1;
    private static int[][]  successors      = null;
    private static int[]    spareLocations  = null;
    private static int      goalLocation    = -1;
    private static int      initialLocation = -1;
    private static double   flatProb        = Double.NaN;
    private static int      nbSteps         = -1;

    // -------------------------------------------------------------------------
    // CP model (rebuilt on each RESET)
    // -------------------------------------------------------------------------

    private static Solver   cp          = null;
    // S[k]         : vehicle state at step k,  D = {0..4N-1}
    private static IntVar[] S           = null;
    // M[k]         : raw CP action at step k,  D = {0..5}
    private static IntVar[] M           = null;
    // A_spare[l][k]: spare present at location l at step k, D = {0,1}
    private static IntVar[][] A_spare   = null;
    // sigma[k]     : spare under vehicle at step k = A^k_{p_k}, D = {0,1}
    private static IntVar[] sigma       = null;
    // a[k]         : encoded action = m + 6*sigma, D = {0..11}
    private static IntVar[] a           = null;

    private static IntVar   totalReward = null;
    private static int      currentStep = 0;

    // -------------------------------------------------------------------------
    // Config
    // -------------------------------------------------------------------------

    private static JSONObject allInstancesConfig = null;
    private static String     currentInstanceId  = null;

    // -------------------------------------------------------------------------
    // Entry point
    // -------------------------------------------------------------------------

    public static void main(String[] args) throws Exception {
        // Sub-command: export <instance_id> [outdir]
        if (args.length > 0 && args[0].equalsIgnoreCase("export")) {
            runExport(args);
            return;
        }

        if (args.length > 0) {
            try { PORT = Integer.parseInt(args[0]); }
            catch (NumberFormatException e) {
                System.err.println("WARN: Invalid port '" + args[0]
                    + "'. Using default " + PORT);
            }
        }
        String bpProp = System.getProperty("bp.iterations");
        if (bpProp != null) {
            try { BP_ITERATIONS = Integer.parseInt(bpProp); }
            catch (NumberFormatException e) {
                System.err.println("WARN: Invalid bp.iterations '" + bpProp
                    + "'. Using default " + BP_ITERATIONS);
            }
        }

        System.out.println("TriangleTireworld CP Server starting on port " + PORT
            + " (bp.iterations=" + BP_ITERATIONS + ")");

        if (!loadAllInstancesConfig()) {
            System.err.println("FATAL: Could not load '" + INSTANCES_JSON_FILE + "'. Exiting.");
            System.exit(1);
        }
        System.out.println("Instance configurations loaded from " + INSTANCES_JSON_FILE);

        runServer();
    }

    // -------------------------------------------------------------------------
    // Server loop
    // -------------------------------------------------------------------------

    static void runServer() {
        try (ServerSocket serverSocket = new ServerSocket(PORT)) {
            System.out.println("TriangleTireworld CP Server listening on port " + PORT);

            while (true) {
                try (Socket clientSocket = serverSocket.accept();
                     PrintWriter out = new PrintWriter(
                         clientSocket.getOutputStream(), true);
                     BufferedReader in = new BufferedReader(
                         new InputStreamReader(clientSocket.getInputStream()))) {

                    System.out.println("\nClient connected: "
                        + clientSocket.getInetAddress());
                    out.println("OK Welcome");

                    currentInstanceId = null;
                    cp = null;

                    String line;
                    while ((line = in.readLine()) != null) {
                        String[] tokens = line.trim().split("\\s+");
                        String command  = tokens.length > 0
                            ? tokens[0].toUpperCase() : "";
                        String response =
                            "ERROR Unknown command '" + command + "'";

                        try {
                            switch (command) {
                                case "INIT":
                                    if (tokens.length == 2) {
                                        String id = tokens[1];
                                        if (loadInstanceParameters(id)) {
                                            currentInstanceId = id;
                                            response = "OK INIT successful for " + id;
                                            System.out.println(
                                                "Initialized for instance: " + id);
                                        } else {
                                            response = "ERROR Failed loading instance " + id;
                                            currentInstanceId = null;
                                        }
                                    } else {
                                        response = "ERROR Invalid INIT format."
                                            + " Expected: INIT <instance_id>";
                                    }
                                    break;

                                case "RESET":
                                    response = (currentInstanceId == null)
                                        ? "ERROR Must INIT first"
                                        : handleReset();
                                    break;

                                case "STEP":
                                    if (currentInstanceId == null || cp == null) {
                                        response = "ERROR Must INIT and RESET first";
                                    } else if (tokens.length == 4) {
                                        response = handleStep(
                                            tokens[1], tokens[2], tokens[3]);
                                    } else {
                                        response = "ERROR Invalid STEP format."
                                            + " Expected: STEP <idx> <act> <state>";
                                    }
                                    break;

                                case "QUERY_ETR":
                                    response = (currentInstanceId == null || cp == null)
                                        ? "ERROR Must INIT and RESET first"
                                        : handleQueryETR();
                                    break;

                                case "QUIT":
                                    response = "OK Goodbye";
                                    System.out.println("Client requested QUIT.");
                                    break;

                                default:
                                    System.out.println("Unknown command: " + line);
                                    break;
                            }
                        } catch (Exception e) {
                            System.err.println("Error processing '"
                                + line + "': " + e.getMessage());
                            e.printStackTrace();
                            response = "ERROR Processing failed: "
                                + e.getClass().getSimpleName();
                        }

                        out.println(response);
                        if ("QUIT".equalsIgnoreCase(command)) break;
                    }

                } catch (IOException e) {
                    System.err.println("WARN: Client connection error: " + e.getMessage());
                } finally {
                    System.out.println("Client disconnected.");
                    currentInstanceId = null;
                    cp = null;
                }
            }
        } catch (IOException e) {
            System.err.println("FATAL: Server socket error on port "
                + PORT + ": " + e.getMessage());
            System.exit(1);
        }
    }

    // -------------------------------------------------------------------------
    // Config loading
    // -------------------------------------------------------------------------

    static boolean loadAllInstancesConfig() {
        try {
            String json = new String(Files.readAllBytes(
                Paths.get(INSTANCES_JSON_FILE)));
            allInstancesConfig = new JSONObject(json);
            return true;
        } catch (IOException e) {
            System.err.println("ERROR reading '" + INSTANCES_JSON_FILE
                + "': " + e.getMessage());
            System.err.println("  CWD: "
                + Paths.get(".").toAbsolutePath().normalize());
            return false;
        } catch (JSONException e) {
            System.err.println("ERROR parsing JSON: " + e.getMessage());
            return false;
        }
    }

    static boolean loadInstanceParameters(String instanceId) {
        if (allInstancesConfig == null) {
            System.err.println("ERROR: Config not loaded.");
            return false;
        }
        if (!allInstancesConfig.has(instanceId)) {
            System.err.println("ERROR: Instance '" + instanceId + "' not found.");
            return false;
        }
        try {
            JSONObject d = allInstancesConfig.getJSONObject(instanceId);
            N               = d.getInt("n_locations");
            goalLocation    = d.getInt("goal_location");
            initialLocation = d.getInt("initial_location");
            flatProb        = d.getDouble("flat_prob");
            nbSteps         = d.getInt("cp_nbSteps");

            JSONArray sucJson = d.getJSONArray("successors");
            successors = new int[N][];
            for (int i = 0; i < N; i++) {
                JSONArray row = sucJson.getJSONArray(i);
                successors[i] = new int[row.length()];
                for (int j = 0; j < row.length(); j++)
                    successors[i][j] = row.getInt(j);
            }

            JSONArray spJson = d.getJSONArray("spare_locations");
            spareLocations = new int[spJson.length()];
            for (int i = 0; i < spJson.length(); i++)
                spareLocations[i] = spJson.getInt(i);

            System.out.println("Loaded instance '" + instanceId + "': N=" + N
                + " goal=" + goalLocation + " init=" + initialLocation
                + " flatProb=" + flatProb + " cp_nbSteps=" + nbSteps);
            return true;
        } catch (JSONException e) {
            System.err.println("ERROR parsing instance '"
                + instanceId + "': " + e.getMessage());
            return false;
        }
    }

    // -------------------------------------------------------------------------
    // Command handlers
    // -------------------------------------------------------------------------

    static String handleReset() {
        System.out.println("Handling RESET (TriangleTireworld, N=" + N + ")...");
        try {
            cp = makeSolver();

            int nbStates = 4 * N;

            // Markov sequence: init --a[0]--> S[0] --a[1]--> S[1] ... --a[T-1]--> S[T-1]
            // For action a[k], the vehicle is at position decodePos(S[k-1]) (or initialLocation
            // for k=0). sigma[k] = spare present at that pre-action position.
            // A_spare[l][k] = spare state at location l BEFORE action k (nbSteps+1 slots).

            S       = makeIntVarArray(cp, nbSteps, 0, nbStates - 1);
            M       = makeIntVarArray(cp, nbSteps,
                          0, TriangleTireworldDecomposition.N_ACTIONS - 1);
            sigma   = makeIntVarArray(cp, nbSteps, 0, 1);
            a       = makeIntVarArray(cp, nbSteps,
                          0, TriangleTireworldDecomposition.N_ENC_ACTIONS - 1);
            // nbSteps+1 spare slots: A_spare[l][k] = spare at l before action k.
            // A_spare[l][nbSteps] = spare at l after the last action (unused by Markov).
            A_spare = new IntVar[N][nbSteps + 1];
            for (int l = 0; l < N; l++)
                A_spare[l] = makeIntVarArray(cp, nbSteps + 1, 0, 1);

            totalReward = makeIntVar(cp, -nbSteps, 100);

            // 1. Fix initial spare layout (slot 0 = before action 0)
            Set<Integer> spareSet = new HashSet<>();
            for (int l : spareLocations) spareSet.add(l);
            for (int l = 0; l < N; l++) {
                A_spare[l][0].assign(spareSet.contains(l) ? 1 : 0);
            }

            // 2. pos_pre[k] = decodePos of the state BEFORE action k.
            //    pos_pre[0] = initialLocation (fixed).
            //    pos_pre[k] = decodePos(S[k-1]) for k >= 1.
            IntVar[] pos_pre = makeIntVarArray(cp, nbSteps, 0, N - 1);
            pos_pre[0].assign(initialLocation);
            int[][] posTbl = buildPosDecodingTable(N);
            for (int k = 1; k < nbSteps; k++) {
                IntVar[] scope = {S[k - 1], pos_pre[k]};
                cp.post(table(scope, posTbl));
            }

            // 3. sigma[k] = spare at pos_pre[k] before action k
            //    ShortTableCT on (pos_pre[k], A_spare[0][k], ..., A_spare[N-1][k], sigma[k])
            int[][] sigmaTbl = TriangleTireworldDecomposition.buildSigmaTable(N);
            int star = TriangleTireworldDecomposition.STAR;
            for (int k = 0; k < nbSteps; k++) {
                IntVar[] scope = new IntVar[N + 2];
                scope[0] = pos_pre[k];
                for (int l = 0; l < N; l++) scope[1 + l] = A_spare[l][k];
                scope[N + 1] = sigma[k];
                cp.post(shortTable(scope, sigmaTbl, star));
            }

            // 4. Action encoding: a[k] = M[k] + 6*sigma[k]
            int[][] encTbl = TriangleTireworldDecomposition.buildActionEncodingTable();
            for (int k = 0; k < nbSteps; k++) {
                IntVar[] scope = {M[k], sigma[k], a[k]};
                cp.post(table(scope, encTbl));
            }

            // 5. Spare dynamics: A_spare[l][k+1] from (pos_pre[k], M[k], A_spare[l][k])
            for (int l = 0; l < N; l++) {
                int[][] tbl = buildSpareDynamicsTableByPos(N, l);
                for (int k = 0; k < nbSteps; k++) {
                    // scope: (pos_pre[k], M[k], A_spare[l][k], A_spare[l][k+1])
                    IntVar[] scope = {pos_pre[k], M[k], A_spare[l][k], A_spare[l][k + 1]};
                    cp.post(table(scope, tbl));
                }
            }

            // 6. Markov constraint
            int initState = TriangleTireworldDecomposition.encodeState(
                initialLocation, 1, 0);  // intact tire, no spare loaded
            double[][][] P = TriangleTireworldDecomposition.buildTransition(
                N, successors, goalLocation, flatProb);
            int[][][] R = TriangleTireworldDecomposition.buildReward(
                N, goalLocation);

            cp.post(markov(a, S, P, R, initState, totalReward, true));

            currentStep = 0;
            cp.fixPoint();
            System.out.println("TriangleTireworld CP model reset successfully"
                + " (nbStates=" + nbStates + ", nbSteps=" + nbSteps + ").");
            return "OK RESET successful";

        } catch (Exception e) {
            System.err.println("Error during RESET: " + e.getMessage());
            e.printStackTrace();
            cp = null;
            return "ERROR RESET failed: " + e.getMessage();
        }
    }

    static String handleStep(String idxStr, String actStr, String stateStr) {
        if (cp == null) return "ERROR Must RESET first";
        try {
            int k   = Integer.parseInt(idxStr);
            int act = Integer.parseInt(actStr);
            int st  = Integer.parseInt(stateStr);

            if (k != currentStep)
                return "ERROR Step index mismatch. Expected "
                    + currentStep + ", got " + k;
            if (k < 0 || k >= nbSteps)
                return "ERROR Step index out of bounds [0.." + (nbSteps - 1) + "]";
            if (act < 0 || act >= TriangleTireworldDecomposition.N_ACTIONS)
                return "ERROR Action out of bounds [0.."
                    + (TriangleTireworldDecomposition.N_ACTIONS - 1) + "]";
            if (st < 0 || st >= 4 * N)
                return "ERROR State out of bounds [0.." + (4 * N - 1) + "]";

            M[k].assign(act);
            S[k].assign(st);
            cp.fixPoint();

            currentStep++;
            return "OK STEP processed";

        } catch (InconsistencyException e) {
            System.err.println("Inconsistency on STEP " + idxStr
                + " (act=" + actStr + ",s=" + stateStr + "): " + e.getMessage());
            return "ERROR Inconsistency STEP " + idxStr;
        } catch (NumberFormatException e) {
            return "ERROR Invalid number in STEP: " + e.getMessage();
        } catch (Exception e) {
            System.err.println("Unexpected error on STEP: " + e.getMessage());
            e.printStackTrace();
            return "ERROR Unexpected failure: " + e.getMessage();
        }
    }

    static String handleQueryETR() {
        if (cp == null) return "ERROR Must RESET first";
        try {
            cp.fixPoint();
            cp.vanillaBP(BP_ITERATIONS);

            double etr = 0.0;
            for (int v = totalReward.min(); v <= totalReward.max(); v++) {
                if (totalReward.contains(v))
                    etr += v * totalReward.marginal(v);
            }
            System.err.println("ETR at step " + currentStep + " = " + etr);
            return "ETR_VALUE " + etr;

        } catch (InconsistencyException e) {
            System.err.println("ERROR: Inconsistency during QUERY_ETR at step "
                + currentStep);
            return "ERROR Inconsistency QUERY_ETR " + currentStep;
        } catch (Exception e) {
            System.err.println("ERROR getting ETR: " + e.getMessage());
            e.printStackTrace();
            return "ETR_VALUE 0.0";
        }
    }

    // -------------------------------------------------------------------------
    // Internal helpers
    // -------------------------------------------------------------------------

    /**
     * Table mapping encoded state s to position p = decodePos(s).
     * Tuples: (s, p) for s in {0..4N-1}.
     */
    private static int[][] buildPosDecodingTable(int N) {
        int nbStates = 4 * N;
        int[][] tbl  = new int[nbStates][2];
        for (int s = 0; s < nbStates; s++) {
            tbl[s][0] = s;
            tbl[s][1] = TriangleTireworldDecomposition.decodePos(s);
        }
        return tbl;
    }

    /**
     * Spare dynamics table indexed by pre-action position p directly.
     * Tuples: (p, m_k, A^k_l, A^{k+1}_l) with p in {0..N-1}.
     */
    private static int[][] buildSpareDynamicsTableByPos(int N, int location) {
        int nActions = TriangleTireworldDecomposition.N_ACTIONS;
        int maxRows  = N * nActions * 2;
        int[][] tbl  = new int[maxRows][4];
        int row = 0;

        for (int p = 0; p < N; p++) {
            for (int m = 0; m < nActions; m++) {
                for (int Ak = 0; Ak <= 1; Ak++) {
                    int Ak1 = (p == location && m == 4 && Ak == 1) ? 0 : Ak;
                    tbl[row][0] = p;
                    tbl[row][1] = m;
                    tbl[row][2] = Ak;
                    tbl[row][3] = Ak1;
                    row++;
                }
            }
        }
        return tbl;
    }

    // -------------------------------------------------------------------------
    // MDP export — writes tireworld_instance_<id>_mdp.json to outDir
    //
    // JSON schema:
    //   states        : int   — number of states (4*N)
    //   actions       : int   — number of encoded actions (12)
    //   horizon       : int   — cp_nbSteps
    //   initial_state : int   — encodeState(initialLocation, 1, 0)
    //   goal_location : int
    //   P             : [s][a][s'] transition probabilities (sparse: only nonzero entries)
    //   R             : [s][a][s'] rewards (int, sparse: only nonzero entries)
    //
    // Sparse representation: P_sparse[s][a] = list of {s': prob} pairs (prob > 0).
    // R_sparse[s][a][s'] stored only where P[s][a][s'] > 0 and R != 0.
    //
    // Also validates: for every (s, a), Σ_{s'} P[s][a][s'] == 1.0 (±1e-9).
    // -------------------------------------------------------------------------

    static void exportMDP(String instanceId, String outDir) throws Exception {
        if (!loadInstanceParameters(instanceId)) {
            throw new RuntimeException("Failed to load instance " + instanceId);
        }

        double[][][] P = TriangleTireworldDecomposition.buildTransition(
                N, successors, goalLocation, flatProb);
        int[][][] R = TriangleTireworldDecomposition.buildReward(N, goalLocation);

        int nbStates  = 4 * N;
        int nbActions = TriangleTireworldDecomposition.N_ENC_ACTIONS;
        int initState = TriangleTireworldDecomposition.encodeState(initialLocation, 1, 0);

        // Sanity check: every (s,a) row sums to 1.0
        for (int s = 0; s < nbStates; s++) {
            for (int a = 0; a < nbActions; a++) {
                double sum = 0.0;
                for (int ns = 0; ns < nbStates; ns++) sum += P[s][a][ns];
                if (Math.abs(sum - 1.0) > 1e-9) {
                    throw new RuntimeException(
                        "Transition sanity FAILED: P[" + s + "][" + a + "] sums to " + sum);
                }
            }
        }
        System.out.println("Transition sanity check PASSED for instance " + instanceId);

        // Build JSON
        JSONObject root = new JSONObject();
        root.put("instance_id",    instanceId);
        root.put("states",         nbStates);
        root.put("actions",        nbActions);
        root.put("horizon",        nbSteps);
        root.put("initial_state",  initState);
        root.put("goal_location",  goalLocation);
        root.put("flat_prob",      flatProb);
        root.put("n_locations",    N);

        // Spare locations
        JSONArray sparesArr = new JSONArray();
        for (int l : spareLocations) sparesArr.put(l);
        root.put("spare_locations", sparesArr);

        // Successors
        JSONArray succArr = new JSONArray();
        for (int p = 0; p < N; p++) {
            JSONArray row = new JSONArray();
            for (int s : successors[p]) row.put(s);
            succArr.put(row);
        }
        root.put("successors", succArr);

        // Sparse P: P_sparse[s] = list of {a: list of [s', prob]}
        // We store as a flat list of [s, a, s_prime, prob] for easy Python loading.
        JSONArray pArr = new JSONArray();
        JSONArray rArr = new JSONArray();
        for (int s = 0; s < nbStates; s++) {
            for (int a = 0; a < nbActions; a++) {
                for (int ns = 0; ns < nbStates; ns++) {
                    double prob = P[s][a][ns];
                    if (prob > 0.0) {
                        JSONArray entry = new JSONArray();
                        entry.put(s); entry.put(a); entry.put(ns); entry.put(prob);
                        pArr.put(entry);
                        if (R[s][a][ns] != 0) {
                            JSONArray rEntry = new JSONArray();
                            rEntry.put(s); rEntry.put(a); rEntry.put(ns); rEntry.put(R[s][a][ns]);
                            rArr.put(rEntry);
                        }
                    }
                }
            }
        }
        root.put("P_sparse", pArr);
        root.put("R_sparse", rArr);

        // Write file
        String filename = outDir + "/tireworld_instance_" + instanceId + "_mdp.json";
        try (PrintWriter pw = new PrintWriter(new FileWriter(filename))) {
            pw.print(root.toString(2));
        }
        System.out.println("MDP exported to " + filename
            + "  (" + pArr.length() + " P entries, " + rArr.length() + " R entries)");
    }

    // Entry point for MDP export: java TriangleTireworldService export <instance_id> [outdir]
    static void runExport(String[] args) throws Exception {
        String instanceId = (args.length > 1) ? args[1] : "1";
        String outDir     = (args.length > 2) ? args[2] : ".";
        if (!loadAllInstancesConfig()) {
            throw new RuntimeException("Could not load " + INSTANCES_JSON_FILE);
        }
        exportMDP(instanceId, outDir);
    }

    // -------------------------------------------------------------------------
    // TEST SUPPORT — resets static state between unit tests
    // -------------------------------------------------------------------------
    static void resetStateForTests(JSONObject instancesConfig) {
        allInstancesConfig = instancesConfig;
        currentInstanceId  = null;
        N = goalLocation = initialLocation = nbSteps = -1;
        flatProb = Double.NaN;
        successors = null; spareLocations = null;
        cp = null; S = null; M = null; sigma = null; a = null;
        A_spare = null; totalReward = null;
        currentStep = 0;
    }
}
