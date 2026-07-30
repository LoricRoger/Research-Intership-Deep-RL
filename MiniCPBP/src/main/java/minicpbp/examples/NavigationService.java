package minicpbp.examples;

import minicpbp.engine.core.Constraint;
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
 * TCP server exposing a single-Markov-constraint CP model for Navigation MDP (ETR mode).
 *
 * Wire protocol (same as other CP services):
 *   INIT <instance_id>               -> OK INIT successful for <id>
 *   SET_INIT <state_idx>             -> OK SET_INIT <state_idx> successful (reset with given init)
 *   RESET                            -> OK RESET successful
 *   STEP <step_idx> <action> <state> -> OK STEP processed
 *   QUERY_ETR                        -> ETR_VALUE <float>
 *   QUIT                             -> OK Goodbye
 *
 * State encoding: state_idx = xi * nY + yj  (column-major, 0-based)
 *   dead_state = nX * nY        (only extra state; goal IS the grid cell goalPosIdx)
 *   nbStates   = nX * nY + 1
 *
 * Actions: 0=north, 1=south, 2=east, 3=west, 4=noop  (5 total)
 *
 * Transitions are deterministic for movement (bounded by grid edges), with a
 * Bernoulli(P(xi,yj)) disappearance probability on the destination cell.
 * DEAD and GOAL (goalPosIdx) are absorbing states. GOAL gives reward 0; all others -1.
 *
 * Default port: 12350
 */
public class NavigationService {

    private static int PORT = 12350;
    private static final String INSTANCES_JSON_FILE = "navigation_mdp_ippc2011.json";

    private static int    nX             = -1;
    private static int    nY             = -1;
    private static int    goalPosIdx     = -1;  // grid cell index of the goal (xi*nY+yj)
    private static int    deadStatIdx    = -1;  // = nX*nY
    private static int    initStatIdx    = -1;
    private static int    nbSteps        = -1;
    private static double[][] P_grid     = null;  // P_grid[xi][yj]

    private static final int NB_ACTIONS = 5;

    private static Solver  cp            = null;
    private static IntVar[] action       = null;
    private static IntVar[] state        = null;
    private static IntVar   totalReward  = null;
    private static int      currentStep  = 0;

    private static JSONObject allInstancesConfig = null;
    private static String     currentInstanceId  = null;
    private static int        BP_ITERATIONS      = 1;

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
                System.err.println("WARN: invalid port '" + args[0] + "', using " + PORT);
            }
        }
        String bpProp = System.getProperty("bp.iterations");
        if (bpProp != null) {
            try { BP_ITERATIONS = Integer.parseInt(bpProp); }
            catch (NumberFormatException e) {
                System.err.println("WARN: invalid bp.iterations '" + bpProp + "'");
            }
        }
        System.out.println("Navigation CP Server starting on port " + PORT
            + " (bp.iterations=" + BP_ITERATIONS + ")");

        if (!loadAllInstancesConfig()) {
            System.err.println("FATAL: could not load '" + INSTANCES_JSON_FILE + "'");
            System.exit(1);
        }
        runServer();
    }

    // -------------------------------------------------------------------------
    // Server loop
    // -------------------------------------------------------------------------

    static void runServer() {
        try (ServerSocket serverSocket = new ServerSocket(PORT)) {
            System.out.println("Navigation CP Server listening on port " + PORT);
            while (true) {
                try (Socket clientSocket = serverSocket.accept();
                     PrintWriter out = new PrintWriter(clientSocket.getOutputStream(), true);
                     BufferedReader in = new BufferedReader(
                             new InputStreamReader(clientSocket.getInputStream()))) {

                    System.out.println("\nClient connected: " + clientSocket.getInetAddress());
                    out.println("OK Welcome");
                    currentInstanceId = null;
                    cp = null;

                    String line;
                    while ((line = in.readLine()) != null) {
                        String[] tokens = line.trim().split("\\s+");
                        String cmd = tokens.length > 0 ? tokens[0].toUpperCase() : "";
                        String response = "ERROR Unknown command '" + cmd + "'";

                        try {
                            switch (cmd) {
                                case "INIT":
                                    if (tokens.length == 2) {
                                        if (loadInstanceParameters(tokens[1])) {
                                            currentInstanceId = tokens[1];
                                            response = "OK INIT successful for " + currentInstanceId;
                                        } else {
                                            response = "ERROR Failed loading instance " + tokens[1];
                                        }
                                    } else {
                                        response = "ERROR Invalid INIT format";
                                    }
                                    break;

                                case "SET_INIT":
                                    if (currentInstanceId == null)
                                        response = "ERROR Must INIT first";
                                    else if (tokens.length == 2)
                                        response = handleSetInit(tokens[1]);
                                    else
                                        response = "ERROR Invalid SET_INIT format";
                                    break;

                                case "RESET":
                                    response = (currentInstanceId == null)
                                        ? "ERROR Must INIT first" : handleReset();
                                    break;

                                case "STEP":
                                    if (currentInstanceId == null || cp == null)
                                        response = "ERROR Must INIT and RESET first";
                                    else if (tokens.length == 4)
                                        response = handleStep(tokens[1], tokens[2], tokens[3]);
                                    else
                                        response = "ERROR Invalid STEP format";
                                    break;

                                case "QUERY_ETR":
                                    response = (currentInstanceId == null || cp == null)
                                        ? "ERROR Must INIT and RESET first" : handleQueryETR();
                                    break;

                                case "QUIT":
                                    response = "OK Goodbye";
                                    break;

                                default:
                                    System.out.println("Unknown command: " + line);
                                    break;
                            }
                        } catch (Exception e) {
                            System.err.println("Error processing '" + line + "': " + e.getMessage());
                            e.printStackTrace();
                            response = "ERROR Processing failed: " + e.getClass().getSimpleName();
                        }

                        out.println(response);
                        if ("QUIT".equalsIgnoreCase(cmd)) break;
                    }
                } catch (IOException e) {
                    System.err.println("WARN: client error: " + e.getMessage());
                } finally {
                    System.out.println("Client disconnected.");
                    currentInstanceId = null;
                    cp = null;
                }
            }
        } catch (IOException e) {
            System.err.println("FATAL: server error on port " + PORT + ": " + e.getMessage());
            System.exit(1);
        }
    }

    // -------------------------------------------------------------------------
    // Config loading
    // -------------------------------------------------------------------------

    static boolean loadAllInstancesConfig() {
        try {
            String json = new String(Files.readAllBytes(Paths.get(INSTANCES_JSON_FILE)));
            allInstancesConfig = new JSONObject(json);
            return true;
        } catch (IOException e) {
            System.err.println("ERROR reading '" + INSTANCES_JSON_FILE + "': " + e.getMessage());
            System.err.println("  CWD: " + Paths.get(".").toAbsolutePath().normalize());
            return false;
        } catch (JSONException e) {
            System.err.println("ERROR parsing JSON: " + e.getMessage());
            return false;
        }
    }

    static boolean loadInstanceParameters(String instanceId) {
        if (allInstancesConfig == null || !allInstancesConfig.has(instanceId)) {
            System.err.println("ERROR: instance '" + instanceId + "' not found");
            return false;
        }
        try {
            JSONObject d = allInstancesConfig.getJSONObject(instanceId);
            nX          = d.getInt("grid_x");
            nY          = d.getInt("grid_y");
            initStatIdx = d.getInt("init_state_idx");
            goalPosIdx  = d.getInt("goal_pos_idx");
            deadStatIdx = nX * nY;
            nbSteps     = d.getInt("cp_nbSteps");

            // Load P[xi][yj] from JSON object "P": {"xi,yj": value, ...}
            P_grid = new double[nX][nY];
            JSONObject Pjson = d.getJSONObject("P");
            for (String key : Pjson.keySet()) {
                String[] parts = key.split(",");
                int xi = Integer.parseInt(parts[0]);
                int yj = Integer.parseInt(parts[1]);
                P_grid[xi][yj] = Pjson.getDouble(key);
            }
            System.out.println("Loaded instance '" + instanceId + "': grid=" + nX + "x" + nY
                + " nbStates=" + (nX * nY + 1) + " init=" + initStatIdx
                + " goalPos=" + goalPosIdx + " dead=" + deadStatIdx);
            return true;
        } catch (JSONException e) {
            System.err.println("ERROR parsing instance '" + instanceId + "': " + e.getMessage());
            return false;
        }
    }

    // -------------------------------------------------------------------------
    // Command handlers
    // -------------------------------------------------------------------------

    static String handleSetInit(String idxStr) {
        try {
            int idx = Integer.parseInt(idxStr);
            int nbStates = nX * nY + 1;
            if (idx < 0 || idx >= nbStates)
                return "ERROR SET_INIT index out of bounds: " + idx;
            initStatIdx = idx;
            return handleReset().replace("RESET", "SET_INIT " + idx);
        } catch (NumberFormatException e) {
            return "ERROR Invalid SET_INIT index: " + idxStr;
        }
    }

    static String handleReset() {
        try {
            int nbStates = nX * nY + 1;
            cp = makeSolver();
            action      = makeIntVarArray(cp, nbSteps, NB_ACTIONS);
            state       = makeIntVarArray(cp, nbSteps, nbStates);
            totalReward = makeIntVar(cp, -nbSteps, 0);

            double[][][] P = buildTransitionMatrix(nbStates);
            int[][][]    R = buildRewardMatrix(nbStates);

            Constraint markovC = markov(action, state, P, R, initStatIdx, totalReward, true);
            cp.post(markovC);

            currentStep = 0;
            cp.fixPoint();
            System.out.println("Navigation CP model reset (nbStates=" + nbStates
                + ", nbSteps=" + nbSteps + ").");
            return "OK RESET successful";
        } catch (Exception e) {
            System.err.println("Error during RESET: " + e.getMessage());
            e.printStackTrace();
            cp = null;
            return "ERROR RESET failed: " + e.getMessage();
        }
    }

    static String handleStep(String iStr, String aStr, String sStr) {
        int i = -1, a = -1, s = -1;
        try {
            i = Integer.parseInt(iStr);
            a = Integer.parseInt(aStr);
            s = Integer.parseInt(sStr);

            if (i != currentStep)
                return "ERROR step index mismatch: expected " + currentStep + " got " + i;
            if (i < 0 || i >= nbSteps)        return "ERROR step out of bounds";
            if (a < 0 || a >= NB_ACTIONS)     return "ERROR action out of bounds";
            if (s < 0 || s >= nX * nY + 1)   return "ERROR state out of bounds";

            action[i].assign(a);
            state[i].assign(s);
            state[nbSteps-1].assign(goalPosIdx);  // Regarder si le -1 est nécessaire #TODO
            cp.fixPoint();
            currentStep++;
            return "OK STEP processed";

        } catch (InconsistencyException e) {
            System.err.println("Inconsistency on STEP " + i + " (a=" + a + ",s=" + s + ")");
            return "ERROR Inconsistency STEP " + iStr;
        } catch (NumberFormatException e) {
            return "ERROR Invalid number in STEP: " + e.getMessage();
        } catch (Exception e) {
            System.err.println("Unexpected error on STEP: " + e.getMessage());
            e.printStackTrace();
            return "ERROR Unexpected failure: " + e.getMessage();
        }
    }

    static String handleQueryETR() {
        try {
            cp.fixPoint();
            cp.vanillaBP(BP_ITERATIONS);

            double etr = 0.0;
            for (int v = totalReward.min(); v <= totalReward.max(); v++) {
                if (totalReward.contains(v)) etr += v * totalReward.marginal(v);
            }
            System.err.println("ETR at step " + currentStep + " = " + etr);
            return "ETR_VALUE " + etr;
        } catch (InconsistencyException e) {
            System.err.println("WARN: inconsistency during QUERY_ETR");
            return "ETR_VALUE 0.0";
        } catch (Exception e) {
            System.err.println("ERROR during QUERY_ETR: " + e.getMessage());
            return "ETR_VALUE 0.0";
        }
    }

    // -------------------------------------------------------------------------
    // Transition and reward matrices
    // -------------------------------------------------------------------------

    /**
     * Builds P[s][a][s'] for Navigation.
     *
     * Encoding: state_idx = xi * nY + yj
     *   dead = nX*nY  (only extra state)
     *   goal = goalPosIdx  (the physical goal cell — absorbing, reward 0)
     *   nbStates = nX*nY + 1
     *
     * Actions: 0=north (yj+1), 1=south (yj-1), 2=east (xi+1), 3=west (xi-1), 4=noop
     *
     * Transitions:
     *   - DEAD and GOAL (goalPosIdx) are absorbing.
     *   - From any other (xi, yj): apply action → dest.
     *     If out of bounds → stay at (xi, yj).
     *     If dest == goalPosIdx → P[src][a][goalPosIdx] = 1.0.
     *     Otherwise → dest with prob (1 - P[dxi][dyj]), dead with prob P[dxi][dyj].
     */
    private static double[][][] buildTransitionMatrix(int nbStates) {
        double[][][] P = new double[nbStates][NB_ACTIONS][nbStates];

        // Absorbing: DEAD and GOAL self-loop
        for (int a = 0; a < NB_ACTIONS; a++) {
            P[deadStatIdx][a][deadStatIdx] = 1.0;
            P[goalPosIdx][a][goalPosIdx]   = 1.0;
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
                        double pDie = P_grid[dxi][dyj];
                        P[src][a][deadStatIdx] += pDie;
                        P[src][a][dest]        += 1.0 - pDie;
                    }
                }
            }
        }
        return P;
    }

    /**
     * Reward: R[s][a][s'] = 0 if s'==goalPosIdx, else -1.
     */
    private static int[][][] buildRewardMatrix(int nbStates) {
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
    // MDP export — same JSON format as TriangleTireworldService.exportMDP()
    //
    // Sparse representation: flat list of [s, a, s', prob] for P_sparse,
    //                        flat list of [s, a, s', reward] for R_sparse.
    // Also validates: every (s,a) row of P sums to 1.0 (±1e-9).
    // -------------------------------------------------------------------------

    static void exportMDP(String instanceId, String outDir) throws Exception {
        if (!loadInstanceParameters(instanceId)) {
            throw new RuntimeException("Failed to load instance " + instanceId);
        }

        int nbStates = nX * nY + 1;

        // Reuse the private builders (no logic duplication)
        double[][][] P = buildTransitionMatrix(nbStates);
        int[][][]    R = buildRewardMatrix(nbStates);

        // Sanity check: every (s,a) row sums to 1.0
        for (int s = 0; s < nbStates; s++) {
            for (int a = 0; a < NB_ACTIONS; a++) {
                double sum = 0.0;
                for (int ns = 0; ns < nbStates; ns++) sum += P[s][a][ns];
                if (Math.abs(sum - 1.0) > 1e-9) {
                    throw new RuntimeException(
                        "Transition sanity FAILED: P[" + s + "][" + a + "] sums to " + sum);
                }
            }
        }
        System.out.println("Transition sanity check PASSED for instance " + instanceId);

        // Build JSON — same keys as TriangleTireworldService.exportMDP()
        JSONObject root = new JSONObject();
        root.put("instance_id",   instanceId);
        root.put("states",        nbStates);
        root.put("actions",       NB_ACTIONS);
        root.put("horizon",       nbSteps);
        root.put("initial_state", initStatIdx);
        root.put("goal_state",    goalPosIdx);
        root.put("grid_x",        nX);
        root.put("grid_y",        nY);

        // P_grid as JSON object {"xi,yj": prob} — domain-specific, replaces spare_locations/successors
        JSONObject pGridJson = new JSONObject();
        for (int xi = 0; xi < nX; xi++) {
            for (int yj = 0; yj < nY; yj++) {
                if (P_grid[xi][yj] > 0.0) {
                    pGridJson.put(xi + "," + yj, P_grid[xi][yj]);
                }
            }
        }
        root.put("P_grid", pGridJson);

        // Sparse P and R — same flat [s, a, s', value] format as TriangleTireworldService
        JSONArray pArr = new JSONArray();
        JSONArray rArr = new JSONArray();
        for (int s = 0; s < nbStates; s++) {
            for (int a = 0; a < NB_ACTIONS; a++) {
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

        // Write file — filename mirrors tireworld convention
        String filename = outDir + "/navigation_instance_" + instanceId + "_mdp.json";
        try (PrintWriter pw = new PrintWriter(new FileWriter(filename))) {
            pw.print(root.toString(2));
        }
        System.out.println("MDP exported to " + filename
            + "  (" + pArr.length() + " P entries, " + rArr.length() + " R entries)");
    }

    // Entry point for MDP export: java NavigationService export <instance_id> [outdir]
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
        nX = nY = goalPosIdx = deadStatIdx = initStatIdx = nbSteps = -1;
        P_grid = null;
        cp = null; action = null; state = null; totalReward = null;
        currentStep = 0;
    }
}
