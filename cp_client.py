"""Client TCP pour le serveur CP Java (TrafficCrossingCPService)."""

import socket

CP_SERVER_HOST = "localhost"
CP_SERVER_PORT = 12346
CP_TIMEOUT     = 120.0


class CPClient:
    """Communicates with the Java CP server over a TCP socket.

    Protocol (text, newline-terminated):
        INIT <instance_id>               -> OK INIT successful for <id>
        RESET                            -> OK RESET successful
        STEP <step_idx> <action> <state> -> OK STEP processed
        QUERY_ETR                        -> ETR_VALUE <float>
        QUIT                             -> OK Goodbye

    Session state
    -------------
    _instance_id     : memorized at init(), replayed on reconnect
    episode_broken   : set True after a reconnect mid-episode; cleared at
                       the next reset() call. While True, step/query_etr
                       return immediately without touching the Java server.
    """

    def __init__(self, host: str = CP_SERVER_HOST, port: int = CP_SERVER_PORT):
        self.host = host
        self.port = port
        self.socket = None
        self._reader = None  # line-buffered file wrapper — guarantees one response per recv
        self.is_connected = False
        self._instance_id: str | None = None
        self.episode_broken: bool = False

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Connects to the server and reads the welcome message."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.settimeout(CP_TIMEOUT)
            sock.connect((self.host, self.port))
            self.socket = sock
            self._reader = sock.makefile("r", encoding="utf-8")
            welcome = self._reader.readline().strip()
            if welcome.startswith("OK Welcome"):
                self.is_connected = True
                return True
            sock.close()
            self.socket = None
            self._reader = None
            print(f"[CPClient] ERROR: Unexpected welcome: '{welcome}'")
            return False
        except ConnectionRefusedError:
            print(f"[CPClient] ERROR: Connection refused at {self.host}:{self.port}. Is the server running?")
            return False
        except socket.timeout:
            print(f"[CPClient] ERROR: Timeout connecting to {self.host}:{self.port}.")
            return False
        except Exception as e:
            print(f"[CPClient] ERROR: Unexpected connect error: {e}")
            return False

    def close(self):
        """Closes the socket connection."""
        if self._reader:
            try:
                self._reader.close()
            except Exception:
                pass
            self._reader = None
        if self.socket:
            try:
                self.socket.shutdown(socket.SHUT_RDWR)
                self.socket.close()
            except Exception:
                pass
        self.socket = None
        self.is_connected = False

    # ------------------------------------------------------------------
    # Send / receive — with single reconnect on transient errors
    # ------------------------------------------------------------------

    def _raw_send_receive(self, command: str) -> str:
        """Low-level send/receive — no retry logic."""
        if not self.is_connected or self.socket is None or self._reader is None:
            return "ERROR Not Connected"
        try:
            self.socket.sendall(f"{command.strip()}\n".encode("utf-8"))
            response = self._reader.readline().strip()
            return response if response else "ERROR Empty Response"
        except socket.timeout:
            print(f"[CPClient] ERROR: Timeout waiting for response to: {command}")
            self.close()
            return "ERROR Timeout"
        except socket.error as e:
            print(f"[CPClient] ERROR: Socket error: {e}")
            self.close()
            return f"ERROR Socket Error: {e}"
        except Exception as e:
            print(f"[CPClient] ERROR: Unexpected send/receive error: {e}")
            self.close()
            return f"ERROR Communication Failed: {e}"

    def _try_reconnect(self) -> bool:
        """One reconnect attempt: connect() + INIT + RESET.

        Sets episode_broken=True on success so the caller skips the rest of
        the current episode. Returns False (no retry) on any failure.
        """
        self.close()
        print("[CPClient] INFO: Attempting reconnect ...")
        if not self.connect():
            print("[CPClient] ERROR: Reconnect failed (could not connect).")
            return False
        if self._instance_id is not None:
            resp = self._raw_send_receive(f"INIT {self._instance_id}")
            if not resp.startswith("OK INIT"):
                print(f"[CPClient] ERROR: Reconnect failed (INIT error): {resp}")
                self.close()
                return False
        resp = self._raw_send_receive("RESET")
        if not resp.startswith("OK RESET"):
            print(f"[CPClient] ERROR: Reconnect failed (RESET error): {resp}")
            self.close()
            return False
        self.episode_broken = True
        print("[CPClient] INFO: Reconnected — episode marked broken, will resume at next reset().")
        return True

    def send_receive(self, command: str) -> str:
        """Sends a command and returns exactly one response line.

        On timeout or socket error, attempts a single reconnect (connect +
        INIT + RESET). If the reconnect succeeds the episode is marked broken
        and the original error string is returned (caller should not retry the
        command — the server state has been reset). If the reconnect fails the
        error string is also returned.
        """
        response = self._raw_send_receive(command)
        if response.startswith("ERROR"):
            self._try_reconnect()
        return response

    # ------------------------------------------------------------------
    # Protocol methods
    # ------------------------------------------------------------------

    def init(self, instance_id: str) -> bool:
        """Sends INIT <instance_id>. Returns True on success."""
        self._instance_id = instance_id
        response = self.send_receive(f"INIT {instance_id}")
        if response.startswith("OK INIT"):
            return True
        print(f"[CPClient] ERROR: INIT failed: {response}")
        return False

    def reset(self) -> bool:
        """Sends RESET. Clears episode_broken so shaping resumes."""
        self.episode_broken = False
        response = self.send_receive("RESET")
        if response.startswith("OK RESET"):
            return True
        print(f"[CPClient] ERROR: RESET failed: {response}")
        return False

    def send_step(self, step_idx: int, action: int, next_state: int) -> bool:
        """Sends STEP <step_idx> <action> <next_state>. Returns True on success."""
        if self.episode_broken:
            return False
        response = self.send_receive(f"STEP {step_idx} {action} {next_state}")
        if response.startswith("OK STEP"):
            return True
        print(f"[CPClient] WARN: STEP {step_idx} failed: {response}")
        return False

    def send_step_sysadmin(self, step_idx: int, reboot_idx: int, alive_mask: int) -> bool:
        """Sends STEP <step_idx> <reboot_idx> <alive_mask> (SysAdmin protocol). Returns True on success."""
        if self.episode_broken:
            return False
        response = self.send_receive(f"STEP {step_idx} {reboot_idx} {alive_mask}")
        if response.startswith("OK STEP"):
            return True
        print(f"[CPClient] WARN: STEP {step_idx} failed: {response}")
        return False

    def query_etr(self) -> float | None:
        """Queries ETR (P(reaching goal)). Returns float in [0, 1] or None on error."""
        if self.episode_broken:
            return None
        response = self.send_receive("QUERY_ETR")
        if response.startswith("ETR_VALUE "):
            try:
                return float(response.split()[1])
            except (ValueError, IndexError):
                print(f"[CPClient] ERROR: Could not parse ETR_VALUE response: '{response}'")
                return None
        if response.startswith("ERROR"):
            return None
        print(f"[CPClient] WARN: Unexpected QUERY_ETR response: '{response}'")
        return None
