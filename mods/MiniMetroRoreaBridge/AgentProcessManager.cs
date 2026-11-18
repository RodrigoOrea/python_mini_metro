using System;
using System.Diagnostics;
using System.Globalization;
using System.IO;

namespace MiniMetroRoreaBridge
{
    /// <summary>
    /// Manages the lifecycle of the external Python AI agent process (<c>scripts/play_real_metro.py</c>)
    /// so players can toggle autonomous play directly from the in-game HUD.
    /// </summary>
    public sealed class AgentProcessManager
    {
        private const string DefaultRepoPath = "/home/rodrigo/python_mini_metro";
        private const int DefaultMaxDecisions = 1000000;

        private readonly string _configuredRepoPath;
        private readonly string _configuredPythonPath;
        private readonly Action<string> _logInfo;
        private readonly Action<string> _logError;
        private readonly object _gate = new object();

        private Process _process;

        public AgentProcessManager(
            string configuredRepoPath,
            string configuredPythonPath,
            Action<string> logInfo,
            Action<string> logError)
        {
            _configuredRepoPath = configuredRepoPath;
            _configuredPythonPath = configuredPythonPath;
            _logInfo = logInfo ?? (_ => { });
            _logError = logError ?? (_ => { });
        }

        public bool IsRunning
        {
            get
            {
                lock (_gate)
                {
                    if (_process == null) return false;
                    try
                    {
                        return !_process.HasExited;
                    }
                    catch
                    {
                        return false;
                    }
                }
            }
        }

        public string LastError { get; private set; }

        public void Poll()
        {
            lock (_gate)
            {
                if (_process == null) return;
                bool exited;
                try
                {
                    exited = _process.HasExited;
                }
                catch
                {
                    exited = true;
                }
                if (!exited) return;

                int exitCode = 0;
                try
                {
                    exitCode = _process.ExitCode;
                }
                catch
                {
                    // Ignore exit code read errors during teardown.
                }
                CleanupProcessLocked();
                if (exitCode != 0)
                {
                    LastError = "Agent exited (" + exitCode.ToString(CultureInfo.InvariantCulture) + ")";
                    _logError("AI agent process exited with code " + exitCode);
                }
                else
                {
                    _logInfo("AI agent process finished.");
                }
            }
        }

        public bool Start(int port)
        {
            lock (_gate)
            {
                if (_process != null)
                {
                    try
                    {
                        if (!_process.HasExited) return true;
                    }
                    catch
                    {
                        // Fall through to cleanup.
                    }
                    CleanupProcessLocked();
                }

                LastError = null;
                string repoPath = ResolveRepoPath();
                string scriptPath = Path.Combine(repoPath, "scripts", "play_real_metro.py");
                if (!File.Exists(scriptPath))
                {
                    LastError = "play_real_metro.py not found";
                    _logError("Cannot start AI agent; missing script: " + scriptPath);
                    return false;
                }

                string pythonPath = ResolvePythonPath(repoPath);
                string args = string.Format(
                    CultureInfo.InvariantCulture,
                    "\"{0}\" --live --host {1} --port {2} --max-decisions {3}",
                    scriptPath,
                    BridgeProtocol.DefaultHost,
                    port,
                    DefaultMaxDecisions);

                try
                {
                    var psi = new ProcessStartInfo
                    {
                        FileName = pythonPath,
                        Arguments = args,
                        WorkingDirectory = repoPath,
                        UseShellExecute = false,
                        CreateNoWindow = true,
                        RedirectStandardOutput = true,
                        RedirectStandardError = true
                    };
                    psi.EnvironmentVariables.Remove("PYTHONHOME");
                    psi.EnvironmentVariables.Remove("PYTHONPATH");

                    var proc = new Process { StartInfo = psi, EnableRaisingEvents = false };
                    proc.OutputDataReceived += (_, e) =>
                    {
                        if (!string.IsNullOrWhiteSpace(e.Data))
                            _logInfo("[AI Agent] " + e.Data);
                    };
                    proc.ErrorDataReceived += (_, e) =>
                    {
                        if (!string.IsNullOrWhiteSpace(e.Data))
                            _logError("[AI Agent] " + e.Data);
                    };

                    if (!proc.Start())
                    {
                        proc.Dispose();
                        LastError = "Failed to start Python";
                        _logError("Process.Start returned false for: " + pythonPath);
                        return false;
                    }

                    proc.BeginOutputReadLine();
                    proc.BeginErrorReadLine();
                    _process = proc;
                    _logInfo("Started AI agent (" + pythonPath + " on port " + port + ")");
                    return true;
                }
                catch (Exception ex)
                {
                    LastError = ex.Message;
                    _logError("Failed to start AI agent: " + ex);
                    return false;
                }
            }
        }

        public void Stop()
        {
            lock (_gate)
            {
                if (_process == null) return;
                try
                {
                    if (!_process.HasExited)
                    {
                        _process.Kill();
                        _process.WaitForExit(500);
                        _logInfo("Stopped AI agent process.");
                    }
                }
                catch (Exception ex)
                {
                    _logError("Error stopping AI agent process: " + ex.Message);
                }
                finally
                {
                    CleanupProcessLocked();
                }
            }
        }

        private void CleanupProcessLocked()
        {
            if (_process == null) return;
            try
            {
                _process.Dispose();
            }
            catch
            {
                // Ignore dispose errors.
            }
            _process = null;
        }

        private string ResolveRepoPath()
        {
            string envRepo = Environment.GetEnvironmentVariable("MINIMETRO_AGENT_REPO");
            if (!string.IsNullOrWhiteSpace(envRepo) && Directory.Exists(envRepo.Trim()))
                return envRepo.Trim();
            if (!string.IsNullOrWhiteSpace(_configuredRepoPath) && Directory.Exists(_configuredRepoPath.Trim()))
                return _configuredRepoPath.Trim();
            return DefaultRepoPath;
        }

        private string ResolvePythonPath(string repoPath)
        {
            string envPython = Environment.GetEnvironmentVariable("MINIMETRO_AGENT_PYTHON");
            if (!string.IsNullOrWhiteSpace(envPython) && File.Exists(envPython.Trim()))
                return envPython.Trim();
            if (!string.IsNullOrWhiteSpace(_configuredPythonPath) && File.Exists(_configuredPythonPath.Trim()))
                return _configuredPythonPath.Trim();

            string venvPython = Path.Combine(repoPath, ".venv", "bin", "python");
            if (File.Exists(venvPython)) return venvPython;

            string venvPythonWin = Path.Combine(repoPath, ".venv", "Scripts", "python.exe");
            if (File.Exists(venvPythonWin)) return venvPythonWin;

            return "python3";
        }
    }
}
