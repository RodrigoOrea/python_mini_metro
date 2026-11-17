using System;
using System.Collections.Generic;
using System.Reflection;
using System.Threading;
using BepInEx;
using BepInEx.Configuration;
using UnityEngine;

namespace MiniMetroRoreaBridge
{
    [BepInPlugin(BridgeProtocol.PluginGuid, BridgeProtocol.PluginName, BridgeProtocol.PluginVersion)]
    public sealed class Plugin : BaseUnityPlugin
    {
        private const int MaxRequestsPerFrame = 64;
        private const string StateReaderTypeName = "MiniMetroRoreaBridge.StateReader";
        private const string ActionExecutorTypeName = "MiniMetroRoreaBridge.ActionExecutor";
        private const BindingFlags InstanceFlags = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;

        private static readonly FieldInfo MainControllerField = typeof(Main).GetField("controller", InstanceFlags);
        private static readonly FieldInfo MenuControllerMenuField = typeof(MenuController).GetField("menu", InstanceFlags);

        private readonly LineRegistry _lineRegistry = new LineRegistry();
        private RpcServer _server;
        private IBridgeStateReader _stateReader;
        private IBridgeActionExecutor _actionExecutor;
        private long _seq;
        private string _autostartCity;
        private bool _autostartAttempted;

        private void Awake()
        {
            int port = ResolvePort();
            int timeoutMs = ResolveTimeoutMs();
            string rawAutostart = Environment.GetEnvironmentVariable("MINIMETRO_BRIDGE_AUTOSTART_CITY");
            _autostartCity = string.IsNullOrWhiteSpace(rawAutostart) ? null : rawAutostart.Trim();

            _stateReader = TryCreateComponent<IBridgeStateReader>(StateReaderTypeName);
            _actionExecutor = TryCreateComponent<IBridgeActionExecutor>(ActionExecutorTypeName);

            _server = new RpcServer(
                port: port,
                mainThreadTimeoutMs: timeoutMs,
                getSeq: () => Interlocked.Read(ref _seq),
                logInfo: msg => Logger.LogInfo(msg),
                logError: msg => Logger.LogError(msg));
            _server.Start();
            Logger.LogInfo("MiniMetroRoreaBridge v" + BridgeProtocol.PluginVersion + " listening on " + BridgeProtocol.DefaultHost + ":" + _server.Port);
        }

        private void Update()
        {
            TryAutostartCity();
            if (_server == null) return;
            for (int i = 0; i < MaxRequestsPerFrame; i++)
            {
                if (!_server.TryDequeue(out PendingRpcWork work)) break;
                if (work.IsAbandoned) continue;
                RpcResponseDto response;
                try
                {
                    response = DispatchRequest(work.Request);
                }
                catch (Exception ex)
                {
                    Logger.LogError("Unhandled bridge dispatch error (" + work.Request.Cmd + "): " + ex);
                    response = RpcResponseDto.Failure(work.Request.Id, Interlocked.Read(ref _seq), BridgeErrorCodes.INTERNAL_ERROR, ex.Message);
                }
                work.Complete(response);
            }
        }

        private void OnDestroy() => StopServer();
        private void OnApplicationQuit() => StopServer();

        private void StopServer()
        {
            RpcServer srv = _server;
            _server = null;
            if (srv != null)
            {
                srv.Stop();
                Logger.LogInfo("MiniMetroRoreaBridge TCP RPC server stopped.");
            }
        }

        private RpcResponseDto DispatchRequest(RpcRequestDto req)
        {
            long seq = Interlocked.Increment(ref _seq);
            string cmd = req.Cmd;

            if (cmd == BridgeCommands.PING)
            {
                if (req.Args != null && req.Args.Count > 0)
                    return RpcResponseDto.Failure(req.Id, seq, BridgeErrorCodes.INVALID_ARGS, "PING does not accept args");
                var pongData = new Dictionary<string, object>(StringComparer.Ordinal)
                {
                    { "pong", true },
                    { "protocol", BridgeProtocol.Version },
                    { "unity_version", Application.unityVersion },
                    { "plugin_version", BridgeProtocol.PluginVersion },
                    { "seq", seq }
                };
                return RpcResponseDto.Success(req.Id, seq, pongData);
            }

            if (cmd == BridgeCommands.GET_STATE)
            {
                if (req.Args != null && req.Args.Count > 0)
                    return RpcResponseDto.Failure(req.Id, seq, BridgeErrorCodes.INVALID_ARGS, "GET_STATE does not accept args");
                if (_stateReader == null)
                    return RpcResponseDto.Failure(req.Id, seq, BridgeErrorCodes.INTERNAL_ERROR, "StateReader is not loaded");
                SnapshotDto snap = _stateReader.CaptureSnapshot(seq, _lineRegistry);
                if (snap == null)
                    return RpcResponseDto.Failure(req.Id, seq, BridgeErrorCodes.INTERNAL_ERROR, "StateReader returned null snapshot");
                var stateData = new Dictionary<string, object>(StringComparer.Ordinal) { { "snapshot", snap.ToDictionary() } };
                return RpcResponseDto.Success(req.Id, seq, stateData);
            }

            if (!BridgeCommands.IsKnownActionOrControl(cmd))
                return RpcResponseDto.Failure(req.Id, seq, BridgeErrorCodes.UNKNOWN_COMMAND, "Unknown command: " + cmd);

            if (_actionExecutor == null)
                return RpcResponseDto.Failure(req.Id, seq, BridgeErrorCodes.INTERNAL_ERROR, "ActionExecutor is not loaded");

            RpcResponseDto execResp = _actionExecutor.Execute(req, _lineRegistry, _stateReader, seq);
            return execResp ?? RpcResponseDto.Failure(req.Id, seq, BridgeErrorCodes.INTERNAL_ERROR, "ActionExecutor returned null response");
        }

        private TInterface TryCreateComponent<TInterface>(string typeName) where TInterface : class
        {
            try
            {
                Type type = typeof(Plugin).Assembly.GetType(typeName, throwOnError: false);
                if (type == null)
                {
                    Logger.LogInfo("Optional bridge component not found yet: " + typeName);
                    return null;
                }
                if (!typeof(TInterface).IsAssignableFrom(type))
                {
                    Logger.LogWarning(typeName + " does not implement " + typeof(TInterface).Name);
                    return null;
                }
                return Activator.CreateInstance(type) as TInterface;
            }
            catch (Exception ex)
            {
                Logger.LogError("Failed to initialize " + typeName + ": " + ex);
                return null;
            }
        }

        private int ResolvePort()
        {
            ConfigEntry<int> entry = Config.Bind("Network", "Port", BridgeProtocol.DefaultPort, "TCP NDJSON RPC port on 127.0.0.1");
            int port = entry.Value;
            string env = Environment.GetEnvironmentVariable("MINIMETRO_BRIDGE_PORT");
            if (!string.IsNullOrWhiteSpace(env) && int.TryParse(env, out int parsed) && parsed >= 1 && parsed <= 65535)
                port = parsed;
            return port >= 1 && port <= 65535 ? port : BridgeProtocol.DefaultPort;
        }

        private int ResolveTimeoutMs()
        {
            ConfigEntry<int> entry = Config.Bind("Network", "MainThreadTimeoutMs", 5000, "Max milliseconds to wait for Unity main thread per RPC");
            return entry.Value > 0 ? entry.Value : 5000;
        }

        private void TryAutostartCity()
        {
            if (_autostartAttempted || string.IsNullOrEmpty(_autostartCity)) return;
            Main main = Main.Instance;
            if (main == null) return;
            if (main.IsInGame)
            {
                _autostartAttempted = true;
                return;
            }
            if (!(MainControllerField?.GetValue(main) is MenuController menuController)) return;
            if (!(MenuControllerMenuField?.GetValue(menuController) is Menu menu) || menu.Option == MenuOption.None) return;

            _autostartAttempted = true;
            try
            {
                CityDefinition cityDef = CityDatabase.Instance[_autostartCity];
                if (cityDef == null)
                {
                    Logger.LogError("Autostart city not found in CityDatabase: " + _autostartCity);
                    return;
                }
                Logger.LogInfo("Autostarting classic game for city: " + cityDef.Id);
                main.StartGame(GameMode.CLASSIC, null, cityDef);
            }
            catch (Exception ex)
            {
                Logger.LogError("Autostart city failed (" + _autostartCity + "): " + ex);
            }
        }
    }
}
