using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;

namespace MiniMetroRoreaBridge
{
    /// <summary>Thread-safe request work item queued for Unity main-thread execution.</summary>
    public sealed class PendingRpcWork
    {
        private readonly ManualResetEventSlim _done = new ManualResetEventSlim(false);
        private RpcResponseDto _response;
        private bool _abandoned;

        public PendingRpcWork(RpcRequestDto request) => Request = request ?? throw new ArgumentNullException(nameof(request));
        public RpcRequestDto Request { get; }
        public bool IsAbandoned { get { lock (this) return _abandoned; } }

        public void Complete(RpcResponseDto response)
        {
            lock (this)
            {
                if (_abandoned) return;
                _response = response;
                _done.Set();
            }
        }

        public bool Wait(int timeoutMs, CancellationToken ct, out RpcResponseDto response)
        {
            try { _done.Wait(timeoutMs, ct); }
            catch (OperationCanceledException) { }
            lock (this)
            {
                if (_response != null) { response = _response; _done.Dispose(); return true; }
                _abandoned = true;
                _done.Dispose();
                response = null;
                return false;
            }
        }
    }

    /// <summary>Raw-Socket TCP NDJSON RPC server marshaling requests onto the Unity main thread.</summary>
    public sealed class RpcServer : IDisposable
    {
        private const int MaxLineBytes = 4 * 1024 * 1024;
        private static readonly UTF8Encoding StrictUtf8 = new UTF8Encoding(false, true);

        private readonly int _configuredPort, _mainThreadTimeoutMs;
        private readonly Func<long> _getSeq;
        private readonly Action<string> _logError;
        private readonly ConcurrentQueue<PendingRpcWork> _queue = new ConcurrentQueue<PendingRpcWork>();
        private readonly List<Socket> _clients = new List<Socket>();
        private readonly object _stateLock = new object();

        private Socket _listenSocket;
        private Thread _acceptThread;
        private CancellationTokenSource _cts;
        private volatile bool _running;

        public RpcServer(
            int port = BridgeProtocol.DefaultPort, int mainThreadTimeoutMs = 5000,
            Func<long> getSeq = null, Action<string> logInfo = null, Action<string> logError = null)
        {
            if (port < 0 || port > 65535) throw new ArgumentOutOfRangeException(nameof(port));
            if (mainThreadTimeoutMs <= 0) throw new ArgumentOutOfRangeException(nameof(mainThreadTimeoutMs));
            _configuredPort = port;
            _mainThreadTimeoutMs = mainThreadTimeoutMs;
            _getSeq = getSeq ?? (() => 0L);
            _logError = logError;
        }

        public int Port { get; private set; }
        public bool IsRunning => _running;
        public bool TryDequeue(out PendingRpcWork work) => _queue.TryDequeue(out work);
        public void Dispose() => Stop();

        public void Start()
        {
            lock (_stateLock)
            {
                if (_running) return;
                _cts = new CancellationTokenSource();
                _listenSocket = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
                _listenSocket.SetSocketOption(SocketOptionLevel.Socket, SocketOptionName.ReuseAddress, true);
                _listenSocket.Bind(new IPEndPoint(IPAddress.Loopback, _configuredPort));
                _listenSocket.Listen(16);
                Port = ((IPEndPoint)_listenSocket.LocalEndPoint).Port;
                _running = true;
                _acceptThread = new Thread(AcceptLoop) { IsBackground = true, Name = "MiniMetroBridgeAccept" };
                _acceptThread.Start();
            }
        }

        public void Stop()
        {
            Thread acceptThread;
            CancellationTokenSource cts;
            lock (_stateLock)
            {
                if (!_running) return;
                _running = false;
                cts = _cts; _cts = null;
                acceptThread = _acceptThread; _acceptThread = null;
                try { cts?.Cancel(); } catch (Exception) { }
                CloseSocketQuietly(_listenSocket);
                _listenSocket = null;
                lock (_clients)
                {
                    for (int i = 0; i < _clients.Count; i++) CloseSocketQuietly(_clients[i]);
                    _clients.Clear();
                }
            }
            if (acceptThread != null && acceptThread.IsAlive) acceptThread.Join(1000);
            try { cts?.Dispose(); } catch (Exception) { }
        }

        private void AcceptLoop()
        {
            while (_running)
            {
                Socket client;
                try
                {
                    Socket listener = _listenSocket;
                    if (listener == null) break;
                    client = listener.Accept();
                }
                catch (SocketException ex) { if (_running) _logError?.Invoke("Bridge accept error: " + ex.Message); break; }
                catch (ObjectDisposedException) { break; }
                client.NoDelay = true;
                lock (_clients) _clients.Add(client);
                new Thread(() => HandleClient(client)) { IsBackground = true, Name = "MiniMetroBridgeClient" }.Start();
            }
        }

        private void HandleClient(Socket client)
        {
            byte[] recvBuf = new byte[4096];
            var lineBuf = new List<byte>(512);
            try
            {
                while (_running)
                {
                    int n = client.Receive(recvBuf, 0, recvBuf.Length, SocketFlags.None);
                    if (n <= 0) break;
                    for (int i = 0; i < n; i++)
                    {
                        byte b = recvBuf[i];
                        if (b == (byte)'\n')
                        {
                            if (lineBuf.Count > 0 && lineBuf[lineBuf.Count - 1] == (byte)'\r')
                                lineBuf.RemoveAt(lineBuf.Count - 1);
                            if (!ProcessLine(client, lineBuf.ToArray())) return;
                            lineBuf.Clear();
                        }
                        else
                        {
                            if (lineBuf.Count >= MaxLineBytes)
                            {
                                SendResponse(client, RpcResponseDto.Failure(null, _getSeq(), BridgeErrorCodes.INVALID_JSON, "Request line exceeds limit"));
                                return;
                            }
                            lineBuf.Add(b);
                        }
                    }
                }
            }
            catch (SocketException) { }
            catch (ObjectDisposedException) { }
            catch (IOException) { }
            catch (Exception ex) { if (_running) _logError?.Invoke("Bridge client loop error: " + ex); }
            finally { lock (_clients) _clients.Remove(client); CloseSocketQuietly(client); }
        }

        private bool ProcessLine(Socket client, byte[] lineBytes)
        {
            string line;
            try { line = StrictUtf8.GetString(lineBytes); }
            catch (DecoderFallbackException)
            {
                return SendResponse(client, RpcResponseDto.Failure(null, _getSeq(), BridgeErrorCodes.INVALID_JSON, "Invalid UTF-8 in request"));
            }
            if (!RpcRequestDto.TryParse(line, out RpcRequestDto req, out string errCode, out string errMsg, out object rawId))
                return SendResponse(client, RpcResponseDto.Failure(rawId, _getSeq(), errCode, errMsg));

            CancellationTokenSource cts = _cts;
            if (!_running || cts == null) return false;
            var work = new PendingRpcWork(req);
            _queue.Enqueue(work);
            if (!work.Wait(_mainThreadTimeoutMs, cts.Token, out RpcResponseDto resp))
            {
                if (!_running) return false;
                resp = RpcResponseDto.Failure(req.Id, _getSeq(), BridgeErrorCodes.OUT_OF_MAIN_THREAD_TIMEOUT, "Timed out waiting for Unity main thread");
            }
            return SendResponse(client, resp);
        }

        private bool SendResponse(Socket client, RpcResponseDto resp)
        {
            byte[] bytes = StrictUtf8.GetBytes(resp.ToJson() + "\n");
            int sent = 0;
            while (sent < bytes.Length)
            {
                int n = client.Send(bytes, sent, bytes.Length - sent, SocketFlags.None);
                if (n <= 0) return false;
                sent += n;
            }
            return true;
        }

        private static void CloseSocketQuietly(Socket socket)
        {
            if (socket == null) return;
            try { socket.Shutdown(SocketShutdown.Both); } catch (Exception) { }
            try { socket.Close(); } catch (Exception) { }
        }
    }
}
