"""Synchronous TCP NDJSON client for the ``bridge-rpc-v1`` Mini Metro bridge."""

from __future__ import annotations

import socket
from collections.abc import Mapping
from typing import Any, Self

from rl.bridge_protocol import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    EntityId,
    ErrorCode,
    OpenAction,
    ProtocolValidationError,
    RpcRequest,
    RpcResponse,
    Snapshot,
)


class BridgeRpcError(RuntimeError):
    """Raised when the RPC server returns ``ok=False`` or a transport/protocol error occurs."""

    def __init__(
        self,
        message: str,
        error_code: ErrorCode | None = None,
        request_id: EntityId | None = None,
        response: RpcResponse | None = None,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.request_id = request_id
        self.response = response


class BridgeRpcClient:
    """Persistent TCP NDJSON client for ``bridge-rpc-v1``."""

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        timeout_s: float = 5.0,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.timeout_s = float(timeout_s)
        self._sock: socket.socket | None = None
        self._rfile: Any = None
        self._next_id: int = 1

    def connect(self) -> Self:
        """Open the persistent TCP connection if not already connected."""
        if self._sock is None:
            sock = socket.create_connection(
                (self.host, self.port), timeout=self.timeout_s
            )
            sock.settimeout(self.timeout_s)
            self._sock = sock
            self._rfile = sock.makefile("rb")
        return self

    def close(self) -> None:
        """Close the buffered reader and underlying TCP socket."""
        rfile, sock = self._rfile, self._sock
        self._rfile = None
        self._sock = None
        if rfile is not None:
            try:
                rfile.close()
            except OSError:
                pass
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

    def __enter__(self) -> Self:
        return self.connect()

    def __exit__(self, *_: Any) -> None:
        self.close()

    def _alloc_request_id(self) -> int:
        rid = self._next_id
        self._next_id += 1
        return rid

    def send_request(self, req: RpcRequest) -> RpcResponse:
        """Send one ``RpcRequest`` line and decode the matching ``RpcResponse``."""
        self.connect()
        assert self._sock is not None and self._rfile is not None
        payload = (req.to_json() + "\n").encode("utf-8")
        try:
            self._sock.sendall(payload)
            raw_line: bytes = self._rfile.readline()
        except OSError as exc:
            self.close()
            raise BridgeRpcError(
                f"RPC socket error: {exc}",
                error_code=None,
                request_id=req.id,
            ) from exc

        if not raw_line:
            self.close()
            raise BridgeRpcError(
                "Connection closed by server before response",
                error_code=None,
                request_id=req.id,
            )

        try:
            text = raw_line.decode("utf-8")
            resp = RpcResponse.from_json(text)
        except UnicodeDecodeError as exc:
            raise BridgeRpcError(
                f"Invalid UTF-8 in RPC response: {exc}",
                error_code=ErrorCode.INVALID_JSON,
                request_id=req.id,
            ) from exc
        except ProtocolValidationError as exc:
            raise BridgeRpcError(
                str(exc),
                error_code=exc.error_code,
                request_id=req.id,
            ) from exc

        if not resp.ok:
            raise BridgeRpcError(
                resp.error or "RPC request failed",
                error_code=resp.error_code,
                request_id=req.id,
                response=resp,
            )

        if resp.id != req.id:
            raise BridgeRpcError(
                f"Mismatched response id: expected {req.id!r}, got {resp.id!r}",
                error_code=ErrorCode.INVALID_JSON,
                request_id=req.id,
                response=resp,
            )

        return resp

    def _extract_snapshot(self, resp: RpcResponse, req_id: EntityId) -> Snapshot:
        if not isinstance(resp.data, Mapping) or "snapshot" not in resp.data:
            raise BridgeRpcError(
                "RPC response missing 'snapshot' payload",
                error_code=ErrorCode.INVALID_JSON,
                request_id=req_id,
                response=resp,
            )
        try:
            return Snapshot.from_dict(resp.data["snapshot"])
        except ProtocolValidationError as exc:
            raise BridgeRpcError(
                str(exc),
                error_code=exc.error_code,
                request_id=req_id,
                response=resp,
            ) from exc

    def ping(self) -> dict[str, Any]:
        """Send ``PING`` and return the response ``data`` dictionary."""
        req = RpcRequest.ping(self._alloc_request_id())
        resp = self.send_request(req)
        return dict(resp.data or {})

    def get_state(self) -> Snapshot:
        """Fetch and validate the current ``Snapshot`` via ``GET_STATE``."""
        req = RpcRequest.get_state(self._alloc_request_id())
        resp = self.send_request(req)
        return self._extract_snapshot(resp, req.id)

    def set_paused(self, paused: bool) -> Snapshot:
        """Pause or unpause the game and return the updated ``Snapshot``."""
        req = RpcRequest.set_paused(paused, self._alloc_request_id())
        resp = self.send_request(req)
        return self._extract_snapshot(resp, req.id)

    def set_speed(self, speed_multiplier: float) -> Snapshot:
        """Set simulation speed multiplier (1.0 or 2.0) and return the ``Snapshot``."""
        req = RpcRequest.set_speed(speed_multiplier, self._alloc_request_id())
        resp = self.send_request(req)
        return self._extract_snapshot(resp, req.id)

    def execute_action(self, action: OpenAction) -> Snapshot:
        """Execute a validated ``OpenAction`` and return the updated ``Snapshot``."""
        req = RpcRequest.from_action(action, self._alloc_request_id())
        resp = self.send_request(req)
        return self._extract_snapshot(resp, req.id)
