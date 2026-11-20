using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Runtime.CompilerServices;
using System.Text;

namespace MiniMetroRoreaBridge
{
    public static class BridgeProtocol
    {
        public const string Version = "bridge-rpc-v1", DefaultHost = "127.0.0.1";
        public const int DefaultPort = 17842;
        public const string PluginGuid = "com.rorea.minimetro.bridge", PluginName = "MiniMetroRoreaBridge", PluginVersion = "0.1.0";
    }

    public static class BridgeErrorCodes
    {
        public const string INVALID_JSON = "INVALID_JSON", UNSUPPORTED_PROTOCOL = "UNSUPPORTED_PROTOCOL";
        public const string UNKNOWN_COMMAND = "UNKNOWN_COMMAND", INVALID_ARGS = "INVALID_ARGS", INVALID_STATE = "INVALID_STATE";
        public const string INSUFFICIENT_RESOURCES = "INSUFFICIENT_RESOURCES", TUNNEL_BUDGET_EXCEEDED = "TUNNEL_BUDGET_EXCEEDED";
        public const string OUT_OF_MAIN_THREAD_TIMEOUT = "OUT_OF_MAIN_THREAD_TIMEOUT", INTERNAL_ERROR = "INTERNAL_ERROR";
    }

    public static class BridgeScreens
    {
        public const string MENU = "MENU", PLAYING = "PLAYING", PAUSED = "PAUSED";
        public const string WEEKLY_UPGRADE = "WEEKLY_UPGRADE", GAME_OVER = "GAME_OVER", OTHER = "OTHER";
    }

    public static class BridgeCommands
    {
        public const string PING = "PING", GET_STATE = "GET_STATE", SET_PAUSED = "SET_PAUSED", SET_SPEED = "SET_SPEED";
        public const string WAIT = "WAIT", CONNECT = "CONNECT", EXTEND_LINE = "EXTEND_LINE", PREPEND_LINE = "PREPEND_LINE", ADD_TO_LINE = "ADD_TO_LINE";
        public const string INSERT_STATION = "INSERT_STATION", REPLACE_LINE = "REPLACE_LINE", REMOVE_LINE = "REMOVE_LINE";
        public const string ASSIGN_LOCO = "ASSIGN_LOCO", RELOCATE_LOCO = "RELOCATE_LOCO", ATTACH_CARRIAGE = "ATTACH_CARRIAGE";
        public const string PLACE_INTERCHANGE = "PLACE_INTERCHANGE", PICK_UPGRADE = "PICK_UPGRADE";
        private static readonly HashSet<string> ActionOrControl = new HashSet<string>(StringComparer.Ordinal) {
            SET_PAUSED, SET_SPEED, WAIT, CONNECT, EXTEND_LINE, PREPEND_LINE, ADD_TO_LINE, INSERT_STATION,
            REPLACE_LINE, REMOVE_LINE, ASSIGN_LOCO, RELOCATE_LOCO, ATTACH_CARRIAGE, PLACE_INTERCHANGE, PICK_UPGRADE
        };
        public static bool IsKnownActionOrControl(string cmd) => cmd != null && ActionOrControl.Contains(cmd);
        public static bool IsKnownCommand(string cmd) => cmd == PING || cmd == GET_STATE || IsKnownActionOrControl(cmd);
    }

    public sealed class MiniJsonException : FormatException { public MiniJsonException(string msg) : base(msg) { } }

    /// <summary>Strict zero-dependency NDJSON parser and serializer (rejects NaN/Inf/duplicates/trailing data).</summary>
    public static class MiniJson
    {
        private const int MaxDepth = 64;

        public static Dictionary<string, object> ParseObject(string raw)
        {
            if (string.IsNullOrEmpty(raw)) throw new MiniJsonException("Empty JSON line");
            string s = raw.EndsWith("\n", StringComparison.Ordinal)
                ? raw.Substring(0, raw.Length - (raw.EndsWith("\r\n", StringComparison.Ordinal) ? 2 : 1)) : raw;
            if (s.Length == 0 || s.IndexOf('\n') >= 0 || s.IndexOf('\r') >= 0) throw new MiniJsonException("Need 1 NDJSON line");
            int pos = 0;
            if (Peek(s, ref pos) != '{') throw new MiniJsonException("Root JSON must be object");
            Dictionary<string, object> obj = ParseObj(s, ref pos, 0);
            if (Peek(s, ref pos) != '\0') throw new MiniJsonException("Trailing JSON data");
            return obj;
        }

        private static char Peek(string s, ref int pos)
        {
            while (pos < s.Length && (s[pos] == ' ' || s[pos] == '\t')) pos++;
            return pos < s.Length ? s[pos] : '\0';
        }

        private static object ParseValue(string s, ref int pos, int depth)
        {
            if (depth > MaxDepth) throw new MiniJsonException("JSON nesting too deep");
            char c = Peek(s, ref pos);
            return c switch {
                '{' => ParseObj(s, ref pos, depth + 1), '[' => ParseArr(s, ref pos, depth + 1), '"' => ParseStr(s, ref pos),
                't' => Match(s, ref pos, "true", true), 'f' => Match(s, ref pos, "false", false), 'n' => Match(s, ref pos, "null", null),
                _ when c == '-' || (c >= '0' && c <= '9') => ParseNum(s, ref pos),
                _ => throw new MiniJsonException("Unexpected JSON token at " + pos)
            };
        }

        private static object Match(string s, ref int pos, string lit, object val)
        {
            if (pos + lit.Length > s.Length || string.CompareOrdinal(s, pos, lit, 0, lit.Length) != 0)
                throw new MiniJsonException("Invalid literal at " + pos);
            pos += lit.Length; return val;
        }

        private static Dictionary<string, object> ParseObj(string s, ref int pos, int depth)
        {
            pos++; var dict = new Dictionary<string, object>(StringComparer.Ordinal);
            if (Peek(s, ref pos) == '}') { pos++; return dict; }
            for (; ; pos++)
            {
                if (Peek(s, ref pos) != '"') throw new MiniJsonException("Expected key at " + pos);
                string key = ParseStr(s, ref pos);
                if (dict.ContainsKey(key)) throw new MiniJsonException("Duplicate key: " + key);
                if (Peek(s, ref pos) != ':') throw new MiniJsonException("Expected ':' at " + pos);
                pos++; dict[key] = ParseValue(s, ref pos, depth);
                char next = Peek(s, ref pos);
                if (next == '}') { pos++; return dict; }
                if (next != ',') throw new MiniJsonException("Expected ',' or '}' at " + pos);
            }
        }

        private static List<object> ParseArr(string s, ref int pos, int depth)
        {
            pos++; var list = new List<object>();
            if (Peek(s, ref pos) == ']') { pos++; return list; }
            for (; ; pos++)
            {
                list.Add(ParseValue(s, ref pos, depth));
                char next = Peek(s, ref pos);
                if (next == ']') { pos++; return list; }
                if (next != ',') throw new MiniJsonException("Expected ',' or ']' at " + pos);
            }
        }

        private static string ParseStr(string s, ref int pos)
        {
            pos++; var sb = new StringBuilder();
            while (pos < s.Length)
            {
                char c = s[pos++];
                if (c == '"') return sb.ToString();
                if (c < 0x20) throw new MiniJsonException("Unescaped control char");
                if (c != '\\') { sb.Append(c); continue; }
                if (pos >= s.Length) throw new MiniJsonException("Unterminated escape");
                char esc = s[pos++];
                if (esc == 'u')
                {
                    if (pos + 4 > s.Length) throw new MiniJsonException("Invalid \\u escape");
                    int code = 0;
                    for (int i = 0; i < 4; i++)
                    {
                        char x = s[pos++];
                        int hv = x >= '0' && x <= '9' ? x - '0' : x >= 'a' && x <= 'f' ? x - 'a' + 10 : x >= 'A' && x <= 'F' ? x - 'A' + 10 : -1;
                        if (hv < 0) throw new MiniJsonException("Invalid \\u hex");
                        code = (code << 4) | hv;
                    }
                    sb.Append((char)code); continue;
                }
                sb.Append(esc switch { '"' => '"', '\\' => '\\', '/' => '/', 'b' => '\b', 'f' => '\f', 'n' => '\n', 'r' => '\r', 't' => '\t', _ => throw new MiniJsonException("Invalid escape \\" + esc) });
            }
            throw new MiniJsonException("Unterminated string");
        }

        private static void ScanDigits(string s, ref int pos)
        {
            int p0 = pos; while (pos < s.Length && s[pos] >= '0' && s[pos] <= '9') pos++;
            if (pos == p0) throw new MiniJsonException("Expected digit at " + p0);
        }

        private static object ParseNum(string s, ref int pos)
        {
            int start = pos; if (s[pos] == '-') pos++;
            if (pos >= s.Length || s[pos] < '0' || s[pos] > '9') throw new MiniJsonException("Invalid number at " + start);
            if (s[pos] == '0') { pos++; if (pos < s.Length && s[pos] >= '0' && s[pos] <= '9') throw new MiniJsonException("Leading zero"); }
            else ScanDigits(s, ref pos);
            bool isFloat = false;
            if (pos < s.Length && s[pos] == '.') { isFloat = true; pos++; ScanDigits(s, ref pos); }
            if (pos < s.Length && (s[pos] == 'e' || s[pos] == 'E')) { isFloat = true; pos++; if (pos < s.Length && (s[pos] == '+' || s[pos] == '-')) pos++; ScanDigits(s, ref pos); }
            string raw = s.Substring(start, pos - start);
            if (!isFloat && long.TryParse(raw, NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture, out long lv)) return lv;
            if (!double.TryParse(raw, NumberStyles.Float, CultureInfo.InvariantCulture, out double dv) || double.IsNaN(dv) || double.IsInfinity(dv))
                throw new MiniJsonException("Non-finite number: " + raw);
            return dv;
        }

        public static string Serialize(object value) { var sb = new StringBuilder(256); WriteValue(sb, value); return sb.ToString(); }

        private static void WriteValue(StringBuilder sb, object value)
        {
            switch (value)
            {
                case null: sb.Append("null"); return;
                case bool b: sb.Append(b ? "true" : "false"); return;
                case string s: WriteString(sb, s); return;
                case int i: sb.Append(i.ToString(CultureInfo.InvariantCulture)); return;
                case long l: sb.Append(l.ToString(CultureInfo.InvariantCulture)); return;
                case float f:
                    if (float.IsNaN(f) || float.IsInfinity(f)) throw new ArgumentException("Non-finite float");
                    sb.Append(f == 0f ? "0" : f.ToString("R", CultureInfo.InvariantCulture)); return;
                case double d:
                    if (double.IsNaN(d) || double.IsInfinity(d)) throw new ArgumentException("Non-finite double");
                    sb.Append(d == 0.0 ? "0" : d.ToString("R", CultureInfo.InvariantCulture)); return;
                case IDictionary dict:
                    sb.Append('{'); bool first = true;
                    foreach (DictionaryEntry e in dict)
                    {
                        if (!(e.Key is string k)) throw new ArgumentException("JSON keys must be strings");
                        if (!first) sb.Append(','); first = false; WriteString(sb, k); sb.Append(':'); WriteValue(sb, e.Value);
                    }
                    sb.Append('}'); return;
                case IEnumerable seq:
                    sb.Append('['); bool firstItem = true;
                    foreach (object item in seq) { if (!firstItem) sb.Append(','); firstItem = false; WriteValue(sb, item); }
                    sb.Append(']'); return;
                case IFormattable fmt: sb.Append(fmt.ToString(null, CultureInfo.InvariantCulture)); return;
                default: throw new ArgumentException("Unsupported JSON type: " + value.GetType().FullName);
            }
        }

        private static void WriteString(StringBuilder sb, string s)
        {
            sb.Append('"');
            for (int i = 0; i < s.Length; i++)
            {
                char c = s[i];
                string esc = c switch { '"' => "\\\"", '\\' => "\\\\", '\b' => "\\b", '\f' => "\\f", '\n' => "\\n", '\r' => "\\r", '\t' => "\\t", _ => null };
                if (esc != null) sb.Append(esc);
                else if (c < 0x20) { sb.Append("\\u"); sb.Append(((int)c).ToString("x4", CultureInfo.InvariantCulture)); }
                else sb.Append(c);
            }
            sb.Append('"');
        }
    }

    internal static class DtoUtil
    {
        public static Dictionary<string, object> Dict(params object[] kv)
        {
            var d = new Dictionary<string, object>(kv.Length / 2, StringComparer.Ordinal);
            for (int i = 0; i < kv.Length; i += 2) d[(string)kv[i]] = kv[i + 1];
            return d;
        }
        public static List<object> MapList<T>(List<T> items, Func<T, Dictionary<string, object>> map)
        {
            var list = new List<object>(items != null ? items.Count : 0);
            if (items != null) for (int i = 0; i < items.Count; i++) list.Add(map(items[i]));
            return list;
        }
    }

    public sealed class RpcRequestDto
    {
        public string Protocol = BridgeProtocol.Version, Cmd;
        public object Id;
        public Dictionary<string, object> Args = new Dictionary<string, object>(StringComparer.Ordinal);

        public static bool IsValidEntityId(object v) =>
            (v is long l && l >= 0) || (v is int i && i >= 0) || (v is string s && !string.IsNullOrWhiteSpace(s));

        public static bool TryParse(string rawLine, out RpcRequestDto req, out string errorCode, out string errorMsg, out object rawId)
        {
            req = null; rawId = null; errorCode = null; errorMsg = null;
            Dictionary<string, object> obj;
            try { obj = MiniJson.ParseObject(rawLine); }
            catch (MiniJsonException ex) { errorCode = BridgeErrorCodes.INVALID_JSON; errorMsg = ex.Message; return false; }
            if (obj.TryGetValue("id", out object idVal) && IsValidEntityId(idVal)) rawId = idVal;
            if (!obj.TryGetValue("protocol", out object protoObj) || !(protoObj is string proto) || proto != BridgeProtocol.Version)
                return Fail(out errorCode, out errorMsg, BridgeErrorCodes.UNSUPPORTED_PROTOCOL, "Unsupported or missing protocol");
            foreach (string k in obj.Keys)
                if (k != "protocol" && k != "id" && k != "cmd" && k != "args")
                    return Fail(out errorCode, out errorMsg, BridgeErrorCodes.INVALID_ARGS, "Unexpected key: " + k);
            if (rawId == null) return Fail(out errorCode, out errorMsg, BridgeErrorCodes.INVALID_ARGS, "Missing or invalid request id");
            if (!obj.TryGetValue("cmd", out object cmdObj) || !(cmdObj is string cmd) || string.IsNullOrWhiteSpace(cmd))
                return Fail(out errorCode, out errorMsg, BridgeErrorCodes.INVALID_ARGS, "Missing or invalid cmd");
            var args = new Dictionary<string, object>(StringComparer.Ordinal);
            if (obj.TryGetValue("args", out object argsObj))
            {
                if (!(argsObj is Dictionary<string, object> dict))
                    return Fail(out errorCode, out errorMsg, BridgeErrorCodes.INVALID_ARGS, "Request args must be an object");
                args = dict;
            }
            req = new RpcRequestDto { Protocol = proto, Id = rawId, Cmd = cmd, Args = args };
            return true;
        }

        private static bool Fail(out string ec, out string msg, string code, string err) { ec = code; msg = err; return false; }
    }

    public sealed class RpcResponseDto
    {
        public string Protocol = BridgeProtocol.Version, ErrorCode, Error;
        public object Id; public bool Ok; public long Seq; public Dictionary<string, object> Data;
        public static RpcResponseDto Success(object id, long seq, Dictionary<string, object> data = null) =>
            new RpcResponseDto { Id = id, Ok = true, Seq = seq, Data = data };
        public static RpcResponseDto Failure(object id, long seq, string errorCode, string error) => new RpcResponseDto {
            Id = id, Ok = false, Seq = seq,
            ErrorCode = string.IsNullOrWhiteSpace(errorCode) ? BridgeErrorCodes.INTERNAL_ERROR : errorCode,
            Error = string.IsNullOrWhiteSpace(error) ? "Unknown error" : error
        };
        public Dictionary<string, object> ToDictionary() => DtoUtil.Dict(
            "protocol", Protocol, "id", Id, "ok", Ok, "seq", Seq, "error_code", ErrorCode, "error", Error, "data", Data);
        public string ToJson() => MiniJson.Serialize(ToDictionary());
    }

    public sealed class CameraDto
    {
        public double CenterX, CenterY, OrthoSize, WorldToRefScale; public int ScreenWidth, ScreenHeight;
        public Dictionary<string, object> ToDictionary() => DtoUtil.Dict(
            "center_x", CenterX, "center_y", CenterY, "ortho_size", OrthoSize,
            "screen_width", ScreenWidth, "screen_height", ScreenHeight, "world_to_ref_scale", WorldToRefScale);
    }
    public sealed class InventoryDto
    {
        public int LinesAvailable, LinesTotal, LocosAvailable, CarriagesAvailable, TunnelsAvailable, InterchangesAvailable;
        public Dictionary<string, object> ToDictionary() => DtoUtil.Dict(
            "lines_available", LinesAvailable, "lines_total", LinesTotal, "locos_available", LocosAvailable,
            "carriages_available", CarriagesAvailable, "tunnels_available", TunnelsAvailable, "interchanges_available", InterchangesAvailable);
    }
    public sealed class WeeklyOfferDto
    {
        public int Index, Count = 1; public string Kind, Label;
        public Dictionary<string, object> ToDictionary() => DtoUtil.Dict("index", Index, "kind", Kind, "label", Label, "count", Count);
    }
    public sealed class WaitingPassengerDto
    {
        public string DestShape; public int WaitMs;
        public Dictionary<string, object> ToDictionary() => DtoUtil.Dict("dest_shape", DestShape, "wait_ms", WaitMs);
    }
    public sealed class StationDto
    {
        public int Id, PeepCapacity = 6; public double X, Y, ScreenX, ScreenY, OverloadProgress;
        public string Shape; public bool IsInterchange;
        public List<WaitingPassengerDto> WaitingPassengers = new List<WaitingPassengerDto>();
        public Dictionary<string, object> ToDictionary() => DtoUtil.Dict(
            "id", Id, "x", X, "y", Y, "screen_x", ScreenX, "screen_y", ScreenY, "shape", Shape,
            "is_interchange", IsInterchange, "overload_progress", OverloadProgress,
            "waiting_passengers", DtoUtil.MapList(WaitingPassengers, x => x.ToDictionary()), "peep_capacity", PeepCapacity);
    }
    public sealed class LineDto
    {
        public int Id, ColorIndex, TunnelsUsed; public List<int> StationIds = new List<int>(); public bool IsLooped, IsGhost;
        public Dictionary<string, object> ToDictionary() => DtoUtil.Dict(
            "id", Id, "color_index", ColorIndex, "station_ids", StationIds ?? new List<int>(),
            "is_looped", IsLooped, "is_ghost", IsGhost, "tunnels_used", TunnelsUsed);
    }
    public sealed class TrainDto
    {
        public int Id, LineId, Carriages, Capacity, OnboardCount;
        public bool IsUnassignmentQueued, IsGhost, IsWaitingForLocomotive, IsPending; public string LocoType = "LOCOMOTIVE";
        public Dictionary<string, object> ToDictionary() => DtoUtil.Dict(
            "id", Id, "line_id", LineId, "carriages", Carriages, "capacity", Capacity, "onboard_count", OnboardCount,
            "is_unassignment_queued", IsUnassignmentQueued, "is_ghost", IsGhost,
            "is_waiting_for_locomotive", IsWaitingForLocomotive, "loco_type", LocoType, "is_pending", IsPending);
    }
    public sealed class CrossingCostDto
    {
        public int U, V, Tunnels = 1;
        public Dictionary<string, object> ToDictionary() => DtoUtil.Dict("u", U, "v", V, "tunnels", Tunnels);
    }

    public sealed class SnapshotDto
    {
        public string Protocol = BridgeProtocol.Version, Screen = BridgeScreens.MENU, CityId = "none";
        public long Seq, TimestampMs; public bool IsPaused; public double SpeedMultiplier = 1.0; public int Deliveries, Day;
        public CameraDto Camera; public InventoryDto Inventory;
        public List<WeeklyOfferDto> WeeklyOffers = new List<WeeklyOfferDto>();
        public List<StationDto> Stations = new List<StationDto>();
        public List<LineDto> Lines = new List<LineDto>();
        public List<TrainDto> Trains = new List<TrainDto>();
        public List<CrossingCostDto> CrossingCosts = new List<CrossingCostDto>();
        public Dictionary<string, object> ToDictionary() => DtoUtil.Dict(
            "protocol", Protocol, "seq", Seq, "timestamp_ms", TimestampMs, "screen", Screen, "city_id", CityId,
            "is_paused", IsPaused, "speed_multiplier", SpeedMultiplier, "deliveries", Deliveries, "day", Day,
            "camera", Camera?.ToDictionary(), "inventory", Inventory?.ToDictionary(),
            "weekly_offers", DtoUtil.MapList(WeeklyOffers, x => x.ToDictionary()), "stations", DtoUtil.MapList(Stations, x => x.ToDictionary()),
            "lines", DtoUtil.MapList(Lines, x => x.ToDictionary()), "trains", DtoUtil.MapList(Trains, x => x.ToDictionary()),
            "crossing_costs", DtoUtil.MapList(CrossingCosts, x => x.ToDictionary()));
    }

    public interface IBridgeStateReader { SnapshotDto CaptureSnapshot(long seq, LineRegistry lineRegistry); }
    public interface IBridgeActionExecutor { RpcResponseDto Execute(RpcRequestDto req, LineRegistry lineRegistry, IBridgeStateReader reader, long seq); }

    /// <summary>Maps Line references to monotonic line_id values and caches last non-empty active routes.</summary>
    public sealed class LineRegistry
    {
        private sealed class RefEq<T> : IEqualityComparer<T> where T : class
        {
            public static readonly RefEq<T> Instance = new RefEq<T>();
            public bool Equals(T x, T y) => ReferenceEquals(x, y);
            public int GetHashCode(T obj) => obj == null ? 0 : RuntimeHelpers.GetHashCode(obj);
        }
        private sealed class Entry { public int Id; public List<int> CachedStationIds = new List<int>(); }
        private readonly Dictionary<Line, Entry> _entries = new Dictionary<Line, Entry>(RefEq<Line>.Instance);
        private readonly Dictionary<int, Line> _byId = new Dictionary<int, Line>();
        private int _nextLineId = 1;

        public int GetOrAssignId(Line line)
        {
            if (line == null) throw new ArgumentNullException(nameof(line));
            if (_entries.TryGetValue(line, out Entry e)) return e.Id;
            int id = _nextLineId++; _entries[line] = new Entry { Id = id }; _byId[id] = line; return id;
        }
        public bool TryGetId(Line line, out int lineId)
        {
            if (line != null && _entries.TryGetValue(line, out Entry e)) { lineId = e.Id; return true; }
            lineId = 0; return false;
        }
        public void UpdateCachedRoute(Line line, IList<int> stationIds)
        {
            if (line == null || stationIds == null || stationIds.Count == 0) return;
            GetOrAssignId(line); _entries[line].CachedStationIds = new List<int>(stationIds);
        }
        public List<int> GetCachedRoute(Line line) =>
            line != null && _entries.TryGetValue(line, out Entry e) ? new List<int>(e.CachedStationIds) : new List<int>();
        public List<int> ExtractOrGetCachedRoute(Line line)
        {
            if (line == null) return new List<int>();
            GetOrAssignId(line);
            if (!line.IsMothballed && line.FirstActiveLink != null)
            {
                var route = new List<int>(); var visited = new HashSet<Link>(RefEq<Link>.Instance);
                Link first = line.FirstActiveLink, cur = first;
                if (cur.Start?.Station != null) route.Add(cur.Start.Station.Id);
                while (cur != null && visited.Add(cur))
                {
                    if (cur.End?.Station != null)
                    {
                        int endId = cur.End.Station.Id;
                        if (!(cur.NextLink == first && route.Count > 0 && route[0] == endId)) route.Add(endId);
                    }
                    if (cur.NextLink == first) break; cur = cur.NextLink;
                }
                if (route.Count > 0) { _entries[line].CachedStationIds = new List<int>(route); return route; }
            }
            return GetCachedRoute(line);
        }
        public Line ResolveLine(City city, int lineId)
        {
            if (city == null || !_byId.TryGetValue(lineId, out Line c) || c == null) return null;
            for (int i = 0; i < city.LineCount; i++) if (ReferenceEquals(city.GetLine(i), c)) return c;
            return null;
        }
        public Line ResolveActiveLine(City city, int lineId) { Line ln = ResolveLine(city, lineId); return ln != null && !ln.IsMothballed ? ln : null; }
        public void Prune(City city)
        {
            if (city == null) { _entries.Clear(); _byId.Clear(); return; }
            var live = new HashSet<Line>(RefEq<Line>.Instance);
            for (int i = 0; i < city.LineCount; i++) { Line ln = city.GetLine(i); if (ln != null) live.Add(ln); }
            var dead = new List<Line>();
            foreach (Line key in _entries.Keys) if (!live.Contains(key)) dead.Add(key);
            for (int i = 0; i < dead.Count; i++) { _byId.Remove(_entries[dead[i]].Id); _entries.Remove(dead[i]); }
        }
    }
}
