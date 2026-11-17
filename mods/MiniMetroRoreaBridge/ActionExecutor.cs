using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Reflection;
using Metro;
using UI;

namespace MiniMetroRoreaBridge
{
    public sealed class ActionExecutor : IBridgeActionExecutor
    {
        private static readonly BindingFlags Flags = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;
        private static readonly FieldInfo MainCtrlField = typeof(Main).GetField("controller", Flags);
        private static readonly FieldInfo GameCtrlGameField = typeof(GameController).GetField("game", Flags);
        private static readonly FieldInfo GameScreensField = typeof(Game).GetField("screens", Flags);
        private static readonly FieldInfo AssetPanelField = typeof(NewAssetScreen).GetField("newAssetPanel", Flags);
        private static readonly FieldInfo AssetButtonsField = typeof(NewAssetPanel).GetField("buttons", Flags);

        public RpcResponseDto Execute(RpcRequestDto req, LineRegistry lineRegistry, IBridgeStateReader reader, long seq)
        {
            if (req == null || string.IsNullOrWhiteSpace(req.Cmd)) return Fail(req?.Id, seq, BridgeErrorCodes.INVALID_ARGS, "Missing request command");
            if (lineRegistry == null) return Fail(req.Id, seq, BridgeErrorCodes.INTERNAL_ERROR, "LineRegistry is null");
            Game game = ResolveActiveGame();
            if (Main.Instance == null || !Main.Instance.IsInGame || game == null || game.City == null) return Fail(req.Id, seq, BridgeErrorCodes.INVALID_STATE, "No active in-game city session");
            if (game.IsOver || game.OvercrowdedStation != null || game.Screen == GameScreen.GameOver) return Fail(req.Id, seq, BridgeErrorCodes.INVALID_STATE, "Game is over");

            City city = game.City;
            var args = req.Args ?? new Dictionary<string, object>(StringComparer.Ordinal);
            string cmd = req.Cmd;
            if (cmd == BridgeCommands.WAIT)
            {
                RpcResponseDto waitErr = ExecuteWait(args, req.Id, seq);
                return waitErr ?? CaptureSuccess(req.Id, seq, reader, lineRegistry);
            }
            if (cmd == BridgeCommands.PICK_UPGRADE) return ExecutePickUpgrade(game, args, req.Id, seq) ?? CaptureSuccess(req.Id, seq, reader, lineRegistry);
            if (game.IsScreenChangeScheduled || game.IsLocked) return Fail(req.Id, seq, BridgeErrorCodes.INVALID_STATE, "Game is locked or transitioning screens");
            if (game.Screen != GameScreen.None) return Fail(req.Id, seq, BridgeErrorCodes.INVALID_STATE, "Command requires live board state (GameScreen.None)");

            RpcResponseDto err = cmd switch {
                BridgeCommands.SET_PAUSED => ExecuteSetPaused(game, args, req.Id, seq),
                BridgeCommands.SET_SPEED => ExecuteSetSpeed(game, args, req.Id, seq),
                BridgeCommands.WAIT => ExecuteWait(args, req.Id, seq),
                BridgeCommands.CONNECT => ExecuteConnectCmd(game, city, lineRegistry, args, req.Id, seq),
                BridgeCommands.EXTEND_LINE => ExecuteExtendOrPrependCmd(game, city, lineRegistry, args, LineDirection.FORWARDS, req.Id, seq),
                BridgeCommands.PREPEND_LINE => ExecuteExtendOrPrependCmd(game, city, lineRegistry, args, LineDirection.BACKWARDS, req.Id, seq),
                BridgeCommands.INSERT_STATION => ExecuteInsertStationCmd(game, city, lineRegistry, args, req.Id, seq),
                BridgeCommands.REPLACE_LINE => ExecuteReplaceLineCmd(game, city, lineRegistry, args, req.Id, seq),
                BridgeCommands.REMOVE_LINE => ExecuteRemoveLineCmd(game, city, lineRegistry, args, req.Id, seq),
                BridgeCommands.ASSIGN_LOCO => ExecuteAssignLocoCmd(game, city, lineRegistry, args, req.Id, seq),
                BridgeCommands.RELOCATE_LOCO => ExecuteRelocateLocoCmd(game, city, lineRegistry, args, req.Id, seq),
                BridgeCommands.ATTACH_CARRIAGE => ExecuteAttachCarriageCmd(game, city, lineRegistry, args, req.Id, seq),
                BridgeCommands.PLACE_INTERCHANGE => ExecutePlaceInterchangeCmd(game, city, args, req.Id, seq),
                _ => Fail(req.Id, seq, BridgeErrorCodes.UNKNOWN_COMMAND, "Unknown command: " + cmd)
            };
            return err ?? CaptureSuccess(req.Id, seq, reader, lineRegistry);
        }

        private static Game ResolveActiveGame() => (Main.Instance != null && MainCtrlField?.GetValue(Main.Instance) is GameController gc && GameCtrlGameField?.GetValue(gc) is Game g) ? g : DevStationView.Instance?.City?.Game;

        private static RpcResponseDto CaptureSuccess(object id, long seq, IBridgeStateReader reader, LineRegistry lineRegistry) =>
            RpcResponseDto.Success(id, seq, new Dictionary<string, object>(StringComparer.Ordinal) { { "snapshot", reader?.CaptureSnapshot(seq, lineRegistry)?.ToDictionary() } });

        private static RpcResponseDto Fail(object id, long seq, string code, string msg) => RpcResponseDto.Failure(id, seq, code, msg);

        private static RpcResponseDto ExecuteSetPaused(Game game, Dictionary<string, object> args, object id, long seq)
        {
            if (!CheckAllowedKeys(args, "paused") || !args.TryGetValue("paused", out object v) || !(v is bool paused)) return Fail(id, seq, BridgeErrorCodes.INVALID_ARGS, "SET_PAUSED requires boolean 'paused'");
            ClockWidget cw = game.HudScreen?.ClockWidget;
            if (cw == null) return Fail(id, seq, BridgeErrorCodes.INVALID_STATE, "ClockWidget unavailable");
            if (game.IsPaused != paused) cw.TriggerVCRButtonPress(paused ? VCRButtonType.PAUSE : (game.ScheduledTimeScale == TimeScale.Double ? VCRButtonType.FAST_FORWARD : VCRButtonType.PLAY));
            return null;
        }

        private static RpcResponseDto ExecuteSetSpeed(Game game, Dictionary<string, object> args, object id, long seq)
        {
            if (!CheckAllowedKeys(args, "speed_multiplier") || !args.TryGetValue("speed_multiplier", out object v) || !TryGetDouble(v, out double speed) || (speed != 1.0 && speed != 2.0)) return Fail(id, seq, BridgeErrorCodes.INVALID_ARGS, "SET_SPEED requires 'speed_multiplier' in {1.0, 2.0}");
            ClockWidget cw = game.HudScreen?.ClockWidget;
            if (cw == null) return Fail(id, seq, BridgeErrorCodes.INVALID_STATE, "ClockWidget unavailable");
            cw.TriggerVCRButtonPress(speed == 2.0 ? VCRButtonType.FAST_FORWARD : VCRButtonType.PLAY); return null;
        }

        private static RpcResponseDto ExecuteWait(Dictionary<string, object> args, object id, long seq) =>
            !CheckAllowedKeys(args, "ticks") ? Fail(id, seq, BridgeErrorCodes.INVALID_ARGS, "Unexpected args for WAIT")
            : (args.TryGetValue("ticks", out object tv) && tv != null && (!TryGetInt(tv, out int ticks) || ticks < 1))
                ? Fail(id, seq, BridgeErrorCodes.INVALID_ARGS, "WAIT 'ticks' must be an integer >= 1") : null;

        private static RpcResponseDto ExecutePickUpgrade(Game game, Dictionary<string, object> args, object id, long seq)
        {
            if (!CheckAllowedKeys(args, "offer_index") || !args.TryGetValue("offer_index", out object ov) || !TryGetInt(ov, out int offerIndex) || offerIndex < 0) return Fail(id, seq, BridgeErrorCodes.INVALID_ARGS, "PICK_UPGRADE requires integer 'offer_index' >= 0");
            if (game.Screen != GameScreen.NewAsset) return Fail(id, seq, BridgeErrorCodes.INVALID_STATE, "PICK_UPGRADE requires Weekly Upgrade screen (GameScreen.NewAsset)");
            Screen[] screens = GameScreensField?.GetValue(game) as Screen[];
            NewAssetScreen assetScreen = (screens != null && (int)GameScreen.NewAsset < screens.Length) ? screens[(int)GameScreen.NewAsset] as NewAssetScreen : null;
            NewAssetPanel panel = assetScreen != null ? AssetPanelField?.GetValue(assetScreen) as NewAssetPanel : null;
            var buttons = panel != null ? AssetButtonsField?.GetValue(panel) as List<NewAssetButton> : null;
            if (assetScreen == null || buttons == null) return Fail(id, seq, BridgeErrorCodes.INVALID_STATE, "NewAssetScreen buttons are unavailable");
            if (offerIndex >= buttons.Count || buttons[offerIndex] == null || !buttons[offerIndex].IsEnabled) return Fail(id, seq, BridgeErrorCodes.INVALID_ARGS, "Invalid or disabled 'offer_index': " + offerIndex);
            assetScreen.OnAsset(buttons[offerIndex]);
            return null;
        }

        private static RpcResponseDto ExecuteConnectCmd(Game game, City city, LineRegistry lineRegistry, Dictionary<string, object> args, object id, long seq)
        {
            if (!CheckAllowedKeys(args, "station_ids", "is_looped", "loop", "color_index") || !TryGetLoopedFlag(args, out bool isLooped)) return Fail(id, seq, BridgeErrorCodes.INVALID_ARGS, "Invalid CONNECT arguments");
            int? colorIndex = null;
            if (args.TryGetValue("color_index", out object cv) && cv != null) { if (!TryGetInt(cv, out int ci) || ci < 0) return Fail(id, seq, BridgeErrorCodes.INVALID_ARGS, "Invalid 'color_index'"); colorIndex = ci; }
            if (!TryResolveStations(city, args, isLooped, out List<Station> stations, out string stErr)) return Fail(id, seq, BridgeErrorCodes.INVALID_ARGS, stErr);
            return ExecuteConnect(game, city, lineRegistry, stations, isLooped, colorIndex, id, seq);
        }

        private static RpcResponseDto ExecuteConnect(Game game, City city, LineRegistry lineRegistry, List<Station> stations, bool isLooped, int? colorIndex, object id, long seq)
        {
            if (game.AssetDatabase.GetAvailableAssets(AssetType.Line) <= 0) return Fail(id, seq, BridgeErrorCodes.INSUFFICIENT_RESOURCES, "No available lines in inventory");
            int slotIndex = (colorIndex.HasValue && colorIndex.Value >= 0 && colorIndex.Value < city.MaxLineCount && city.GetLineByIndex(colorIndex.Value) == null) ? colorIndex.Value : city.GetNextFreeLineIndex();
            if (slotIndex < 0 || slotIndex >= city.MaxLineCount) return Fail(id, seq, BridgeErrorCodes.INSUFFICIENT_RESOURCES, "No free line slot available");

            Line line = city.AddLine(stations[0], slotIndex);
            if (line == null || line.Index == -1) { line?.Remove(); return Fail(id, seq, BridgeErrorCodes.INVALID_STATE, "City.AddLine failed"); }
            line[0].End.Station = stations[1];
            if (line[0].End.Station != stations[1])
            {
                bool noTunnel = LacksTunnelForSegment(game, city, stations[0], stations[1]);
                line[0].Remove(); line.Remove(); game.OnLineEditComplete();
                return Fail(id, seq, noTunnel ? BridgeErrorCodes.TUNNEL_BUDGET_EXCEEDED : BridgeErrorCodes.INVALID_STATE, "Failed to connect initial link");
            }
            line[0].RemoveAllErrors();
            var createdLinks = new List<Link>(stations.Count) { line[0] };
            for (int i = 2; i < stations.Count; i++)
            {
                Link prev = createdLinks[createdLinks.Count - 1];
                line.RemoveTerminator(prev.End.Station);
                Link next = line.AppendLink(prev, LineDirection.FORWARDS, connectLink: true);
                next.End.Station = stations[i];
                if (next.End.Station != stations[i])
                {
                    bool noTunnel = LacksTunnelForSegment(game, city, stations[i - 1], stations[i]);
                    next.Remove(); RollbackNewLine(game, line, createdLinks);
                    return Fail(id, seq, noTunnel ? BridgeErrorCodes.TUNNEL_BUDGET_EXCEEDED : BridgeErrorCodes.INVALID_STATE, "Failed to connect station at index " + i);
                }
                next.RemoveAllErrors(); createdLinks.Add(next);
            }
            Link closingLink = null;
            if (isLooped)
            {
                Link last = createdLinks[createdLinks.Count - 1];
                line.RemoveTerminator(last.End.Station);
                closingLink = line.AppendLink(last, LineDirection.FORWARDS, connectLink: true);
                closingLink.End.Station = stations[0];
                if (closingLink.End.Station != stations[0])
                {
                    bool noTunnel = LacksTunnelForSegment(game, city, stations[stations.Count - 1], stations[0]);
                    closingLink.Remove(); RollbackNewLine(game, line, createdLinks);
                    return Fail(id, seq, noTunnel ? BridgeErrorCodes.TUNNEL_BUDGET_EXCEEDED : BridgeErrorCodes.INVALID_STATE, "Failed to close loop");
                }
                closingLink.RemoveAllErrors();
            }
            line.Start(LineDirection.FORWARDS, closingLink, startTrain: true, randomizeTrainPosition: false, tween: true);
            AssetType loco = ResolveAvailableLocoForLine(game, city, line.Index);
            if (loco != AssetType.None) line.ApplyAsset(loco, null, LineDirection.FORWARDS, null, scheduleAudioEvent: true, duringSetup: false, tween: true);
            line.DirtyTracks(); game.OnLineEditComplete(); lineRegistry.ExtractOrGetCachedRoute(line);
            return null;
        }

        private static void RollbackNewLine(Game game, Line line, List<Link> createdLinks)
        {
            for (int i = createdLinks.Count - 1; i >= 0; i--) createdLinks[i].Remove();
            line.Remove(); game.OnLineEditComplete();
        }

        private static RpcResponseDto ExecuteExtendOrPrependCmd(Game game, City city, LineRegistry lineRegistry, Dictionary<string, object> args, LineDirection dir, object id, long seq)
        {
            if (!CheckAllowedKeys(args, "line_id", "station_id", "is_looped", "loop") || !TryGetLoopedFlag(args, out bool isLooped) || !TryGetEntityId(args, "line_id", out int lineId) || !TryGetEntityId(args, "station_id", out int stationId))
                return Fail(id, seq, BridgeErrorCodes.INVALID_ARGS, "Invalid arguments for line endpoint extension");
            if (!TryResolveActiveLine(city, lineRegistry, lineId, out Line line, out string lineCode, out string lineErr)) return Fail(id, seq, lineCode, lineErr);
            Station targetStation = city.GetStationById(stationId);
            if (targetStation == null || !targetStation.IsActive || !targetStation.CanBeConnected) return Fail(id, seq, BridgeErrorCodes.INVALID_ARGS, "Target station is invalid or inactive: " + stationId);
            return ExecuteExtendOrPrepend(game, city, lineRegistry, line, targetStation, dir, isLooped, id, seq);
        }

        private static RpcResponseDto ExecuteExtendOrPrepend(Game game, City city, LineRegistry lineRegistry, Line line, Station targetStation, LineDirection dir, bool isLooped, object id, long seq)
        {
            if (line.IsLooping) return Fail(id, seq, BridgeErrorCodes.INVALID_STATE, "Cannot extend or prepend an already looped line");
            Link endpointLink = dir == LineDirection.FORWARDS ? line.LastActiveLink : line.FirstActiveLink;
            Link oppositeLink = dir == LineDirection.FORWARDS ? line.FirstActiveLink : line.LastActiveLink;
            Station endpointStation = dir == LineDirection.FORWARDS ? endpointLink?.End?.Station : endpointLink?.Start?.Station;
            Station oppositeStation = dir == LineDirection.FORWARDS ? oppositeLink?.Start?.Station : oppositeLink?.End?.Station;
            if (endpointLink == null || endpointStation == null || oppositeStation == null) return Fail(id, seq, BridgeErrorCodes.INVALID_STATE, "Line has no valid active endpoint link");
            bool directLoopClose = targetStation == oppositeStation;
            if (line.ContainsStation(targetStation) && !directLoopClose) return Fail(id, seq, BridgeErrorCodes.INVALID_ARGS, "Line already contains target station");
            if (directLoopClose && line.LiveLinkCount < 2) return Fail(id, seq, BridgeErrorCodes.INVALID_ARGS, "Loop requires at least 3 distinct stations");

            line.RemoveTerminator(endpointStation);
            Link newLink = line.AppendLink(endpointLink, dir, connectLink: true);
            LinkStop openStop = dir == LineDirection.FORWARDS ? newLink.End : newLink.Start;
            openStop.Station = targetStation;
            if (openStop.Station != targetStation)
            {
                bool noTunnel = LacksTunnelForSegment(game, city, endpointStation, targetStation);
                newLink.Remove(); line.GenerateTerminators(); line.DirtyTracks(); game.OnLineEditComplete();
                return Fail(id, seq, noTunnel ? BridgeErrorCodes.TUNNEL_BUDGET_EXCEEDED : BridgeErrorCodes.INVALID_STATE, "Failed to extend line to station " + targetStation.Id);
            }
            newLink.RemoveAllErrors();
            Link endLoopLink = directLoopClose ? newLink : null;
            if (!directLoopClose && isLooped)
            {
                line.RemoveTerminator(targetStation);
                Link closeLink = line.AppendLink(newLink, dir, connectLink: true);
                LinkStop closeStop = dir == LineDirection.FORWARDS ? closeLink.End : closeLink.Start;
                closeStop.Station = oppositeStation;
                if (closeStop.Station != oppositeStation)
                {
                    bool noTunnel = LacksTunnelForSegment(game, city, targetStation, oppositeStation);
                    closeLink.Remove(); newLink.Remove(); line.GenerateTerminators(); line.DirtyTracks(); game.OnLineEditComplete();
                    return Fail(id, seq, noTunnel ? BridgeErrorCodes.TUNNEL_BUDGET_EXCEEDED : BridgeErrorCodes.INVALID_STATE, "Failed to close loop after extension");
                }
                closeLink.RemoveAllErrors(); endLoopLink = closeLink;
            }
            line.Start(dir, endLoopLink, startTrain: false, randomizeTrainPosition: false, tween: true);
            line.ScheduleAudioEvent(LineChangeType.StationAdded, targetStation);
            line.DirtyTracks(); game.OnLineEditComplete(); lineRegistry.ExtractOrGetCachedRoute(line);
            return null;
        }

        private static RpcResponseDto ExecuteInsertStationCmd(Game game, City city, LineRegistry lineRegistry, Dictionary<string, object> args, object id, long seq)
        {
            if (!CheckAllowedKeys(args, "line_id", "station_id", "after_station_id") || !TryGetEntityId(args, "line_id", out int lineId) || !TryGetEntityId(args, "station_id", out int stationId) || !TryGetEntityId(args, "after_station_id", out int afterStationId) || stationId == afterStationId)
                return Fail(id, seq, BridgeErrorCodes.INVALID_ARGS, "Invalid INSERT_STATION arguments");
            if (!game.CanRemoveTracks) return Fail(id, seq, BridgeErrorCodes.INVALID_STATE, "Track modification is disabled in Extreme mode");
            if (!TryResolveActiveLine(city, lineRegistry, lineId, out Line line, out string lineCode, out string lineErr)) return Fail(id, seq, lineCode, lineErr);
            Station midStation = city.GetStationById(stationId), afterStation = city.GetStationById(afterStationId);
            if (midStation == null || !midStation.IsActive || !midStation.CanBeConnected || afterStation == null || !afterStation.IsActive) return Fail(id, seq, BridgeErrorCodes.INVALID_ARGS, "Station is invalid or inactive");
            return ExecuteInsertStation(game, city, lineRegistry, line, midStation, afterStationId, id, seq);
        }

        private static RpcResponseDto ExecuteInsertStation(Game game, City city, LineRegistry lineRegistry, Line line, Station midStation, int afterStationId, object id, long seq)
        {
            if (line.ContainsStation(midStation)) return Fail(id, seq, BridgeErrorCodes.INVALID_ARGS, "Line already contains station " + midStation.Id);
            Link existingLink = FindActiveLinkForInsert(line, afterStationId);
            if (existingLink == null) return Fail(id, seq, BridgeErrorCodes.INVALID_ARGS, "Line has no active segment matching after_station_id " + afterStationId);

            Station segStart = existingLink.Start?.Station, segEnd = existingLink.End?.Station;
            existingLink.IsPendingMothball = true;
            Link prevLink = existingLink.PreviousLink, nextLink = existingLink.NextLink;
            Link firstHalf = line.InsertLink(existingLink, LineDirection.FORWARDS);
            firstHalf.Start.Configuration = existingLink.Start.Configuration; firstHalf.End.Station = midStation;
            Link secondHalf = line.InsertLink(existingLink, LineDirection.BACKWARDS);
            secondHalf.End.Configuration = existingLink.End.Configuration; secondHalf.Start.Station = midStation;

            if (firstHalf.End.Station != midStation || secondHalf.Start.Station != midStation)
            {
                bool noTunnel = (firstHalf.End.Station != midStation && LacksTunnelForSegment(game, city, segStart, midStation)) ||
                                (secondHalf.Start.Station != midStation && LacksTunnelForSegment(game, city, midStation, segEnd));
                firstHalf.Remove(); secondHalf.Remove(); line.RollbackMiddleEdit();
                return Fail(id, seq, noTunnel ? BridgeErrorCodes.TUNNEL_BUDGET_EXCEEDED : BridgeErrorCodes.INVALID_STATE, "Failed to split segment for INSERT_STATION");
            }
            LinkConnection.Connect(firstHalf, secondHalf);
            line.CommitMiddleEdit(prevLink, nextLink);
            line.Start(LineDirection.FORWARDS, null, startTrain: false, randomizeTrainPosition: false, tween: true);
            line.DirtyTracks(); game.OnLineEditComplete(); lineRegistry.ExtractOrGetCachedRoute(line);
            return null;
        }

        private static RpcResponseDto ExecuteReplaceLineCmd(Game game, City city, LineRegistry lineRegistry, Dictionary<string, object> args, object id, long seq)
        {
            if (!CheckAllowedKeys(args, "line_id", "station_ids", "is_looped", "loop") || !TryGetLoopedFlag(args, out bool isLooped)) return Fail(id, seq, BridgeErrorCodes.INVALID_ARGS, "Invalid REPLACE_LINE arguments");
            if (!TryGetEntityId(args, "line_id", out int lineId)) return Fail(id, seq, BridgeErrorCodes.INVALID_ARGS, "Missing or invalid 'line_id'");
            if (!TryResolveStations(city, args, isLooped, out List<Station> stations, out string stErr)) return Fail(id, seq, BridgeErrorCodes.INVALID_ARGS, stErr);
            if (!game.CanRemoveTracks) return Fail(id, seq, BridgeErrorCodes.INVALID_STATE, "Cannot replace lines in Extreme mode");
            if (!TryResolveActiveLine(city, lineRegistry, lineId, out Line oldLine, out string lineCode, out string lineErr)) return Fail(id, seq, lineCode, lineErr);

            List<int> curRoute = lineRegistry.ExtractOrGetCachedRoute(oldLine);
            if (!oldLine.IsLooping && curRoute.Count >= 2 && stations.Count == curRoute.Count + 1)
            {
                if (MatchesSlice(curRoute, stations, 0)) return ExecuteExtendOrPrepend(game, city, lineRegistry, oldLine, stations[stations.Count - 1], LineDirection.FORWARDS, isLooped, id, seq);
                if (MatchesSlice(curRoute, stations, 1)) return ExecuteExtendOrPrepend(game, city, lineRegistry, oldLine, stations[0], LineDirection.BACKWARDS, isLooped, id, seq);
                if (!isLooped && TryFindSingleInsertionIndex(curRoute, stations, out int insIdx)) return ExecuteInsertStation(game, city, lineRegistry, oldLine, stations[insIdx], stations[insIdx - 1].Id, id, seq);
            }
            if (!oldLine.IsLooping && isLooped && curRoute.Count >= 3 && stations.Count == curRoute.Count && MatchesSlice(curRoute, stations, 0))
                return ExecuteExtendOrPrepend(game, city, lineRegistry, oldLine, stations[0], LineDirection.FORWARDS, false, id, seq);

            int preferredColor = oldLine.ColorIndex; oldLine.Mothball();
            return ExecuteConnect(game, city, lineRegistry, stations, isLooped, preferredColor, id, seq);
        }

        private static RpcResponseDto ExecuteRemoveLineCmd(Game game, City city, LineRegistry lineRegistry, Dictionary<string, object> args, object id, long seq)
        {
            if (!CheckAllowedKeys(args, "line_id") || !TryGetEntityId(args, "line_id", out int lineId)) return Fail(id, seq, BridgeErrorCodes.INVALID_ARGS, "REMOVE_LINE requires 'line_id'");
            if (!game.CanRemoveTracks) return Fail(id, seq, BridgeErrorCodes.INVALID_STATE, "Cannot remove lines in Extreme mode");
            if (!TryResolveActiveLine(city, lineRegistry, lineId, out Line line, out string lineCode, out string lineErr)) return Fail(id, seq, lineCode, lineErr);
            lineRegistry.ExtractOrGetCachedRoute(line); line.Mothball(); return null;
        }

        private static RpcResponseDto ExecuteAssignLocoCmd(Game game, City city, LineRegistry lineRegistry, Dictionary<string, object> args, object id, long seq)
        {
            if (!CheckAllowedKeys(args, "line_id") || !TryGetEntityId(args, "line_id", out int lineId)) return Fail(id, seq, BridgeErrorCodes.INVALID_ARGS, "ASSIGN_LOCO requires 'line_id'");
            if (!TryResolveActiveLine(city, lineRegistry, lineId, out Line line, out string lineCode, out string lineErr)) return Fail(id, seq, lineCode, lineErr);
            if (line.ActiveTrainCount >= line.MaxLocomotivesPerLine()) return Fail(id, seq, BridgeErrorCodes.INVALID_STATE, "Line has reached maximum locomotives per line");
            AssetType loco = ResolveAvailableLocoForLine(game, city, line.Index);
            if (loco == AssetType.None) return Fail(id, seq, BridgeErrorCodes.INSUFFICIENT_RESOURCES, "No available locomotive in inventory");
            return line.ApplyAsset(loco, null, LineDirection.FORWARDS, null, scheduleAudioEvent: true, duringSetup: false, tween: true)
                ? null : Fail(id, seq, BridgeErrorCodes.INVALID_STATE, "Failed to assign locomotive to line");
        }

        private static RpcResponseDto ExecuteRelocateLocoCmd(Game game, City city, LineRegistry lineRegistry, Dictionary<string, object> args, object id, long seq)
        {
            if (!CheckAllowedKeys(args, "source_line_id", "target_line_id", "train_id")) return Fail(id, seq, BridgeErrorCodes.INVALID_ARGS, "Unexpected arguments for RELOCATE_LOCO");
            if (!TryGetEntityId(args, "source_line_id", out int srcLineId) || !TryGetEntityId(args, "target_line_id", out int dstLineId) || srcLineId == dstLineId || !TryGetOptionalEntityId(args, "train_id", out int? trainId))
                return Fail(id, seq, BridgeErrorCodes.INVALID_ARGS, "Invalid RELOCATE_LOCO arguments");
            if (!game.CanMoveAssets) return Fail(id, seq, BridgeErrorCodes.INVALID_STATE, "Cannot relocate locomotives in Extreme mode");
            if (!TryResolveActiveLine(city, lineRegistry, srcLineId, out Line srcLine, out string sc, out string se)) return Fail(id, seq, sc, se);
            if (!TryResolveActiveLine(city, lineRegistry, dstLineId, out Line dstLine, out string dc, out string de)) return Fail(id, seq, dc, de);
            if (ReferenceEquals(srcLine, dstLine) || dstLine.ActiveTrainCount >= dstLine.MaxLocomotivesPerLine()) return Fail(id, seq, BridgeErrorCodes.INVALID_STATE, "Target line cannot accept relocated locomotive");
            Train candTrain = FindRelocatableTrain(srcLine, trainId, out bool matchedTrainId);
            if (candTrain == null) return Fail(id, seq, trainId.HasValue && !matchedTrainId ? BridgeErrorCodes.INVALID_ARGS : BridgeErrorCodes.INVALID_STATE, "No eligible active locomotive found on source line");
            Locomotive existingLoco = candTrain.Locomotive; existingLoco.State = RailcarState.MOTHBALLED;
            if (dstLine.ApplyAsset(existingLoco.AssetType, null, LineDirection.FORWARDS, existingLoco, scheduleAudioEvent: true, duringSetup: false, tween: true)) return null;
            existingLoco.State = RailcarState.ACTIVE;
            return Fail(id, seq, BridgeErrorCodes.INVALID_STATE, "Failed to relocate locomotive onto target line");
        }

        private static RpcResponseDto ExecuteAttachCarriageCmd(Game game, City city, LineRegistry lineRegistry, Dictionary<string, object> args, object id, long seq)
        {
            if (!CheckAllowedKeys(args, "line_id", "train_id") || !TryGetEntityId(args, "line_id", out int lineId) || !TryGetOptionalEntityId(args, "train_id", out int? trainId)) return Fail(id, seq, BridgeErrorCodes.INVALID_ARGS, "Invalid arguments for ATTACH_CARRIAGE");
            if (!TryResolveActiveLine(city, lineRegistry, lineId, out Line line, out string lc, out string le)) return Fail(id, seq, lc, le);
            if (line.ActiveTrainCount <= 0 || line.CarriageCount >= Constants.MAX_CARRIAGES_PER_ROUTE) return Fail(id, seq, BridgeErrorCodes.INVALID_STATE, "Line cannot accept a carriage");
            if (game.AssetDatabase.GetAvailableAssets(AssetType.Carriage) <= 0) return Fail(id, seq, BridgeErrorCodes.INSUFFICIENT_RESOURCES, "No available carriage in inventory");
            TrackPosition? pos = null;
            if (trainId.HasValue)
            {
                Train tr = FindTrainById(line, trainId.Value);
                if (tr == null) return Fail(id, seq, BridgeErrorCodes.INVALID_ARGS, "Train not found on line: " + trainId.Value);
                if (tr.Locomotive == null || tr.Locomotive.State != RailcarState.ACTIVE || tr.Link.Link == null) return Fail(id, seq, BridgeErrorCodes.INVALID_STATE, "Target train is not active");
                pos = tr.Link.GetTrackPosition(tr.Distance);
            }
            return line.ApplyAsset(AssetType.Carriage, pos, LineDirection.FORWARDS, null, scheduleAudioEvent: true, duringSetup: false, tween: true)
                ? null : Fail(id, seq, BridgeErrorCodes.INVALID_STATE, "Failed to attach carriage");
        }

        private static RpcResponseDto ExecutePlaceInterchangeCmd(Game game, City city, Dictionary<string, object> args, object id, long seq)
        {
            if (!CheckAllowedKeys(args, "station_id") || !TryGetEntityId(args, "station_id", out int stationId)) return Fail(id, seq, BridgeErrorCodes.INVALID_ARGS, "PLACE_INTERCHANGE requires 'station_id'");
            Station st = city.GetStationById(stationId);
            if (st == null || !st.IsActive) return Fail(id, seq, BridgeErrorCodes.INVALID_ARGS, "Unknown or inactive station: " + stationId);
            if (st.IsInterchange || !st.CanBeConnected) return Fail(id, seq, BridgeErrorCodes.INVALID_STATE, "Station is already an interchange or cannot be upgraded");
            if (game.AssetDatabase.GetAvailableAssets(AssetType.Interchange) <= 0) return Fail(id, seq, BridgeErrorCodes.INSUFFICIENT_RESOURCES, "No available interchange in inventory");
            return st.ApplyAsset(AssetType.Interchange, quiet: false) ? null : Fail(id, seq, BridgeErrorCodes.INVALID_STATE, "Failed to apply interchange to station");
        }

        private static AssetType ResolveAvailableLocoForLine(Game game, City city, int lineIndex) =>
            (city.Definition?.AllowedTrains != null && lineIndex >= 0 && lineIndex < city.Definition.AllowedTrains.Count)
                ? game.AssetDatabase.SpecificAvailableLocomotiveAsset(city.Definition.AllowedTrains[lineIndex]) : game.AssetDatabase.AvailableLocomotiveAsset;

        private static bool LacksTunnelForSegment(Game game, City city, Station a, Station b)
        {
            if (a == null || b == null || game.AssetDatabase.GetAvailableAssets(AssetType.Crossing) > 0) return false;
            var fp = new LinkFootprint(null); fp.Update(a, b); return city.HasCrossingSection(fp);
        }

        private static Link FindActiveLinkForInsert(Line line, int afterStationId)
        {
            Link fallbackEnd = null, first = line.FirstActiveLink, cur = first;
            var visited = new HashSet<Link>();
            while (cur != null && visited.Add(cur))
            {
                if (cur.State == LinkState.ACTIVE && !cur.IsPendingMothball)
                {
                    if (cur.Start?.Station != null && cur.Start.Station.Id == afterStationId) return cur;
                    if (fallbackEnd == null && cur.End?.Station != null && cur.End.Station.Id == afterStationId) fallbackEnd = cur;
                }
                if (cur.NextLink == first) break;
                cur = cur.NextLink;
            }
            return fallbackEnd;
        }

        private static Train FindRelocatableTrain(Line line, int? trainId, out bool matchedTrainId)
        {
            matchedTrainId = false;
            for (int i = 0; i < line.TrainCount; i++)
            {
                Train tr = line.GetTrain(i);
                if (tr == null || (trainId.HasValue && tr.Id != trainId.Value)) continue;
                matchedTrainId = true;
                if (tr.Locomotive != null && tr.Locomotive.State == RailcarState.ACTIVE && !tr.HasMothballedRailcar) return tr;
                if (trainId.HasValue) return null;
            }
            return null;
        }

        private static Train FindTrainById(Line line, int trainId)
        {
            for (int i = 0; i < line.TrainCount; i++) { Train tr = line.GetTrain(i); if (tr != null && tr.Id == trainId) return tr; }
            return null;
        }

        private static bool TryResolveActiveLine(City city, LineRegistry reg, int lineId, out Line line, out string code, out string err)
        {
            Line any = reg.ResolveLine(city, lineId);
            if (any == null) { line = null; code = BridgeErrorCodes.INVALID_ARGS; err = "Unknown line_id: " + lineId; return false; }
            if (any.IsMothballed) { line = null; code = BridgeErrorCodes.INVALID_STATE; err = "Line is mothballed/ghost: " + lineId; return false; }
            line = any; code = null; err = null; return true;
        }

        private static bool TryResolveStations(City city, Dictionary<string, object> args, bool isLooped, out List<Station> stations, out string err)
        {
            stations = null; err = null;
            if (!args.TryGetValue("station_ids", out object raw) || !(raw is IList list) || list.Count < (isLooped ? 3 : 2))
            {
                err = isLooped ? "Looped line requires at least 3 distinct station_ids" : "Requires at least 2 distinct station_ids"; return false;
            }
            var seen = new HashSet<int>(); var res = new List<Station>(list.Count);
            for (int i = 0; i < list.Count; i++)
            {
                if (!TryParseEntityId(list[i], out int sid) || !seen.Add(sid)) { err = "Invalid or duplicate station_ids"; return false; }
                Station st = city.GetStationById(sid);
                if (st == null || !st.IsActive || !st.CanBeConnected) { err = "Station invalid or cannot be connected: " + sid; return false; }
                res.Add(st);
            }
            stations = res; return true;
        }

        private static bool MatchesSlice(List<int> route, List<Station> stations, int offset)
        {
            for (int i = 0; i < route.Count; i++) if (route[i] != stations[i + offset].Id) return false;
            return true;
        }

        private static bool TryFindSingleInsertionIndex(List<int> route, List<Station> stations, out int insertIdx)
        {
            insertIdx = -1; int r = 0;
            for (int s = 0; s < stations.Count; s++)
            {
                if (r < route.Count && route[r] == stations[s].Id) { r++; continue; }
                if (insertIdx != -1) return false;
                insertIdx = s;
            }
            return insertIdx > 0 && insertIdx < stations.Count - 1 && r == route.Count;
        }

        private static bool CheckAllowedKeys(Dictionary<string, object> args, params string[] allowed)
        {
            foreach (string k in args.Keys) { if (Array.IndexOf(allowed, k) < 0) return false; }
            return true;
        }

        private static bool TryGetLoopedFlag(Dictionary<string, object> args, out bool isLooped)
        {
            isLooped = false;
            bool hasIso = args.TryGetValue("is_looped", out object v1), hasLoop = args.TryGetValue("loop", out object v2);
            if (hasIso && hasLoop) return false;
            if (!hasIso && !hasLoop) return true;
            if ((hasIso ? v1 : v2) is bool b) { isLooped = b; return true; }
            return false;
        }

        private static bool TryGetEntityId(Dictionary<string, object> args, string key, out int id) { id = -1; return args.TryGetValue(key, out object v) && TryParseEntityId(v, out id); }

        private static bool TryGetOptionalEntityId(Dictionary<string, object> args, string key, out int? id)
        {
            id = null;
            if (!args.TryGetValue(key, out object v) || v == null) return true;
            if (TryParseEntityId(v, out int parsed)) { id = parsed; return true; }
            return false;
        }

        private static bool TryParseEntityId(object v, out int id)
        {
            id = -1;
            if (v is int i && i >= 0) { id = i; return true; }
            if (v is long l && l >= 0 && l <= int.MaxValue) { id = (int)l; return true; }
            if (v is string s && !string.IsNullOrWhiteSpace(s) && int.TryParse(s.Trim(), NumberStyles.None, CultureInfo.InvariantCulture, out int parsed) && parsed >= 0) { id = parsed; return true; }
            return false;
        }

        private static bool TryGetInt(object v, out int val)
        {
            if (v is int i) { val = i; return true; }
            if (v is long l && l >= int.MinValue && l <= int.MaxValue) { val = (int)l; return true; }
            val = 0; return false;
        }

        private static bool TryGetDouble(object v, out double val)
        {
            if (v is double d && !double.IsNaN(d) && !double.IsInfinity(d)) { val = d; return true; }
            if (v is float f && !float.IsNaN(f) && !float.IsInfinity(f)) { val = f; return true; }
            if (v is long l) { val = l; return true; }
            if (v is int i) { val = i; return true; }
            val = 0.0; return false;
        }
    }
}
