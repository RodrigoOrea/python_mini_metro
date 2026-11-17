using System;
using System.Collections.Generic;
using System.Reflection;
using Metro;
using UnityEngine;

namespace MiniMetroRoreaBridge
{
    /// <summary>
    /// Reads live Mini Metro simulation state on the Unity main thread and builds a validated <see cref="SnapshotDto"/>.
    /// </summary>
    public sealed class StateReader : IBridgeStateReader
    {
        private const BindingFlags InstanceFlags = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;

        private static readonly FieldInfo MainControllerField = typeof(Main).GetField("controller", InstanceFlags);
        private static readonly FieldInfo GameControllerGameField = typeof(GameController).GetField("game", InstanceFlags);
        private static readonly FieldInfo GameScreensField = typeof(Game).GetField("screens", InstanceFlags);
        private static readonly FieldInfo NewAssetPanelField = typeof(NewAssetScreen).GetField("newAssetPanel", InstanceFlags);
        private static readonly FieldInfo NewAssetButtonsField = typeof(NewAssetPanel).GetField("buttons", InstanceFlags);
        private static readonly FieldInfo PeepStartWaitTimeField = typeof(Peep).GetField("startWaitTime", InstanceFlags);
        private static readonly FieldInfo TrainPendingDeletionField = typeof(Train).GetField("pendingDeletion", InstanceFlags);

        private readonly LinkFootprint _footprint = new LinkFootprint(null);

        public SnapshotDto CaptureSnapshot(long seq, LineRegistry lineRegistry)
        {
            long safeSeq = Math.Max(0L, seq);
            LineRegistry registry = lineRegistry ?? new LineRegistry();

            Game game = ResolveActiveGame();
            City city = game != null ? game.City : null;
            if (Main.Instance == null || !Main.Instance.IsInGame || game == null || city == null)
            {
                registry.Prune(null);
                return new SnapshotDto
                {
                    Protocol = BridgeProtocol.Version,
                    Seq = safeSeq,
                    TimestampMs = 0L,
                    Screen = BridgeScreens.MENU,
                    CityId = "none",
                    IsPaused = false,
                    SpeedMultiplier = 1.0,
                    Deliveries = 0,
                    Day = 0,
                    Camera = BuildCamera(null),
                    Inventory = new InventoryDto()
                };
            }

            string screen = ResolveScreenState(game);
            double speedMultiplier = game.ScheduledTimeScale != null && game.ScheduledTimeScale.Scale >= 1.5f ? 2.0 : 1.0;
            float clockTime = city.Clock != null ? city.Clock.Time : 0f;
            long timestampMs = Math.Max(0L, (long)Math.Round(SanitizeDouble(clockTime, 0.0) * 1000.0));
            int day = Math.Max(0, city.Clock != null ? city.Clock.Day : 0);
            int deliveries = Math.Max(0, game.Score);
            string cityId = city.Definition != null && !string.IsNullOrWhiteSpace(city.Definition.Id)
                ? city.Definition.Id
                : "unknown";

            var snapshot = new SnapshotDto
            {
                Protocol = BridgeProtocol.Version,
                Seq = safeSeq,
                TimestampMs = timestampMs,
                Screen = screen,
                CityId = cityId,
                IsPaused = game.IsPaused,
                SpeedMultiplier = speedMultiplier,
                Deliveries = deliveries,
                Day = day,
                Camera = BuildCamera(city),
                Inventory = BuildInventory(game.AssetDatabase),
                WeeklyOffers = BuildWeeklyOffers(game)
            };

            var activeStations = new List<Station>();
            var activeStationIds = new HashSet<int>();
            PopulateStations(city, clockTime, snapshot.Stations, activeStations, activeStationIds);
            PopulateLinesAndTrains(city, registry, activeStationIds, snapshot.Lines, snapshot.Trains);
            PopulateCrossingCosts(city, activeStations, snapshot.CrossingCosts);
            return snapshot;
        }

        private static Game ResolveActiveGame()
        {
            Main main = Main.Instance;
            if (main == null || !main.IsInGame) return null;

            if (MainControllerField != null && GameControllerGameField != null &&
                MainControllerField.GetValue(main) is GameController gc &&
                GameControllerGameField.GetValue(gc) is Game reflectedGame)
            {
                return reflectedGame;
            }

            return DevStationView.Instance != null && DevStationView.Instance.City != null
                ? DevStationView.Instance.City.Game
                : null;
        }

        private static string ResolveScreenState(Game game)
        {
            if (game.IsOver || game.OvercrowdedStation != null || game.Screen == GameScreen.GameOver)
                return BridgeScreens.GAME_OVER;
            if (game.Screen == GameScreen.NewAsset)
                return BridgeScreens.WEEKLY_UPGRADE;
            if (game.Screen == GameScreen.Pause || (game.Screen == GameScreen.None && game.IsPaused))
                return BridgeScreens.PAUSED;
            if (game.Screen == GameScreen.None && !game.IsPaused)
                return BridgeScreens.PLAYING;
            return BridgeScreens.OTHER;
        }

        private static CameraDto BuildCamera(City city)
        {
            FScreen fscreen = Futile.screen;
            int screenW = fscreen != null && fscreen.pixelWidth >= 1f
                ? Math.Max(1, (int)Math.Round(SanitizeDouble(fscreen.pixelWidth, 1920.0)))
                : 1920;
            int screenH = fscreen != null && fscreen.pixelHeight >= 1f
                ? Math.Max(1, (int)Math.Round(SanitizeDouble(fscreen.pixelHeight, 1080.0)))
                : 1080;
            double orthoSize = fscreen != null ? SanitizeDouble(fscreen.height * 0.5, 540.0) : 540.0;
            if (orthoSize < 1.0) orthoSize = 540.0;

            double centerX = 0.0, centerY = 0.0;
            if (city != null && city.CityLayer != null && fscreen != null)
            {
                Rect visible = city.VisibleRect;
                centerX = SanitizeDouble(visible.center.x, 0.0);
                centerY = SanitizeDouble(visible.center.y, 0.0);
            }

            return new CameraDto
            {
                CenterX = centerX,
                CenterY = centerY,
                OrthoSize = orthoSize,
                ScreenWidth = screenW,
                ScreenHeight = screenH,
                WorldToRefScale = 1.0
            };
        }

        private static InventoryDto BuildInventory(AssetDatabase db)
        {
            if (db == null) return new InventoryDto();

            int linesAvail = Math.Max(0, db.GetAvailableAssets(AssetType.Line));
            int linesTotal = Math.Max(linesAvail, db.GetTotalAssets(AssetType.Line));
            int locosAvail = Math.Max(
                0,
                db.GetAvailableAssets(AssetType.Locomotive)
                + db.GetAvailableAssets(AssetType.Shinkansen)
                + db.GetAvailableAssets(AssetType.Tram));

            return new InventoryDto
            {
                LinesAvailable = linesAvail,
                LinesTotal = linesTotal,
                LocosAvailable = locosAvail,
                CarriagesAvailable = Math.Max(0, db.GetAvailableAssets(AssetType.Carriage)),
                TunnelsAvailable = Math.Max(0, db.GetAvailableAssets(AssetType.Crossing)),
                InterchangesAvailable = Math.Max(0, db.GetAvailableAssets(AssetType.Interchange))
            };
        }

        private static List<WeeklyOfferDto> BuildWeeklyOffers(Game game)
        {
            var offers = new List<WeeklyOfferDto>();
            if (game == null || game.Screen != GameScreen.NewAsset || GameScreensField == null) return offers;
            if (!(GameScreensField.GetValue(game) is UI.Screen[] screens)) return offers;

            int idx = (int)GameScreen.NewAsset;
            if (idx < 0 || idx >= screens.Length || !(screens[idx] is NewAssetScreen assetScreen)) return offers;
            if (NewAssetPanelField == null || !(NewAssetPanelField.GetValue(assetScreen) is NewAssetPanel panel)) return offers;
            if (NewAssetButtonsField == null || !(NewAssetButtonsField.GetValue(panel) is List<NewAssetButton> buttons)) return offers;

            for (int i = 0; i < buttons.Count; i++)
            {
                NewAssetButton button = buttons[i];
                if (button == null) continue;
                string kind = MapOfferAssetKind(button.Type);
                if (kind == null) continue;
                string rawLabel = button.Type.ToString();
                offers.Add(new WeeklyOfferDto
                {
                    Index = i,
                    Kind = kind,
                    Label = !string.IsNullOrWhiteSpace(rawLabel) ? rawLabel : kind,
                    Count = Math.Max(1, button.Count)
                });
            }
            return offers;
        }

        private static void PopulateStations(
            City city,
            float clockTime,
            List<StationDto> stationDtos,
            List<Station> activeStations,
            HashSet<int> activeStationIds)
        {
            FScreen fscreen = Futile.screen;
            float halfW = fscreen != null ? fscreen.halfWidth : 960f;
            float halfH = fscreen != null ? fscreen.halfHeight : 540f;
            float pixelH = fscreen != null && fscreen.pixelHeight >= 1f ? fscreen.pixelHeight : 1080f;
            float dispScale = Futile.displayScale > 0f ? Futile.displayScale : 1f;

            int count = Math.Max(0, city.StationCount);
            for (int i = 0; i < count; i++)
            {
                Station st = city.GetStation(i);
                if (st == null || !st.IsActive || st.Id < 0 || !activeStationIds.Add(st.Id)) continue;

                activeStations.Add(st);
                Vector2 pos = st.Position;
                Vector2 globalPos = city.CityLayer != null ? city.CityLayer.LocalToGlobal(pos) : pos;
                double sx = (globalPos.x + halfW) * dispScale;
                double syTopLeft = pixelH - ((globalPos.y + halfH) * dispScale);
                double overload = Math.Min(1.0, Math.Max(0.0, SanitizeDouble(st.ExpiryTimerCompletion, 0.0)));

                var waiting = new List<WaitingPassengerDto>();
                int peepCount = Math.Max(0, st.PeepCount);
                for (int j = 0; j < peepCount; j++)
                {
                    Peep p = st.GetPeep(j);
                    if (p == null) continue;
                    float startWait = ReflectStartWaitTime(p);
                    float waitSec = startWait >= 0f
                        ? Math.Max(0f, clockTime - startWait)
                        : Math.Max(0f, p.TotalWaitTime(clockTime));
                    waiting.Add(new WaitingPassengerDto
                    {
                        DestShape = MapStationShape(p.TypeIndex, p.Type),
                        WaitMs = Math.Max(0, (int)Math.Round(SanitizeDouble(waitSec, 0.0) * 1000.0))
                    });
                }

                stationDtos.Add(new StationDto
                {
                    Id = st.Id,
                    X = SanitizeDouble(pos.x, 0.0),
                    Y = SanitizeDouble(pos.y, 0.0),
                    ScreenX = SanitizeDouble(sx, 0.0),
                    ScreenY = SanitizeDouble(syTopLeft, 0.0),
                    Shape = MapStationShape(st.TypeIndex, st.Type),
                    IsInterchange = st.IsInterchange,
                    OverloadProgress = overload,
                    WaitingPassengers = waiting,
                    PeepCapacity = Math.Max(1, st.PeepCapacity)
                });
            }
        }

        private static void PopulateLinesAndTrains(
            City city,
            LineRegistry registry,
            HashSet<int> activeStationIds,
            List<LineDto> lineDtos,
            List<TrainDto> trainDtos)
        {
            registry.Prune(city);
            var emittedLineIds = new HashSet<int>();
            var emittedTrainIds = new HashSet<int>();
            int fallbackTrainId = 1_000_000;

            int lineCount = Math.Max(0, city.LineCount);
            for (int i = 0; i < lineCount; i++)
            {
                Line ln = city.GetLine(i);
                if (ln == null) continue;

                List<int> rawRoute = registry.ExtractOrGetCachedRoute(ln);
                if (rawRoute == null || rawRoute.Count < 2) continue;

                var route = new List<int>(rawRoute.Count);
                var seenStations = new HashSet<int>();
                bool validStations = true;
                for (int r = 0; r < rawRoute.Count; r++)
                {
                    int sid = rawRoute[r];
                    if (!activeStationIds.Contains(sid)) { validStations = false; break; }
                    if (seenStations.Add(sid)) route.Add(sid);
                }
                if (!validStations || route.Count < 2) continue;

                int lineId = registry.GetOrAssignId(ln);
                if (lineId < 0 || !emittedLineIds.Add(lineId)) continue;

                int rawColor = ln.ColorIndex >= 0 ? ln.ColorIndex : ln.Index;
                lineDtos.Add(new LineDto
                {
                    Id = lineId,
                    ColorIndex = Math.Max(0, rawColor),
                    StationIds = route,
                    IsLooped = ln.IsLooping,
                    IsGhost = ln.IsMothballed,
                    TunnelsUsed = CountTunnelsUsed(ln)
                });

                int trainCount = Math.Max(0, ln.TrainCount);
                for (int t = 0; t < trainCount; t++)
                {
                    Train tr = ln.GetTrain(t);
                    if (tr == null) continue;

                    int trainId = tr.Id;
                    if (trainId < 0 || !emittedTrainIds.Add(trainId))
                    {
                        while (!emittedTrainIds.Add(fallbackTrainId)) fallbackTrainId++;
                        trainId = fallbackTrainId;
                    }

                    int railcarCount = Math.Max(0, tr.RailcarCount);
                    int capacity = 0;
                    for (int r = 0; r < railcarCount; r++)
                    {
                        Railcar rc = tr.GetRailcar(r);
                        if (rc != null && rc.Capacity > 0) capacity += rc.Capacity;
                    }
                    if (capacity <= 0) capacity = 6 * Math.Max(1, railcarCount);

                    Locomotive loco = tr.Locomotive;
                    bool isMothballedLoco = loco != null && loco.State == RailcarState.MOTHBALLED;
                    trainDtos.Add(new TrainDto
                    {
                        Id = trainId,
                        LineId = lineId,
                        Carriages = Math.Max(0, railcarCount - 1),
                        Capacity = capacity,
                        OnboardCount = Math.Max(0, tr.PeepCount),
                        IsUnassignmentQueued = isMothballedLoco || ReflectPendingDeletion(tr),
                        IsGhost = ln.IsMothballed,
                        IsWaitingForLocomotive = ln.IsWaitingForLocomotive,
                        LocoType = loco != null ? MapLocoType(loco.AssetType) : "LOCOMOTIVE",
                        IsPending = loco != null && loco.State == RailcarState.PENDING
                    });
                }
            }
        }

        private void PopulateCrossingCosts(City city, List<Station> activeStations, List<CrossingCostDto> costs)
        {
            int n = activeStations.Count;
            for (int a = 0; a < n; a++)
            {
                Station stA = activeStations[a];
                for (int b = a + 1; b < n; b++)
                {
                    Station stB = activeStations[b];
                    if (stA.Id == stB.Id) continue;
                    Station first = stA.Id < stB.Id ? stA : stB;
                    Station second = stA.Id < stB.Id ? stB : stA;
                    _footprint.Update(first, second);
                    if (city.HasCrossingSection(_footprint))
                        costs.Add(new CrossingCostDto { U = first.Id, V = second.Id, Tunnels = 1 });
                }
            }
        }

        private static int CountTunnelsUsed(Line ln)
        {
            int tunnels = 0, count = Math.Max(0, ln.Count);
            for (int i = 0; i < count; i++)
            {
                Link link = ln[i];
                if (link != null && link.HasCrossing && !link.IsPendingMothball && link.State != LinkState.MOTHBALLED)
                    tunnels++;
            }
            return tunnels;
        }

        private static float ReflectStartWaitTime(Peep peep) =>
            PeepStartWaitTimeField != null && PeepStartWaitTimeField.GetValue(peep) is float v && !float.IsNaN(v) && !float.IsInfinity(v)
                ? v
                : -1f;

        private static bool ReflectPendingDeletion(Train train) =>
            TrainPendingDeletionField != null && TrainPendingDeletionField.GetValue(train) is bool flag && flag;

        private static string MapStationShape(StationTypeIndex typeIndex, StationType typeMask) => typeIndex switch
        {
            StationTypeIndex.CIRCLE => "CIRCLE",
            StationTypeIndex.TRIANGLE => "TRIANGLE",
            StationTypeIndex.SQUARE => "SQUARE",
            StationTypeIndex.CROSS => "CROSS",
            StationTypeIndex.DIAMOND => "DIAMOND",
            StationTypeIndex.EGG => "EGG",
            StationTypeIndex.GEM => "GEM",
            StationTypeIndex.PENTAGON => "PENTAGON",
            StationTypeIndex.STAR => "STAR",
            StationTypeIndex.WEDGE => "WEDGE",
            _ => MapStationTypeMask(typeMask)
        };

        private static string MapStationTypeMask(StationType m)
        {
            if ((m & StationType.CIRCLE) != 0) return "CIRCLE";
            if ((m & StationType.TRIANGLE) != 0) return "TRIANGLE";
            if ((m & StationType.SQUARE) != 0) return "SQUARE";
            if ((m & StationType.CROSS) != 0) return "CROSS";
            if ((m & StationType.DIAMOND) != 0) return "DIAMOND";
            if ((m & StationType.EGG) != 0) return "EGG";
            if ((m & StationType.GEM) != 0) return "GEM";
            if ((m & StationType.PENTAGON) != 0) return "PENTAGON";
            if ((m & StationType.STAR) != 0) return "STAR";
            if ((m & StationType.WEDGE) != 0) return "WEDGE";
            return "CIRCLE";
        }

        private static string MapOfferAssetKind(AssetType type) => type switch
        {
            AssetType.Line => "LINE",
            AssetType.Locomotive => "LOCOMOTIVE",
            AssetType.Shinkansen => "SHINKANSEN",
            AssetType.Tram => "TRAM",
            AssetType.Ferry => "FERRY",
            AssetType.Carriage => "CARRIAGE",
            AssetType.Crossing => "CROSSING",
            AssetType.Bridge => "CROSSING",
            AssetType.Interchange => "INTERCHANGE",
            _ => null
        };

        private static string MapLocoType(AssetType type) => type switch
        {
            AssetType.Shinkansen => "SHINKANSEN",
            AssetType.Tram => "TRAM",
            _ => "LOCOMOTIVE"
        };

        private static double SanitizeDouble(double v, double fallback) =>
            double.IsNaN(v) || double.IsInfinity(v) ? fallback : v;
    }
}
