# Mini Metro Managed Assembly API Reference (`Assembly-CSharp.dll`)

Read-only inspection reference of `/home/rodrigo/.steam/steam/steamapps/common/MiniMetro/Mini Metro_Data/Managed/Assembly-CSharp.dll` (and `Assembly-CSharp-firstpass.dll`, which contains only `Steamworks.*` Steamworks.NET bindings and Unity script metadata).

Most gameplay and UI classes sit in the **global namespace**. Key exceptions live in `namespace Metro` (notably `Metro.AssetDatabase`, `Metro.AStarSearchResult`, `Metro.Debug`, `Metro.LineSegment`) and `namespace UI` (`UI.Screen`, `UI.Button`, `UI.Constants`). Every symbol, visibility (`public` vs `private`), enum value, constant, and call sequence below was directly observed in the decompiled C# source.

---

## 1. Root Entry, Game Lifecycle & Screen Detection

### Entry Points from a Unity Plugin (`MonoBehaviour`)

The game root singleton is `Main` (inherits `MonoBehaviour`, global namespace):

- `Main.Instance` (`public static Main Instance`) gives the active `Main` instance.
- `Main.Instance.IsInGame` (`public bool IsInGame`) returns `controller != null && controller is GameController`.
- `Main.controller` (`private IController controller;`) holds either a `GameController` or a `MenuController`:
  - `GameController.game` (`private Game game;`) holds the active `Game`.
  - `MenuController.menu` (`private Menu menu;`) holds the active `Menu`.
- **Reflection-free in-game shortcut**: When a `Game` is active, `Game` assigns `DevStationView.Instance.City = this.city;` and `DevLineView.Instance.City = this.city;` (`Game` clears both to `null` in `Release()`). If `Main.Instance.IsInGame` is `true` and `DevStationView.Instance.City != null`, `DevStationView.Instance.City.Game` yields `Game` directly without reflection, though reflecting `Main.controller` and `GameController.game` (or `MenuController.menu`) works across all states including menus and transitions.

### Screen & Lifecycle Enums and Properties

**Menu screen state** (when `controller is MenuController`):

- `Menu.Option` (`public MenuOption Option`)
- `Menu.CurrentScreen` (`public MenuScreen CurrentScreen`)
- `Menu.ModalScreen` (`public MenuScreen ModalScreen`)
- `MenuOption` enum values: `None = -1`, `Home = 0`, `Options = 1`, `City = 2`, `Mode = 3`, `Leaderboard = 4`, `DailyChallenge = 5`, `DailyChallengeLeaderboard = 6`, `Credits = 7`, `Changelog = 8`, `Loading = 9`, `Studio = 10`, `FAQ = 11`, `Achievement = 12`, `UgcCity = 13`, `Resume = 14`, `SelectProfile = 15`, `CreateEditProfile = 16`, `Count = 17`.

**In-game screen state** (when `controller is GameController` and `game != null`):

- `Game.Screen` (`public GameScreen Screen`) — active modal overlay (`GameScreen.None` during live HUD gameplay).
- `Game.IsScreenChangeScheduled` (`public bool IsScreenChangeScheduled`) — `true` when `scheduledScreen != GameScreen.None` (for example right before weekly upgrade or game-over transition triggers in `Game.Update()`).
- `Game.IsOver` (`public bool IsOver`) — `true` once game over has occurred (note `Game.ScheduleAudioEvents` sets `scheduledScreen = GameScreen.GameOver` when a station overcrowds).
- `Game.IsLocked` (`public bool IsLocked`) — locks player touch/HUD input during transitions or modals.
- `Game.OvercrowdedStation` (`public Station OvercrowdedStation`) — station that caused game over (if any).
- `Game.screens` (`private UI.Screen[] screens = new UI.Screen[12];`) — modal screen instances indexed by `(int)GameScreen`.
- `GameScreen` enum values: `None = -1`, `Pause = 0`, `GameOver = 1`, `NewAsset = 2`, `Photo = 3`, `Statistics = 4`, `Intro = 5`, `Sigil = 6`, `Buy = 7`, `FAQ = 8`, `Achievement = 9`, `CompletedChallenge = 10`, `GifCapture = 11`, `Count = 12`.

**Normalized screen mapping**:

Note that opening `GameScreen.Pause` or `GameScreen.NewAsset` also sets `Game.IsPaused = true`, so `Game.Screen` and `Game.IsOver` MUST be evaluated before checking `Game.IsPaused`. Furthermore, when a station overcrowds, `Game.ScheduleAudioEvents` assigns `overcrowdedStation = City.OvercrowdedStation;` and `scheduledScreen = GameScreen.GameOver;` before `Game.Update()` transitions `Game.Screen` and `IsOver` flips `true`, so `game.OvercrowdedStation != null` catches imminent game over right away.

| Normalized State | Exact Condition |
| :--- | :--- |
| `MENU` | `Main.Instance == null` or `!Main.Instance.IsInGame` or `game == null` |
| `GAME_OVER` | `game.IsOver \|\| game.Screen == GameScreen.GameOver \|\| game.OvercrowdedStation != null` |
| `WEEKLY_UPGRADE` | `game.Screen == GameScreen.NewAsset` |
| `PAUSED` | `game.Screen == GameScreen.Pause` or (`game.Screen == GameScreen.None && game.IsPaused`) |
| `PLAYING` | `game.Screen == GameScreen.None && !game.IsPaused && !game.IsOver` |
| Other In-Game Overlay | `game.Screen` in `{Photo, Statistics, Intro, Sigil, Buy, FAQ, Achievement, CompletedChallenge, GifCapture}` |

### Pause, Speed Multipliers, Score, Clock & City Metadata

- **Pause & speed**:
  - `Game.IsPaused` (`public bool IsPaused`) — logical pause state (`Game.time.IsPaused` tracks simulation pause).
  - `Game.ScheduledTimeScale` (`public TimeScale ScheduledTimeScale`) — desired gameplay speed.
  - `TimeScale` (`public class TimeScale`, `public float Scale => scale`):
    - `TimeScale.Single` = `1f` (`1x`)
    - `TimeScale.SingleSlow` = `0.75f` (used by `HudScreen.HandleVCRChange` for audio pulse slow-down during pause)
    - `TimeScale.Double` = `2f` (`2x`)
    - `TimeScale.DoubleSlow` = `1.5f` (audio pulse slow-down when paused from fast-forward)
  - `VCRButtonType` enum: `PAUSE`, `PLAY`, `FAST_FORWARD`, `SKIP_NEXT`, `SKIP_PREVIOUS`.
  - **Important**: Vanilla Mini Metro has **no 4x speed** (`VCRButtonType` only binds `PAUSE` -> `Alpha1`, `PLAY` -> `1f` / `Alpha2`, and `FAST_FORWARD` -> `2f` / `Alpha3`). Any bridge exposing `speed_4x` must either clamp/map it to `2x` (`TimeScale.Double`) or report that only `1x`/`2x` exist natively.
- **Score / deliveries**:
  - `Game.Score` (`public int Score => scoreController.Score;`) — integer count of delivered passengers.
  - `Game.scoreController` (`private ScoreController scoreController;`) also exposes `PeakRollingScore` (`int`) and `RollingScore` (`float`, Zen mode rolling average).
- **Time / day / week**:
  - `Game.Week` (`public int Week => week;`) — current week index (`0`-based, matches `City.Clock.Week`).
  - `Game.City.Clock` (`public Clock Clock`):
    - `Clock.DayLength` = `20f` seconds per in-game day (`Constants.DAY_LENGTH = 20f`).
    - `Clock.Time` (`public float Time`) — elapsed simulation seconds.
    - `Clock.Day` (`public int Day`) — `0`-based integer day (`Mathf.FloorToInt(time / DayLength)`).
    - `Clock.DayOfTheWeek` (`public int DayOfTheWeek`) — `Day % 7` (`0 = MONDAY .. 6 = SUNDAY`).
    - `Clock.Hour` (`public float Hour`) — `[0, 24)` within current day.
    - `Clock.Week` (`public int Week`) and `Clock.ClosestWeek` (`public int ClosestWeek`).
    - `Clock.IsPaused` (`public bool IsPaused`).
    - Note: `Game.day` (`private int day;`) is private; read `Game.City.Clock.Day` instead.
- **City ID & Game Mode**:
  - `Game.City.Definition` (`public CityDefinition Definition`):
    - `Definition.Id` (`public string Id { get; set; }`) — map identifier (e.g., `"london"`, `"paris"`, etc.).
    - `Definition.BaseCityId` (`public string BaseCityId { get; private set; }`).
    - `Definition.IsUgc` (`public bool IsUgc { get; private set; }`).
    - `Definition.LocalizedName` (`public LocalizedString LocalizedName`).
    - `Definition.CrossingStyle` (`public CrossingStyle CrossingStyle`: `Bridge = 0`, `Tunnel = 1`).
  - `Game.Mode` (`public GameMode Mode`): enum values `CLASSIC = 0`, `ZEN = 1`, `EXTREME = 2`, `SANDBOX = 3`, `PLAYABLE_COUNT = 4`, `FAQ = 5`.
  - `Game.CanRemoveTracks` (`public bool CanRemoveTracks => mode != GameMode.EXTREME;`) and `Game.CanMoveAssets` (`public bool CanMoveAssets => mode != GameMode.EXTREME;`).

---

## 2. Camera & Coordinate Scaling

### Futile 2D Stage & Orthographic Camera

Mini Metro renders via the Futile 2D framework rather than standard 3D Unity scene transforms:

- `Futile.instance.camera` (`public Camera camera => _cameras[0];`) is tagged `"MainCamera"` with `orthographic = true`, `orthographicSize = screen.pixelHeight / 2f * displayScaleInverse` (`= Futile.screen.height * 0.5f`), and `transform.position = new Vector3(0f, 0f, -10f)` (since `Main.Begin()` sets `futileParams.origin = new Vector2(0.5f, 0.5f)`).
- `Main.Instance.Scene.CityStage` (`FStage`) holds the active `Game.Layer` (`CityStage.AddChild(game.Layer)`), and `GameController` sets `game.City.CityLayer.IsTransformingStage = true` so pan/zoom transforms `CityStage` directly.
- `Main.Instance.Scene.OverlayStage` (`FStage`) renders HUD/modals in screen-centered Futile coordinates (`[-Futile.screen.halfWidth, +Futile.screen.halfWidth] x [-Futile.screen.halfHeight, +Futile.screen.halfHeight]`).

### City World Space, Zoom & Visible Bounds

All simulation positions (`Station.Position`, `LinkFootprint`, `TrackPosition.Position`, `Train`, `Peep`) live in **City local coordinates** centered near `(0, 0)` within a `Constants.TOTAL_WIDTH = 2048f` by `Constants.TOTAL_HEIGHT = 1536f` map canvas:

- `City.Zoom` (`public float Zoom`): current city zoom scale (`CityDefinition.StartZoom` defaults to `1.2f`, `EndZoom` defaults to `0.8f`, `MaxZoom` = `StartZoom * 1.5f`). Note `GameController`/`City` sets `CityLayer.SetScale(zoom)`.
- `City.Offset` (`public Vector2 Offset`): pan offset relative to default zoom center.
- `City.CityLayer` (`public CityLayer CityLayer`):
  - `CityLayer.scale` (`public float scale`): active zoom scale on the city root transform.
  - `CityLayer.LocalToGlobal(Vector2 localVector)` / `CityLayer.GlobalToLocal(Vector2 globalVector)`: convert between **City local world coordinates** and **Futile global stage coordinates** (`[-halfWidth, halfWidth] x [-halfHeight, halfHeight]` centered at screen center). Note `nodes` is `private FContainer[] nodes = new FContainer[3];`.
- `City.VisibleRect` (`public Rect VisibleRect`): returns the currently visible rectangle in **City local coordinates** (`width = Futile.screen.width / CityLayer.scale`, `height = Futile.screen.height / CityLayer.scale`, centered at `CityLayer.GetPosition() / -CityLayer.scale`).
- `City.GetLayerRectAtZoom(float zoom)` and `City.GetPlayableArea(float time)`: return 4:3-normalized map layer rect and station spawnable bounds in City coordinates.

### Converting Between City Coordinates and Screen Pixels (`WorldToScreenPoint`)

From `FNode.LocalToScreen` and `Futile.UpdateCameraPosition()`:

1. **Futile global (`globalPos`) <-> Unity camera world**: Because `futileParams.origin = (0.5f, 0.5f)` puts `(0, 0)` at screen center and `orthographicSize = Futile.screen.height * 0.5f`, Futile global coordinates `(gx, gy)` equal Unity `Camera.main` world coordinates `(gx, gy, 0f)` (up to `Futile.screenPixelOffset` half-pixel correction).
2. **City local -> Unity screen pixels** (bottom-left `(0, 0)` to top-right `(UnityEngine.Screen.width, UnityEngine.Screen.height)`):
   - Note that `CityLayer.nodes` is `private` (`private FContainer[] nodes = new FContainer[3];` in `CityLayer.cs`), so `city.CityLayer.nodes[2].LocalToScreen(cityPos)` requires reflecting `CityLayer.nodes`. In addition, when `IsTransformingStage == true`, `CityLayer` places pan/zoom transforms directly on `CityStage`. Therefore, the safest reflection-free path uses the public `city.CityLayer.LocalToGlobal(cityPos)` followed by the exact formula from `FNode.LocalToScreen`:
     ```csharp
     Vector2 globalPos = city.CityLayer.LocalToGlobal(cityPos);
     // Equivalent to Futile.instance.camera.WorldToScreenPoint(new Vector3(globalPos.x, globalPos.y, 0f))
     float sx = (globalPos.x + Futile.screen.halfWidth) * Futile.displayScale;
     float sy = (globalPos.y + Futile.screen.halfHeight) * Futile.displayScale;
     ```
   - To match top-left image/OpenCV/Python `(x, y)` coordinates, flip Y: `yTopLeft = Futile.screen.pixelHeight - sy`.
3. **Screen pixels -> City local**: Invert with `Vector2 globalPos = new Vector2(sx * Futile.displayScaleInverse - Futile.screen.halfWidth, sy * Futile.displayScaleInverse - Futile.screen.halfHeight); Vector2 cityPos = city.CityLayer.GlobalToLocal(globalPos);` (matching `FNode.ScreenToLocal` + `CityLayer.GlobalToLocal`).

### Mapping Real Mini Metro World Distances to `D_REF = 250.0`

In `src/rl/objective_ml.py`, `D_REF = 250.0` normalizes Euclidean station/route distances in `python_mini_metro`'s simulation coordinate frame. In real Mini Metro:

- `Station.Position` is stored in **zoom-invariant City world units** (`TOTAL_WIDTH = 2048f`, `TOTAL_HEIGHT = 1536f`), NOT screen pixels (stations do not change `Position` as `City.Zoom` zooms out over weeks).
- Key geometric constants in City units (`Constants`):
  - `Constants.STATION_SIZE = 36f`
  - `Constants.MIN_CENTRAL_STATION_SEPARATION = 180f` (minimum spacing between central stations, scaled by `CityDefinition.StationSeparationScale`)
  - `Constants.MIN_LINK_LENGTH = 30f`, `Constants.MIN_LINK_LENGTH_FOR_ERROR = 75f`
  - `Constants.RAILCAR_SPACING = 32.5f`, `Constants.DEFAULT_RAILCAR_WIDTH = 26f`
- `Main.ConfigureResolution` maps the 4:3 reference viewport (`1024 x 768` Futile units, i.e., `DefaultAspectRatio = 4/3`):
  - If `AspectRatio > 4/3`: `aspectScale = Futile.screen.pixelHeight / 768f * Futile.displayScaleInverse`
  - Else: `aspectScale = Futile.screen.pixelWidth / 1024f * Futile.displayScaleInverse`
- Thus, a distance `d_city` in City local units spans `d_city * city.Zoom * Main.Instance.AspectScale` Futile reference units (`1024x768` baseline) or `d_city * city.Zoom * Main.Instance.AspectScale * Futile.displayScale` raw screen pixels. Because real Mini Metro runs heuristic/routing checks in **City local units** (which are invariant across screen resolution and late-game zoom-out), the cleanest correspondence is to treat `Station.Position` directly as world units and either calibrate `D_REF` to City units (~`250f` directly, given `MIN_CENTRAL_STATION_SEPARATION = 180f` and early-week visible width `1024 / 1.2 ≈ 853` units), or scale by `1.0` when `1000`px `python_mini_metro` maps correspond directly to the ~`853..1280` City units across `StartZoom = 1.2` to `EndZoom = 0.8`.

---

## 3. Stations, Waiting Passengers & Overload

### Enumerating Active Stations

- `City.StationCount` (`public int StationCount`) returns the count of spawned, active stations (`stations[i] != null && stations[i].IsActive`).
- `City.GetStation(int index)` (`public Station GetStation(int index)`) returns the `index`-th active station (`0 <= index < StationCount`).
- `City.GetStationById(int id)` (`public Station GetStationById(int id)`) looks up any entry in `City.stations` by `Id` — note that this does **not** filter `IsActive` or `pendingDeletion`, so callers using `GetStationById` should verify `station != null && station.IsActive && station.CanBeConnected`.

### `Station` Type & Members

| Member | Kind / Visibility | Type | Meaning |
| :--- | :--- | :--- | :--- |
| `Id` | `public` property | `int` | Unique integer station ID |
| `Position` | `public` property | `Vector2` | Station center in City local coordinates |
| `Type` | `public` property | `StationType` | Shape bitmask (`[Flags]`) |
| `TypeIndex` | `public` property | `StationTypeIndex` | Shape enum index (`0..11`) |
| `IsActive` | `public` property | `bool` | `true` once station has spawned (excludes pre-spawn previews) |
| `IsGhost` | `public` property | `bool` | `true` for unplaced/ghost stations |
| `CanBeConnected` | `public` property | `bool` | `!pendingDeletion && IsActive` |
| `IsInterchange` | `public` property | `bool` | `true` if upgraded with `AssetType.Interchange` |
| `IsPort` | `public` property | `bool` | `true` if ferry port (`ferryDirections != null`) |
| `PeepCount` | `public` property | `int` | Current waiting passenger count (`peeps.Count`) |
| `PeepCapacity` | `public` property | `int` | Effective overload threshold (`99999` if invincible, else `City.Definition.InterchangeCapacity` [default `18`] if `IsInterchange`, else `City.Definition.StationCapacity` [default `6`]) |
| `NumPeepsOverCapacity` | `public` property | `int` | `Math.Max(0, PeepCount - PeepCapacity)` |
| `GetPeep(int index)` | `public` method | `Peep` | Returns `index`-th non-null waiting `Peep` |
| `PeepTypes` | `public` property | `HashSet<StationType>` | Distinct destination shapes waiting |
| `ExpiryTimerCompletion` | `public` property | `float` | Overload ring fill in `[0f, 1f]` (`1f - Mathf.Max(0f, time / duration)`), `0f` when no timer |
| `HasOvercrowded` | `public` property | `bool` | `expiryTimer != null && expiryTimer.Expired` |
| `IsTrainAtStation` | `public` property | `bool` | `true` while any train on any connected line is stopped here (pauses overload countdown) |
| `Lines` | `public` property | `List<Line>` | Deduplicated non-mothballed lines touching this station |
| `LineCount` | `public` property | `int` | Count of distinct non-mothballed lines at this station |
| `LinkStops` | `public` property | `List<LinkStop>` | All connected `LinkStop` objects |
| `expiryTimer` | `private` field | `StationTimer` | Present only while station has an active/decaying overload timer |
| `pendingDeletion` | `private` field | `bool` | Pending removal flag |

**StationTimer internals** (`private StationTimer expiryTimer` on `Station`):
- `Constants.STATION_EXPIRY_TIME = 45f` (`duration = 45f`), `Constants.STATION_EXPIRY_GRACE_TIME = 2f` (`gracePeriod = 2f`).
- `StationTimer.Time` (`public float Time`) starts at `45f`, decrements by `time.ScaledDelta` only when `peeps.Count > PeepCapacity && !IsTrainAtStation && !Expired`, recovers upwards when below capacity, and sets `Expired => time <= -gracePeriod`.

**Shape enums (`StationType` and `StationTypeIndex`)**:
- `StationType` (`[Flags]`): `NONE = 0`, `CIRCLE = 1`, `TRIANGLE = 2`, `SQUARE = 4`, `STANDARD = 7`, `REPLACEABLE = 3`, `CROSS = 8`, `DIAMOND = 0x10`, `EGG = 0x20`, `GEM = 0x40`, `PENTAGON = 0x80`, `STAR = 0x100`, `WEDGE = 0x200`, `SPECIAL = 0x3F8`.
- `StationTypeIndex`: `NONE = 0`, `CIRCLE = 1`, `TRIANGLE = 2`, `SQUARE = 3`, `CROSS = 4`, `DIAMOND = 5`, `EGG = 6`, `GEM = 7`, `PENTAGON = 8`, `STAR = 9`, `WEDGE = 10`, `MAX = 11`.

### Waiting Passengers (`Peep`)

- `Peep.Id` (`public int Id`)
- `Peep.Type` (`public StationType Type`) — destination station shape bitmask.
- `Peep.TypeIndex` (`public StationTypeIndex TypeIndex`) — destination station shape index.
- `Peep.TotalWaitTime(float currentTime)` (`public float TotalWaitTime(float currentTime)`) — cumulative wait time across all stations/transfers (`totalWaitTime + (currentTime - startWaitTime)` while waiting at a station; pass `city.Clock.Time`).
- `Peep.startWaitTime` (`private float startWaitTime;`) — timestamp (`city.Clock.Time`) when the passenger last arrived at its current station (`-1f` while onboard a train). Reflect this field to compute exact wait age at the current station: `city.Clock.Time - startWaitTime`.
- `Peep.totalWaitTime` (`private float totalWaitTime;`) — accumulated wait time from prior stations before transfers.
- `Peep.StationAbandonTime` (`public float StationAbandonTime`) — time at which passenger abandons station in Zen/Sandbox (`-1f` when inactive/on train).
- `Peep.IsEmbarking` (`public bool IsEmbarking`), `Peep.IsDisembarking` (`public bool IsDisembarking`), `Peep.HasReservation` (`public bool HasReservation`).
- `Peep.NumTransfers` (`public int NumTransfers`), `Peep.VisitedStationsCount` (`public int VisitedStationsCount`), `Peep.DistanceTravelled` (`public float DistanceTravelled`).
- `Peep.CurrentPath` (`public AStarSearchResult CurrentPath`, namespace `Metro`).
- **`PeepContainer` null-slot caveat**: `PeepContainer.peeps` (`private List<Peep>`) keeps fixed slots filled with `null` when `positionPeepsRandomly` is `true` (used on trains/railcars). Always read via `station.GetPeep(i)` (`0 <= i < station.PeepCount`), `railcar.Peeps` (already filters `null`), or `train.GetPeep(i)` (`0 <= i < train.PeepCount`), rather than reflecting `PeepContainer.peeps` directly.

---

## 4. Lines (Loops, Ghost/Mothballed Lines) & Trains/Carriages

### Enumerating Lines & Ghost/Mothballed Semantics

- `City.MaxLineCount` (`public int MaxLineCount => definition.LineCount;`) — maximum line slots in the city (Classic default `7`, range `1..20`).
- `City.LineCount` (`public int LineCount => lines.Count;`) — **includes** mothballed (`IsMothballed == true`) ghost lines that are still draining trains!
- `City.GetLine(int index)` (`public Line GetLine(int index)`) — returns `lines[index]` (`0 <= index < City.LineCount`), including mothballed lines.
- `City.GetLineByIndex(int lineIndex)` (`public Line GetLineByIndex(int index)`) — returns only the active (`!IsMothballed`) `Line` occupying logical color slot `lineIndex` (`null` if free or mothballed).

### `Line` Members

| Member | Kind / Visibility | Type | Meaning |
| :--- | :--- | :--- | :--- |
| `Index` | `public` property | `int` | Logical line slot (`0..MaxLineCount-1`, `-1` if unassigned/error) |
| `ColorIndex` | `public` property | `int` | Color slot (`index` when `colorIndex == -1`) |
| `Type` | `public` property | `LinkType` | Link visual/track type |
| `IsMothballed` | `public` property | `bool` | `true` when line was deleted/ghosted and is draining onboard train passengers before final removal |
| `IsLooping` | `public` property | `bool` | `true` iff `loopingLinkStop != null` |
| `LoopStation` | `public` property | `Station` | Origin/seam station of loop (`null` if non-looping) |
| `LoopLinkStop` | `public` property | `LinkStop` | Loop seam `LinkStop` (`null` if non-looping) |
| `IsWaitingForLocomotive` | `public` property | `bool` | `true` if a newly built line has no train yet and awaits a free locomotive |
| `Count` | `public` property | `int` | Total `Link` count (`links.Count`, includes `DANGLING`/`PENDING`/`MOTHBALLED`) |
| `LiveLinkCount` | `public` property | `int` | Count of live links |
| `this[int index]` | `public` indexer | `Link` | Raw `links[index]` |
| `FirstLink` | `public` property | `Link` | First non-mothballed `Link` (or `LoopLinkStop.Link` on a loop) |
| `FirstActiveLink` | `public` property | `Link` | First `LinkState.ACTIVE` link in chain order (or `LoopLinkStop.Link`) |
| `LastActiveLink` | `public` property | `Link` | Last `LinkState.ACTIVE` link in chain order (or `LoopLinkStop.Link.PreviousLink`) |
| `Length` | `public` property | `float` | Sum of `Link.Length` over `LinkState.ACTIVE` links |
| `TrainCount` | `public` property | `int` | All trains on the line (`trains.Count`, including draining/pending) |
| `ActiveTrainCount` | `public` property | `int` | Trains where `Locomotive.State != RailcarState.MOTHBALLED` |
| `CarriageCount` | `public` property | `int` | Sum of non-mothballed carriages (`railcars[1..]`) across all trains on line |
| `GetTrain(int index)` | `public` method | `Train` | Returns `trains[index]` |
| `GetAssetCount(AssetType)` | `public` method | `int` | Returns active `Locomotive`, `Carriage`, or `Crossing` usage on this line (`0` if `IsMothballed`) |
| `MaxLocomotivesPerLine()` | `public` method | `int` | `City.GetMaxLocomotivesPerLine(Index)` (Classic `Constants.MAX_LOCOMOTIVES_PER_ROUTE = 4`) |

### `Link`, `LinkStop` & Ordered Station Sequence Extraction

- `LinkState` enum (`[Flags]`): `DANGLING = 1`, `PENDING = 2`, `ACTIVE = 4`, `MOTHBALLED = 8`, `CONCRETE = 6`, `LIVE = 12`.
- `Link` key members:
  - `Link.Id` (`public int Id`)
  - `Link.Line` (`public Line Line`)
  - `Link.State` (`public LinkState State`)
  - `Link.IsPendingMothball` (`public bool IsPendingMothball`) — marked during middle-edit replacement before `CommitMiddleEdit` turns it `MOTHBALLED`.
  - `Link.Start` / `Link.End` (`public LinkStop Start`, `public LinkStop End`)
  - `Link.PreviousLink` / `Link.NextLink` (`public Link`) — follow reciprocal `LinkConnection` hops along the route chain.
  - `Link.HasCrossing` (`public bool HasCrossing`) — `true` iff this segment currently reserves **1** `AssetType.Crossing` tunnel/bridge.
  - `Link.Length` (`public float Length`)
  - `Link.Footprint` (`public LinkFootprint Footprint`)
- `LinkStop.Station` (`public Station`), `LinkStop.Connection` (`public LinkConnection`), `LinkStop.OtherStop` (`public LinkStop`), `LinkStop.Direction` (`public CardinalDirection`), `LinkStop.Platform` (`public int`).
- **Ordered station sequence recipe**: Do **not** assume `line[0..Count-1]` is sorted in route order or excludes mothballed segments; traverse from `Link cur = line.FirstActiveLink;` using `cur = cur.NextLink;` while `cur != null`:
  - Emit `cur.Start.Station` first, then append `cur.End.Station` for each step.
  - Stop immediately if `cur.NextLink == line.FirstActiveLink` (for loops) or `cur.NextLink == null` (and keep a visited-link guard set against mid-edit graphs).
  - Note: a non-loop line with `N` stations has `N - 1` active `Link`s; a loop line with `N` distinct stations has `N` active `Link`s (`LastActiveLink.End.Station == FirstActiveLink.Start.Station`, with `line.IsLooping == true`).
  - Total tunnels/bridges consumed on a line = count active links with `link.HasCrossing && !link.IsPendingMothball && link.State != LinkState.MOTHBALLED`. Note that `line.GetAssetCount(AssetType.Crossing)` only skips `!link.HasCrossing || link.State == LinkState.MOTHBALLED` and does **not** filter `link.IsPendingMothball` (even though setting `IsPendingMothball = true` already released the crossing in `AssetDatabase`), so manual traversal with `!link.IsPendingMothball` is safer mid-edit.

### `Train`, `Locomotive` & `Railcar`

- `Train` members:
  - `Id` (`public int Id`), `Index` (`public int Index`), `Line` (`public Line Line`)
  - `Definition` (`public LocomotiveDefinition Definition`): `Type` (`AssetType.Locomotive` / `Shinkansen` / `Tram`), `Speed` (`float`), `Acceleration` (`float`), `Deceleration` (`float`), `Capacity` (`int`, default `Constants.DEFAULT_RAILCAR_CAPACITY = 6`).
  - `Direction` (`public LineDirection`: `FORWARDS = 0`, `BACKWARDS = 1`), `Link` (`public DirectedLink Link`), `RearLink` (`public Link RearLink`), `Distance` (`public float Distance`), `Stop` (`public Station Stop` — non-null only while stopped at a station).
  - `Locomotive` (`public Locomotive Locomotive` — always `railcars[0]`, inherits `Railcar`).
  - `RailcarCount` (`public int RailcarCount` — `1 + carriageCount`; index `0` is the `Locomotive`, indices `1..RailcarCount-1` are `Railcar` carriages).
  - `GetRailcar(int index)` (`public Railcar GetRailcar(int index)`).
  - `PeepCount` (`public int PeepCount`) / `NumPeeps()` (`public int NumPeeps()`) — total passengers across all railcars on the train.
  - `FreePeepCapacity` (`public int FreePeepCapacity`) — remaining capacity across all railcars.
  - Total capacity: **`Train` has no `Capacity` property**; compute `sum(train.GetRailcar(i).Capacity for i in 0..train.RailcarCount-1)`.
  - `HasMothballedRailcar` (`public bool HasMothballedRailcar`), `IsOnMothballedTrack` (`public bool IsOnMothballedTrack`).
  - `pendingDeletion` (`private bool pendingDeletion;`) — queued-return/drain flag set by `Train.MothballTrain()`.
  - `railcars` (`private List<Railcar> railcars;`).
- `Railcar` / `Locomotive` members:
  - `Id` (`public int Id`), `AssetType` (`public virtual AssetType AssetType` — `AssetType.Carriage` on carriages; overridden on `Locomotive` to `Train.Definition.Type`).
  - `State` (`public RailcarState State`: `ACTIVE = 0`, `PENDING = 1`, `MOTHBALLED = 2`).
  - `Capacity` (`public int Capacity => train.Definition.Capacity`).
  - `PeepCount` (`public int PeepCount`), `FreePeepCapacity` (`public int FreePeepCapacity`).
  - `Peeps` (`public List<Peep> Peeps`), `GetPeep(int peepIndex)` (`public Peep`).
  - `LinkedRailcar` (`public Railcar LinkedRailcar`) — links a draining `MOTHBALLED` railcar on the source line to its `PENDING` replacement clone on the destination line when relocating trains/carriages!
  - `IsReleaseScheduled` (`public bool IsReleaseScheduled`), `ReleaseTime` (`public float ReleaseTime`).
  - **Queued-return / move semantics**: `Train` can be draining either because `train.Line.IsMothballed == true`, or because `train.Locomotive.State == RailcarState.MOTHBALLED` (`pendingDeletion` or `Locomotive.LinkedRailcar != null`), or because a trailing carriage has `railcar.State == RailcarState.MOTHBALLED`.

---

## 5. Inventory & Weekly Upgrade Modal

### Inventory (`Metro.AssetDatabase`)

Accessed at `Game.AssetDatabase` (`public Metro.AssetDatabase AssetDatabase`, `namespace Metro`):

- `AssetType` enum (`global` namespace, `Count = 10`, indices `0..9`):
  `None = 0`, `Line = 1`, `Locomotive = 2`, `Shinkansen = 3`, `Tram = 4`, `Ferry = 5`, `Carriage = 6`, `Crossing = 7`, `Interchange = 8`, `Bridge = 9`, `Count = 10`.
- **Important**: `AssetType.Crossing` is the internal inventory bucket for **both** Tunnels (`CityDefinition.CrossingStyle.Tunnel`) and Bridges (`CityDefinition.CrossingStyle.Bridge`). `AssetType.Bridge` is only used as a UI icon key in HUD widgets.
- Key methods/properties on `Metro.AssetDatabase`:
  - `GetAvailableAssets(AssetType type)` -> `int` currently in player inventory (unused).
  - `GetTotalAssets(AssetType type)` -> `int` total owned (initial + weekly unlocks, whether deployed or in inventory).
  - `GetChosenAssets(AssetType type)` -> `int` chosen beyond initial city grant.
  - `AvailableLocomotiveAsset` -> `AssetType` (first available locomotive variant, or `AssetType.None`).
  - `SpecificAvailableLocomotiveAsset(List<AssetType> allowedTrains)` -> `AssetType`.
  - `AreAnyUpgradesAvailable` -> `bool`.
  - `ConsumeAsset(AssetType type, bool snap = false)` -> `bool` (decrements `availableAssets` unless `game.HasInfiniteAssets`, and updates `game.HudScreen.AssetPanel`).
  - `ReleaseAsset(AssetType type)` -> `void` (increments `availableAssets` and updates `AssetPanel`).
  - `AddAsset(AssetType type, bool isInitial = false)` -> `void`.

### Weekly Upgrade Modal (`NewAssetScreen`)

- Trigger flow in `Game.ScheduleAudioEvents` / `Game.Update()`: on a new week boundary, `pendingAssetCount++` and `scheduledScreen = GameScreen.NewAsset`; `Game.Update` opens `Game.Screen = GameScreen.NewAsset`, pausing gameplay (`IsPaused = true`).
- Modal instance is `Game.screens[(int)GameScreen.NewAsset]` (`private UI.Screen[] screens`), cast to `NewAssetScreen`:
  - `Game.PendingAssetCount` (`public int PendingAssetCount`), `Game.PendingLocomotiveCount` (`public int PendingLocomotiveCount`), `Game.ForcedAssets` (`public List<AssetType> ForcedAssets`).
- Two-phase panel flow inside `NewAssetScreen`:
  1. `HandleTransitionIn` initializes `locomotiveChoiceCount` and `assetChoiceCount`, then calls `BuildLocomotivePanel()` first (usually granting 1 `Locomotive` automatically if only one loco type exists, or offering a choice such as `Locomotive` vs `Shinkansen`), or skips directly to `BuildAssetsPanel()` / `BuildCustomAssetsPanel(...)` if no locomotive choice remains.
  2. Each selection passes through `NewAssetScreen.OnAsset(NewAssetButton button)`:
     - Calls `subjectGame.City.UnlockUpgrade(0, button.Type)` `button.Count` times (which invokes `Game.AssetDatabase.AddAsset(assetType)`).
     - Spawns `new AnimatedAssetCircle(subjectGame, button)` and calls `Profile.Instance.SetAssetChosen(button.Type)`.
     - Advances to remaining `locomotiveChoiceCount` -> remaining `assetChoiceCount` -> or sets `subjectGame.Screen = GameScreen.None` (whose `HandleTransitionOut` triggers `subjectGame.StartWeek()` and resumes).
- **Reading offered cards & accepting a choice**:
  - Reflect `NewAssetScreen.newAssetPanel` (`private NewAssetPanel newAssetPanel;`), then `NewAssetPanel.buttons` (`private List<NewAssetButton> buttons;`).
  - Each `NewAssetButton` exposes `public AssetType Type` and `public int Count`.
  - To pick card index `i` cleanly: verify `0 <= i < buttons.Count` and `buttons[i].IsEnabled`, then invoke `newAssetScreen.OnAsset(buttons[i])` (which is `public void OnAsset(NewAssetButton button)`!) — or invoke `buttons[i].Click()` if simulating the exact button press path through `NewAssetPanel.OnAsset(Button, FTouch)`. Calling `newAssetScreen.OnAsset(buttons[i])` directly runs the full upgrade unlock, HUD animation, multi-stage panel advance, and week resume cleanly!

---

## 6. Water / Tunnel Crossing Validation

Mini Metro checks crossings against `CityDefinition.ObstacleHull` (`public ObstacleHull ObstacleHull` on `CityDefinition`) using the segment's octilinear `LinkFootprint`:

- **Exact `City` geometry & intersection APIs**:
  - `City.HasCrossingSection(LinkFootprint footprint)` (`public bool HasCrossingSection(LinkFootprint footprint)`): returns `true` if `definition.ObstacleHull.DoesCross(footprint)` (or on UGC maps with `definition.RequireTunnelsInsideObstacles`, if either endpoint `Station.IsInsideObstacle` is `true`).
  - `City.GetFirstCrossing(LinkFootprint footprint, bool reverse, out Vector2 start)` (`public bool`): returns `true` and the entry point `start` of the first obstacle crossing along `footprint`.
  - `City.GetFirstCrossing(LineSegment line, out float position)` (`public bool`): tests a raw straight `Metro.LineSegment(startPos, endPos)` against `ObstacleHull`.
  - `City.GetCrossingSection(Track firstTrack, Track lastTrack, bool reverse)` (`public TrackSection`): extracts full crossing `TrackSection` on built tracks.
  - `City.GetLinkIntersections(LinkFootprint footprint, bool reverse)` (`public List<Vector2>`): returns all boundary intersection points.
- **Building a `LinkFootprint` cleanly**:
  - `LinkFootprint.Update(Station start, Station end)` computes the octilinear (horizontal/vertical + 45-degree diagonal) geometry between `start.Position` and `end.Position`, or `LinkFootprint.Update(Link link)` computes it using the configured `Start.PlatformPosition`, `Corner`, and `End.PlatformPosition`.
  - `Link.Configure()` calls `LinkBuilder.Instance.ConfigureLink(this)`, which finds valid platform/direction configurations, tests `City.GetFirstCrossing(link.Footprint, link.IsReversed, out var start)`, and either calls `link.ReserveCrossing()` or `link.ReleaseCrossing()`.
- **Important tunnel/bridge cost rule**:
  - A `Link` has `private bool hasCrossing;` (`public bool HasCrossing => hasCrossing;`).
  - Regardless of how many times a single `Link` intersects water/obstacles, **`Link.ReserveCrossing()` consumes at most ONE `AssetType.Crossing` per `Link`** (checking `game.AssetDatabase.GetAvailableAssets(AssetType.Crossing) > 0`).
  - However, note `Link.CanCross` edge case on `LinkState.DANGLING` vs concrete links: for concrete placement validation, `LinkBuilder.ConfigureLink` fails (`returns false`, rolling back `LinkStop.Station = null`) if crossing water while `GetAvailableAssets(AssetType.Crossing) <= 0`.
  - **Clean non-mutating probe**: Construct or reuse a `LinkFootprint`, call `footprint.Update(stationA, stationB)`, and call `city.HasCrossingSection(footprint)` (cheap read-only check; crossing cost is `1` if `true`, `0` if `false`). **Caveat on `LinkStop.PseudoStationSet`**: `LinkStop.PseudoStationSet` calls `link.Configure()` and restores `station = null` / `OtherStop.Configuration`, **without** calling `link.ReleaseCrossing()`. While `ReserveCrossing()` skips `ConsumeAsset` on `LinkState.DANGLING` links, on `LinkState.PENDING` (or `ACTIVE`) links it consumes a crossing in `AssetDatabase` and leaves `hasCrossing = true` leaked; never invoke `PseudoStationSet` on `LinkState.PENDING` links.

---

## 7. Safe Mutation / RPC Execution API

**Threading & pre-action guard requirement**: Unity, `Futile`, `GoTween` (`Go.to`), `FBatchedGeoLayer`, and `AudioSystem` are strictly **main-thread-only**. Any HTTP/named-pipe/socket listener in a MonoBehaviour plugin MUST enqueue commands into a thread-safe queue and execute all reads/mutations inside `Update()` / `LateUpdate()` (e.g., on the plugin `MonoBehaviour` after checking game/screen state). Before executing any gameplay mutation in `ActionExecutor`, guard on `!game.IsOver && game.OvercrowdedStation == null && !game.IsScreenChangeScheduled && !game.IsLocked` (plus `game.Screen == GameScreen.None` for live-board edits in `7.1`-`7.7`, or `game.Screen == GameScreen.NewAsset` for `7.8`), because `Game.ScheduleAudioEvents` sets `overcrowdedStation` and `scheduledScreen = GameScreen.GameOver` on the audio pulse before `Game.IsOver` flips `true`.

### 7.1 Pause, Resume & Speed Control

Preferred UI-synced controller: **`Game.HudScreen.ClockWidget`** (`ClockWidget`):

- `game.HudScreen.ClockWidget.TriggerVCRButtonPress(VCRButtonType.PAUSE)` -> pauses cleanly, tweens pause tint, fires `SignalVCRChange` -> `HudScreen.HandleVCRChange`, updates VCR button highlights.
- `game.HudScreen.ClockWidget.TriggerVCRButtonPress(VCRButtonType.PLAY)` -> resumes at `1x` (`TimeScale.Single`).
- `game.HudScreen.ClockWidget.TriggerVCRButtonPress(VCRButtonType.FAST_FORWARD)` -> resumes at `2x` (`TimeScale.Double`).
- Also available: `ClockWidget.TogglePause()`, `ClockWidget.IncreaseSpeed()`, `ClockWidget.DecreaseSpeed()` (guard on `!game.IsOver && game.OvercrowdedStation == null && !game.IsScreenChangeScheduled && !game.IsLocked && game.Screen == GameScreen.None`).

### 7.2 Create a New Line Between Stations `(stationA, stationB)`

Guard preconditions: `!game.IsOver && game.OvercrowdedStation == null && !game.IsScreenChangeScheduled && game.Screen == GameScreen.None && !game.IsLocked && stationA.CanBeConnected && stationB.CanBeConnected && stationA != stationB && game.AssetDatabase.GetAvailableAssets(AssetType.Line) > 0`.

1. Find a free slot via `int freeIndex = city.GetNextFreeLineIndex();` (checking `game.LineBuilder.IsAllocatedLine(freeIndex)` if scenario line-allocation restrictions apply; note `LineBuilder.allocatedLineIndices` is `private`), ensuring `freeIndex != -1`.
2. Create line: `Line line = city.AddLine(stationA, freeIndex);` (`City.AddLine` consumes `AssetType.Line`, allocates `links[0]` starting at `stationA` in `DANGLING` state). Check `line != null && line.Index != -1`; if `line.Index == -1`, call `line.Remove()` and abort.
3. Connect destination: `line[0].End.Station = stationB;`.
   - `LinkStop.Station` setter automatically invokes `link.Configure()` -> `LinkBuilder.Instance.ConfigureLink(link)` (computing octilinear path/platforms and calling `ReserveCrossing()`).
   - **Check `if (line[0].End.Station != stationB)`**: if validation/platforms/tunnels failed, `Station` rolled back to `null`! Call `line[0].Remove(); line.Remove();` (note `line.Remove()` releases the consumed `AssetType.Line` via `Line.Release()`).
4. Finalize: `line[0].RemoveAllErrors();`, then `line.Start(LineDirection.FORWARDS, null, startTrain: true, randomizeTrainPosition: false, tween: true);`.
   - **Critical decompilation catch (`Line.cs:985`)**: `Line.Start(..., startTrain: true)` ONLY spawns the initial locomotive automatically when `game.LineBuilder.LinkModify == LinkModifyType.NEW_LINE || game.Mode == GameMode.FAQ`! Since `LinkModify` is `private set` (`NONE` outside touch dragging), `Line.Start(..., startTrain: true)` marks `isWaitingForLocomotive = true` without spawning a train, so explicitly place a locomotive right after if one is free:
     ```csharp
     AssetType loco = game.AssetDatabase.SpecificAvailableLocomotiveAsset(city.Definition.AllowedTrains[line.Index]);
     if (loco != AssetType.None)
     {
         line.ApplyAsset(loco, null, LineDirection.FORWARDS, null, scheduleAudioEvent: true, duringSetup: false, tween: true);
     }
     ```
     (`Line.AddTrain` called inside `ApplyAsset` clears `isWaitingForLocomotive = false` automatically.)
5. Run UI/achievement/geometry sync:
   - `line.DirtyTracks();`
   - `game.OnLineEditComplete();` (recomputes achievement restrictions, `UpdateCrossings()`, and `GenerateTerminators()` across all active lines; `Line.Start` already called `city.HandleLineOpened(line)` -> `game.HudScreen.AssetPanel.SetLineOpen(line.Index, true)`).

### 7.3 Extend or Prepend an Endpoint of an Existing Line

Given an active, non-mothballed `Line` (non-looping!) and an endpoint `Link` (`first = line.FirstActiveLink` to prepend before `first.Start.Station`, or `last = line.LastActiveLink` to extend after `last.End.Station`):

- Check preconditions: `!line.IsMothballed && !line.IsLooping` (unless breaking loop first) and `!line.ContainsStation(targetStation)` (unless closing a loop back to the opposite endpoint!).
- **Extend (`LineDirection.FORWARDS` after `last`)** / **Prepend (`LineDirection.BACKWARDS` before `first`)**:
  1. Remove the old terminator at the endpoint station (`Station endpointStation = (direction == LineDirection.FORWARDS) ? endpointLink.End.Station : endpointLink.Start.Station; line.RemoveTerminator(endpointStation);`) so `AppendLink` sees `NextLink == null` / `PreviousLink == null`.
  2. `Link newLink = line.AppendLink(endpointLink, direction, connectLink: true);` (`direction = FORWARDS` connects `endpointLink.End` -> `newLink.Start`; `direction = BACKWARDS` connects `endpointLink.Start` -> `newLink.End`).
  3. Set the open endpoint (`LinkStop openStop = (direction == LineDirection.FORWARDS) ? newLink.End : newLink.Start; openStop.Station = targetStation;`).
  4. If rollback occurred (`openStop.Station != targetStation`), clean up: `newLink.Remove(); line.GenerateTerminators(); line.DirtyTracks(); game.OnLineEditComplete();` and fail cleanly.
  5. **Loop closure case**: if `targetStation` is the opposite terminal station of the same line (`line.ContainsStation(targetStation)` and `targetStation == oppositeEndStation`, with `line.LiveLinkCount >= 2` or valid non-backtracking triangle), pass `endLoopLink: newLink` to `line.Start(direction, endLoopLink: newLink, startTrain: false, randomizeTrainPosition: false, tween: true)` so `Line.Start` creates the reciprocal `LinkConnection` and sets `loopingLinkStop`! Otherwise call `line.Start(direction, null, startTrain: false, randomizeTrainPosition: false, tween: true);`.
  6. Sync: `line.ScheduleAudioEvent(LineChangeType.StationAdded, targetStation);`, `line.DirtyTracks();`, `game.OnLineEditComplete();`.
- **Removing/trimming an endpoint station** (when `game.CanRemoveTracks` is `true`, i.e., not `GameMode.EXTREME`): mothball the terminal `Link` (`terminalLink.State = LinkState.MOTHBALLED;`), break/remove its `LinkConnection` via `LinkStop` / `Line.Start(LineDirection.FORWARDS)`, `line.DirtyTracks()`, `game.OnLineEditComplete()` — note that if only 1 concrete link remains and you remove it, call `line.Mothball()` instead.

### 7.4 Insert an Intermediate Station into an Existing Segment Without Demolishing the Line

`City.ActivateStation` (`City.cs:2828-2874`) and `LineBuilder.HandleTouchEnded` provide the exact canonical pattern for splitting an active `Link existingLink` on `Line line` to insert `Station midStation` without mothballing the whole line or interrupting trains:

1. Check preconditions: `game.CanRemoveTracks` (note `City.ActivateStation` runs even in Extreme when a new station spawns under a track, whereas player edits require `game.CanRemoveTracks`), `!line.IsMothballed && midStation.CanBeConnected && !line.ContainsStation(midStation)`.
2. Mark old segment pending mothball (this immediately releases any crossing on `existingLink` so the replacement segments can reuse that tunnel/bridge!):
   ```csharp
   existingLink.IsPendingMothball = true;
   Link prevLink = existingLink.PreviousLink;
   Link nextLink = existingLink.NextLink;
   ```
3. Create two new `PENDING` links wrapping `existingLink`:
   ```csharp
   Link firstHalf = line.InsertLink(existingLink, LineDirection.FORWARDS);
   firstHalf.Start.Configuration = existingLink.Start.Configuration;
   firstHalf.End.Station = midStation;

   Link secondHalf = line.InsertLink(existingLink, LineDirection.BACKWARDS);
   secondHalf.End.Configuration = existingLink.End.Configuration;
   secondHalf.Start.Station = midStation;
   ```
4. Validate both ends configured cleanly (`firstHalf.End.Station == midStation && secondHalf.Start.Station == midStation`):
   - **If valid**:
     ```csharp
     LinkConnection.Connect(firstHalf, secondHalf);
     line.CommitMiddleEdit(prevLink, nextLink);
     line.Start(LineDirection.FORWARDS, null, startTrain: false, randomizeTrainPosition: false, tween: true);
     line.DirtyTracks();
     game.OnLineEditComplete();
     ```
     (`CommitMiddleEdit` handles loops if `IsLooping && LoopLinkStop.Link.IsPendingMothball`, turns `existingLink` into `LinkState.MOTHBALLED` so any train currently crossing it continues smoothly onto the new track, connects `prevLink -> firstHalf -> secondHalf -> nextLink`, and activates `firstHalf`/`secondHalf`!)
   - **If invalid** (for example segment needs 2 crossings and only 1 is available, or no platform exists):
     ```csharp
     firstHalf.Remove();
     secondHalf.Remove();
     line.RollbackMiddleEdit(); // resets existingLink.IsPendingMothball = false (re-consuming its crossing!)
     ```

### 7.5 Safely Remove a Line (Ghost/Draining State)

- Call **`line.Mothball()`** (`Line.cs:1031-1081`) — **never** call `line.Remove()` directly while trains have passengers!
- What `line.Mothball()` handles automatically:
  - Immediately releases `AssetType.Line` back to `Game.AssetDatabase` and updates HUD (`city.HandleLineMothballed(line)` -> `HudScreen.AssetPanel.SetLineOpen(line.Index, false)`), freeing the color slot immediately for new lines!
  - If `trains.Count == 0`, calls `Remove()` immediately and runs `game.AchievementHelper.HideMessage(); game.OnLineEditComplete();`.
  - If `trains.Count > 0`, sets `isMothballed = true`, switches all `links[i].State = LinkState.MOTHBALLED` (which immediately frees `AssetType.Crossing` tunnels/bridges!), removes terminators, cancels train reservations, schedules `LineChangeType.Mothballed` audio, and runs `game.OnLineEditComplete()`.
  - During subsequent `City.Update` ticks, trains finish unloading at stations (`Locomotive.State` remains `ACTIVE`), `Line.RemoveTrain` returns the locomotive and carriages back to `AssetDatabase`, and `City.Update` automatically removes and releases the drained ghost `Line`.

### 7.6 Assign, Move, or Return Locomotives & Carriages

- **Add a new locomotive from inventory onto an active line**:
  - Preconditions: `!line.IsMothballed`, `line.ActiveTrainCount < line.MaxLocomotivesPerLine()`, `game.AssetDatabase.GetAvailableAssets(locoType) > 0` (`locoType` in `{AssetType.Locomotive, AssetType.Shinkansen, AssetType.Tram}` and allowed on `line.Index`).
  - Call:
    `bool ok = line.ApplyAsset(locoType, position: null, direction: LineDirection.FORWARDS, existingAsset: null, scheduleAudioEvent: true, duringSetup: false, tween: true);`
  - Passing `position: null` automatically picks an appropriate link and balanced direction (`AddTrain` balances opposing directions on loops!), consumes `locoType` from `AssetDatabase`, updates HUD `AssetPanel`, calls `City.DirtyConnectivity()`, and schedules audio.
- **Attach a new carriage from inventory onto a line / specific train**:
  - Preconditions: `!line.IsMothballed && line.TrainCount > 0 && line.CarriageCount < Constants.MAX_CARRIAGES_PER_ROUTE` (`64`), `game.AssetDatabase.GetAvailableAssets(AssetType.Carriage) > 0`.
  - To let the line attach to the train with fewest railcars / closest to a `TrackPosition`:
    `bool ok = line.ApplyAsset(AssetType.Carriage, position: optionalTrackPos, direction: LineDirection.FORWARDS, existingAsset: null, scheduleAudioEvent: true, duringSetup: false, tween: true);`
  - Or to target a specific `Train targetTrain` directly via `Line.ApplyAsset` (which handles validation, `AssetDatabase.ConsumeAsset`, HUD sync, and audio automatically), pass its `TrackPosition` (`targetTrain.Link.GetTrackPosition(targetTrain.Distance)`) so `Line.GetClosestTrain` picks `targetTrain`. Note `targetTrain.AddCarriage(tween: true)` also works directly if paired manually with `game.AssetDatabase.ConsumeAsset(AssetType.Carriage)` and audio events, though `line.ApplyAsset(...)` is preferred.
- **Relocate (move) an existing locomotive or carriage between lines (Classic/Extreme-safe)**:
  - Check `game.CanMoveAssets` (`mode != GameMode.EXTREME`).
  - Follow `AssetBuilder` + `Railcar.HandleTouchMoved`: set `existingRailcar.State = RailcarState.MOTHBALLED;`, then call `targetLine.ApplyAsset(existingRailcar.AssetType, position, direction, existingAsset: existingRailcar, scheduleAudioEvent: true, duringSetup: false, tween: true);` (if `ApplyAsset` returns `false`, restore `existingRailcar.State = RailcarState.ACTIVE;`). Do **not** touch `AssetDatabase` when moving (`existingAsset != null` creates a linked `PENDING` clone via `LinkedRailcar` that activates once the old one drains).
- **Unassign / return a locomotive or carriage to inventory (IMPORTANT CAVEAT)**:
  - **Vanilla Classic Mini Metro has NO player UI action to return an individual locomotive or carriage back to inventory without mothballing its line.**
  - `Train.MothballTrain()` (`Train.cs:1283`) exists ONLY for Edit/Creative mode (`StationEditor.RemoveTrainsDelegate`) and archiving: it sets `GetRailcar(0).State = RailcarState.MOTHBALLED; pendingDeletion = true;`, and when the train finishes draining, `Line.RemoveTrain` checks `if (train.Locomotive.State == RailcarState.ACTIVE)` before calling `AssetDatabase.ReleaseAsset(...)`. Therefore calling `MothballTrain()` destroys the train **without** crediting `AssetDatabase`!
  - In Classic mode, either restrict train/carriage actions to **assign from inventory**, **move to another line**, and **demolish whole line (`line.Mothball()`)**, OR if forced to unassign in Sandbox/custom mode, call `line.Mothball()` (returns all assets cleanly) — never call `Train.MothballTrain()` expecting `AssetDatabase` inventory to increase.

### 7.7 Place an Interchange Upgrade on a Station

- Preconditions: `!station.IsInterchange && station.CanBeConnected && game.AssetDatabase.GetAvailableAssets(AssetType.Interchange) > 0`.
- Call: `bool ok = station.ApplyAsset(AssetType.Interchange, quiet: false);` (`Station.cs:1475` consumes `AssetType.Interchange` from `AssetDatabase`, sets `isInterchange = true`, plays tween + audio, updates `PeepContainer` capacity and layout).

### 7.8 Pick a Weekly Upgrade Card

- Preconditions: `game.Screen == GameScreen.NewAsset` (`NewAssetScreen` is open).
- Retrieve `NewAssetScreen screen = screens[(int)GameScreen.NewAsset] as NewAssetScreen;` (reflect `Game.screens`), reflect `screen.newAssetPanel`, reflect `panel.buttons` (`List<NewAssetButton>`).
- Inspect each `NewAssetButton`: `button.Type` (`AssetType`) and `button.Count` (`int`).
- Call `screen.OnAsset(buttons[choiceIndex])` on Unity's main thread. This executes `City.UnlockUpgrade(0, button.Type)`, updates `AssetDatabase` and HUD `AssetPanel`, triggers `AnimatedAssetCircle`, advances to any remaining locomotive/bonus choice panels, and when done sets `game.Screen = GameScreen.None` (calling `game.StartWeek()` and restoring normal gameplay/pause state).
