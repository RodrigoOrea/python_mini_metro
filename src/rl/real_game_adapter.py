"""Adapter mapping ``bridge-rpc-v1`` ``Snapshot`` objects to ``OpenMetroView`` and ``rorea_ml_open``."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from rl.bridge_protocol import (
    EntityId,
    OfferAssetKind,
    OpenAction,
    PickUpgradeAction,
    ScreenKind,
    Snapshot,
    StationShape,
    TrainSnapshot,
)
from rl.objective_ml import BEST_PHASE1_PARAMS, RoreaMLParams
from rl.rorea_ml_open import (
    OpenLine,
    OpenMetroView,
    OpenPlannerState,
    OpenStation,
    choose_open,
)

SHAPE_TO_ROREA: dict[str, str] = {
    StationShape.CIRCLE.value: "CIRCLE",
    StationShape.TRIANGLE.value: "TRIANGLE",
    StationShape.SQUARE.value: "RECT",
    StationShape.CROSS.value: "CROSS",
    StationShape.DIAMOND.value: "DIAMOND",
    StationShape.EGG.value: "EGG",
    StationShape.GEM.value: "GEM",
    StationShape.PENTAGON.value: "PENTAGON",
    StationShape.STAR.value: "STAR",
    StationShape.WEDGE.value: "WEDGE",
}


def map_shape_to_rorea(shape: str) -> str:
    """Translate a ``bridge-rpc-v1`` shape token to its ``rorea_ml`` shape token."""
    return SHAPE_TO_ROREA.get(shape, shape)


def snapshot_to_open_view(
    snapshot: Snapshot,
    decision: int,
    *,
    max_locos_per_line: int = 4,
    supports_insert_station: bool = True,
    supports_relocate_loco: bool = True,
    can_remove_tracks: bool = True,
    include_ghost_lines: bool = False,
) -> OpenMetroView:
    """Convert a validated ``Snapshot`` into a non-mutating ``OpenMetroView``."""
    scale = float(snapshot.camera.world_to_ref_scale)
    open_stations = tuple(
        OpenStation(
            id=st.id,
            x=float(st.x) * scale,
            y=float(st.y) * scale,
            shape=map_shape_to_rorea(st.shape),
            waiting_dest_shapes=tuple(
                map_shape_to_rorea(wp.dest_shape) for wp in st.waiting_passengers
            ),
            waiting_wait_ms=tuple(int(wp.wait_ms) for wp in st.waiting_passengers),
            peep_capacity=int(st.peep_capacity),
            is_interchange=bool(st.is_interchange),
            overload_progress=float(st.overload_progress),
        )
        for st in snapshot.stations
    )

    trains_by_line: dict[EntityId, list[TrainSnapshot]] = defaultdict(list)
    for tr in snapshot.trains:
        trains_by_line[tr.line_id].append(tr)

    open_lines: list[OpenLine] = []
    for ln in snapshot.lines:
        if ln.is_ghost and not include_ghost_lines:
            continue
        line_trains = trains_by_line.get(ln.id, ())
        non_ghost = [tr for tr in line_trains if not tr.is_ghost]
        active_train_count = sum(
            1
            for tr in non_ghost
            if not tr.is_unassignment_queued
            and not tr.is_pending
            and not tr.is_waiting_for_locomotive
        )
        has_pending_train = any(
            tr.is_pending or tr.is_waiting_for_locomotive or tr.is_unassignment_queued
            for tr in line_trains
        )
        carriage_count = sum(int(tr.carriages) for tr in non_ghost)
        total_capacity = sum(int(tr.capacity) for tr in non_ghost)
        onboard_count = sum(int(tr.onboard_count) for tr in non_ghost)
        can_relocate_donor = active_train_count >= 2 and not has_pending_train
        open_lines.append(
            OpenLine(
                id=ln.id,
                color_index=int(ln.color_index),
                station_ids=tuple(ln.station_ids),
                is_looped=bool(ln.is_looped),
                is_ghost=bool(ln.is_ghost),
                tunnels_used=int(ln.tunnels_used),
                active_train_count=active_train_count,
                carriage_count=carriage_count,
                total_capacity=total_capacity,
                onboard_count=onboard_count,
                can_relocate_donor=can_relocate_donor,
                has_pending_train=has_pending_train,
            )
        )

    inv = snapshot.inventory
    return OpenMetroView(
        decision=int(decision),
        stations=open_stations,
        lines=tuple(open_lines),
        lines_available=int(inv.lines_available),
        locos_available=int(inv.locos_available),
        carriages_available=int(inv.carriages_available),
        tunnels_available=int(inv.tunnels_available),
        interchanges_available=int(inv.interchanges_available),
        crossing_cost_fn=snapshot.crossing_cost,
        max_locos_per_line=int(max_locos_per_line),
        supports_insert_station=bool(supports_insert_station),
        supports_relocate_loco=bool(supports_relocate_loco),
        can_remove_tracks=bool(can_remove_tracks),
    )


def choose_weekly_upgrade(
    snapshot: Snapshot,
    *,
    max_lines_cap: int = 7,
) -> PickUpgradeAction:
    """Pick a weekly upgrade offer from ``snapshot.weekly_offers`` by priority."""
    offers = snapshot.weekly_offers
    if not offers:
        raise ValueError("Cannot choose weekly upgrade when weekly_offers is empty")

    inv = snapshot.inventory
    if inv.lines_total < max_lines_cap:
        for offer in offers:
            if offer.kind == OfferAssetKind.LINE.value:
                return PickUpgradeAction(offer_index=offer.index)

    if inv.tunnels_available <= 1:
        for offer in offers:
            if offer.kind == OfferAssetKind.CROSSING.value:
                return PickUpgradeAction(offer_index=offer.index)

    for loco_kind in (
        OfferAssetKind.SHINKANSEN.value,
        OfferAssetKind.LOCOMOTIVE.value,
        OfferAssetKind.TRAM.value,
    ):
        for offer in offers:
            if offer.kind == loco_kind:
                return PickUpgradeAction(offer_index=offer.index)

    for offer in offers:
        if offer.kind == OfferAssetKind.CARRIAGE.value:
            return PickUpgradeAction(offer_index=offer.index)

    for offer in offers:
        if offer.kind == OfferAssetKind.INTERCHANGE.value:
            return PickUpgradeAction(offer_index=offer.index)

    return PickUpgradeAction(offer_index=offers[0].index)


@dataclass
class RealGameAgent:
    """Stateful ``rorea_ml_open`` policy wrapper for ``bridge-rpc-v1`` snapshots."""

    params: RoreaMLParams = BEST_PHASE1_PARAMS
    state: OpenPlannerState = field(default_factory=OpenPlannerState)
    decision: int = 0
    max_lines_cap: int = 7
    max_locos_per_line: int = 4
    supports_insert_station: bool = True
    supports_relocate_loco: bool = True
    can_remove_tracks: bool = True

    def reset(self) -> None:
        """Reset planner state and decision counter."""
        self.state.reset()
        self.decision = 0

    def choose_for_snapshot(
        self,
        snapshot: Snapshot,
        decision: int | None = None,
    ) -> OpenAction | None:
        """Choose the next ``OpenAction`` for ``snapshot``, or ``None`` if inactive."""
        if snapshot.screen in (
            ScreenKind.MENU,
            ScreenKind.GAME_OVER,
            ScreenKind.OTHER,
        ):
            return None

        if snapshot.screen == ScreenKind.WEEKLY_UPGRADE:
            return choose_weekly_upgrade(
                snapshot,
                max_lines_cap=self.max_lines_cap,
            )

        if snapshot.screen in (ScreenKind.PLAYING, ScreenKind.PAUSED):
            dec = self.decision if decision is None else int(decision)
            if decision is None:
                self.decision += 1
            view = snapshot_to_open_view(
                snapshot,
                dec,
                max_locos_per_line=self.max_locos_per_line,
                supports_insert_station=self.supports_insert_station,
                supports_relocate_loco=self.supports_relocate_loco,
                can_remove_tracks=self.can_remove_tracks,
                include_ghost_lines=self.state.pending_rebuild is not None,
            )
            return choose_open(view, self.state, self.params)

        return None
