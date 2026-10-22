from __future__ import annotations

from pathlib import Path

import pandas as pd


CSV_PATH = Path(
    "data/output/zidane_ball_tracking.csv"
)

CONTACT_DISTANCE_PX = 80.0
DEPARTURE_DISTANCE_PX = 120.0

MIN_DEPARTURE_SPEED_PX_S = 800.0

MAX_CONFIRM_GAP_FRAMES = 5

MAX_REASONABLE_SPEED_PX_S = 5000.0


def main() -> None:
    data = pd.read_csv(
        CSV_PATH
    )

    contact_candidate = None
    events = []

    for _, row in data.iterrows():

        frame = int(
            row["frame"]
        )

        detected = bool(
            row["ball_detected"]
        )

        if not detected:
            continue

        distance = row[
            "nearest_player_distance_px"
        ]

        speed = row[
            "ball_speed_px_s"
        ]

        if pd.isna(
            distance
        ):
            continue

        # -------------------------------------------------
        # Potential ball contact
        # -------------------------------------------------

        if (
            distance
            <= CONTACT_DISTANCE_PX
            and (
                pd.isna(speed)
                or speed
                <= MAX_REASONABLE_SPEED_PX_S
            )
        ):

            if (
                contact_candidate
                is None
                or distance
                < contact_candidate[
                    "distance"
                ]
            ):
                contact_candidate = {
                    "frame":
                        frame,

                    "time":
                        float(
                            row[
                                "time_seconds"
                            ]
                        ),

                    "distance":
                        float(
                            distance
                        ),
                }

        # -------------------------------------------------
        # Confirm that ball moves away
        # -------------------------------------------------

        if contact_candidate is not None:

            frame_gap = (
                frame
                - contact_candidate[
                    "frame"
                ]
            )

            if (
                1
                <= frame_gap
                <= MAX_CONFIRM_GAP_FRAMES
                and distance
                >= DEPARTURE_DISTANCE_PX
                and not pd.isna(
                    speed
                )
                and speed
                >= MIN_DEPARTURE_SPEED_PX_S
                and speed
                <= MAX_REASONABLE_SPEED_PX_S
            ):

                events.append(
                    {
                        "contact_frame":
                            contact_candidate[
                                "frame"
                            ],

                        "contact_time":
                            contact_candidate[
                                "time"
                            ],

                        "contact_distance":
                            contact_candidate[
                                "distance"
                            ],

                        "confirmation_frame":
                            frame,

                        "confirmation_time":
                            float(
                                row[
                                    "time_seconds"
                                ]
                            ),

                        "departure_speed":
                            float(
                                speed
                            ),
                    }
                )

                contact_candidate = None

            elif (
                frame_gap
                > MAX_CONFIRM_GAP_FRAMES
            ):
                contact_candidate = None

    print()
    print(
        "KICK CANDIDATES"
    )
    print(
        "==============="
    )

    for number, event in enumerate(
        events,
        start=1,
    ):

        print()
        print(
            f"#{number}"
        )

        print(
            "Contact: "
            f"{event['contact_time']:.2f}s "
            f"(frame "
            f"{event['contact_frame']})"
        )

        print(
            "Distance: "
            f"{event['contact_distance']:.1f}px"
        )

        print(
            "Departure speed: "
            f"{event['departure_speed']:.0f}px/s"
        )

    print()
    print(
        f"Total candidates: "
        f"{len(events)}"
    )


if __name__ == "__main__":
    main()