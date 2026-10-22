from __future__ import annotations

from pathlib import Path

import cv2
import pandas as pd


CSV_PATH = Path(
    "data/output/zidane_ball_tracking.csv"
)

VIDEO_PATH = Path(
    "data/input/zidane_goal.mp4"
)

OUTPUT_DIR = Path(
    "data/output/shots"
)


CONTACT_DISTANCE_PX = 80.0
DEPARTURE_DISTANCE_PX = 120.0

MIN_DEPARTURE_SPEED_PX_S = 800.0
MAX_REASONABLE_SPEED_PX_S = 5000.0

MAX_CONFIRM_GAP_FRAMES = 5


# ---------------------------------------------------------
# Shot-specific assumptions for this attacking sequence
# ---------------------------------------------------------

ATTACKING_DIRECTION = "right"

# Require contact to happen in the more advanced
# part of the current broadcast frame.
ADVANCED_ZONE_X_RATIO = 0.45

# Ball must clearly move toward the attacking side.
MIN_FORWARD_DISPLACEMENT_PX = 100.0


CLIP_SECONDS_BEFORE = 3.0
CLIP_SECONDS_AFTER = 4.0


def detect_kicks(
    data: pd.DataFrame,
) -> list[dict]:
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

        ball_x = row[
            "ball_x"
        ]

        ball_y = row[
            "ball_y"
        ]

        if (
            pd.isna(distance)
            or pd.isna(ball_x)
            or pd.isna(ball_y)
        ):
            continue

        # -------------------------------------------------
        # Contact candidate
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

                    "ball_x":
                        float(
                            ball_x
                        ),

                    "ball_y":
                        float(
                            ball_y
                        ),
                }

        # -------------------------------------------------
        # Confirm departure
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

                        "contact_ball_x":
                            contact_candidate[
                                "ball_x"
                            ],

                        "contact_ball_y":
                            contact_candidate[
                                "ball_y"
                            ],

                        "confirmation_frame":
                            frame,

                        "confirmation_time":
                            float(
                                row[
                                    "time_seconds"
                                ]
                            ),

                        "confirmation_ball_x":
                            float(
                                ball_x
                            ),

                        "confirmation_ball_y":
                            float(
                                ball_y
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

    return events


def classify_shots(
    events: list[dict],
    frame_width: int,
) -> list[dict]:

    advanced_x = (
        frame_width
        * ADVANCED_ZONE_X_RATIO
    )

    classified = []

    for event in events:

        dx = (
            event[
                "confirmation_ball_x"
            ]
            - event[
                "contact_ball_x"
            ]
        )

        if (
            ATTACKING_DIRECTION
            == "right"
        ):
            forward_displacement = dx

        else:
            forward_displacement = -dx

        advanced_position = (
            event[
                "contact_ball_x"
            ]
            >= advanced_x
        )

        moving_toward_goal_side = (
            forward_displacement
            >= MIN_FORWARD_DISPLACEMENT_PX
        )

        is_shot = (
            advanced_position
            and moving_toward_goal_side
        )

        classified.append(
            {
                **event,

                "forward_displacement":
                    float(
                        forward_displacement
                    ),

                "advanced_position":
                    bool(
                        advanced_position
                    ),

                "moving_toward_goal_side":
                    bool(
                        moving_toward_goal_side
                    ),

                "is_shot":
                    bool(
                        is_shot
                    ),
            }
        )

    return classified


def save_clip(
    event: dict,
    shot_number: int,
) -> Path:

    capture = cv2.VideoCapture(
        str(
            VIDEO_PATH
        )
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open "
            f"{VIDEO_PATH}"
        )

    fps = capture.get(
        cv2.CAP_PROP_FPS
    )

    width = int(
        capture.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        capture.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    total_frames = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    contact_frame = int(
        event[
            "contact_frame"
        ]
    )

    start_frame = max(
        0,
        int(
            contact_frame
            - CLIP_SECONDS_BEFORE
            * fps
        ),
    )

    end_frame = min(
        total_frames - 1,
        int(
            contact_frame
            + CLIP_SECONDS_AFTER
            * fps
        ),
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / (
            f"shot_"
            f"{shot_number:03d}_"
            f"{event['contact_time']:.2f}s"
            f".mp4"
        )
    )

    writer = cv2.VideoWriter(
        str(
            output_path
        ),
        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),
        fps,
        (
            width,
            height,
        ),
    )

    capture.set(
        cv2.CAP_PROP_POS_FRAMES,
        start_frame,
    )

    current_frame = (
        start_frame
    )

    while (
        current_frame
        <= end_frame
    ):

        success, frame = (
            capture.read()
        )

        if not success:
            break

        if (
            abs(
                current_frame
                - contact_frame
            )
            <= int(
                fps * 0.35
            )
        ):

            cv2.putText(
                frame,
                "SHOT DETECTED",
                (
                    40,
                    70,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.4,
                (
                    0,
                    0,
                    255,
                ),
                4,
                cv2.LINE_AA,
            )

        writer.write(
            frame
        )

        current_frame += 1

    capture.release()
    writer.release()

    return output_path


def main() -> None:

    data = pd.read_csv(
        CSV_PATH
    )

    capture = cv2.VideoCapture(
        str(
            VIDEO_PATH
        )
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open "
            f"{VIDEO_PATH}"
        )

    frame_width = int(
        capture.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    capture.release()

    kicks = detect_kicks(
        data
    )

    events = classify_shots(
        kicks,
        frame_width,
    )

    print()
    print(
        "EVENT CANDIDATES"
    )
    print(
        "================"
    )

    shot_number = 0

    for number, event in enumerate(
        events,
        start=1,
    ):

        label = (
            "SHOT"
            if event[
                "is_shot"
            ]
            else "KICK"
        )

        print()
        print(
            f"#{number} "
            f"{label}"
        )

        print(
            "Time: "
            f"{event['contact_time']:.2f}s"
        )

        print(
            "Forward displacement: "
            f"{event['forward_displacement']:.1f}px"
        )

        print(
            "Advanced position: "
            f"{event['advanced_position']}"
        )

        if event[
            "is_shot"
        ]:

            shot_number += 1

            clip_path = save_clip(
                event,
                shot_number,
            )

            print(
                f"Clip: "
                f"{clip_path}"
            )

    print()
    print(
        f"Shots detected: "
        f"{shot_number}"
    )


if __name__ == "__main__":
    main()