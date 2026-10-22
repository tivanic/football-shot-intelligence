from __future__ import annotations

import csv
import math
from collections import deque
from pathlib import Path

import cv2
from ultralytics import YOLO


INPUT_PATH = Path(
    "data/input/zidane_goal.mp4"
)

OUTPUT_PATH = Path(
    "data/output/zidane_ball_analysis.mp4"
)

CSV_PATH = Path(
    "data/output/zidane_ball_tracking.csv"
)

MODEL_NAME = "models/football_detector.pt"

BALL_CLASS_ID = 0
PLAYER_CLASS_ID = 1
REFEREE_CLASS_ID = 2

INFERENCE_CONFIDENCE = 0.05

PERSON_MIN_CONFIDENCE = 0.25
BALL_MIN_CONFIDENCE = 0.05

IMAGE_SIZE = 1280

TRAIL_LENGTH = 20


def box_center(
    xyxy: list[float],
) -> tuple[float, float]:
    x1, y1, x2, y2 = xyxy

    return (
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0,
    )


def player_foot_point(
    xyxy: list[float],
) -> tuple[float, float]:
    """
    Approximate player position using
    the bottom centre of the bounding box.
    """

    x1, _, x2, y2 = xyxy

    return (
        (x1 + x2) / 2.0,
        y2,
    )


def distance(
    point_a: tuple[float, float],
    point_b: tuple[float, float],
) -> float:
    return math.hypot(
        point_a[0] - point_b[0],
        point_a[1] - point_b[1],
    )


def main() -> None:
    model = YOLO(
        MODEL_NAME
    )

    capture = cv2.VideoCapture(
        str(INPUT_PATH)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open {INPUT_PATH}"
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

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    writer = cv2.VideoWriter(
        str(OUTPUT_PATH),
        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),
        fps,
        (
            width,
            height,
        ),
    )

    ball_trail = deque(
        maxlen=TRAIL_LENGTH
    )

    previous_ball = None
    previous_ball_frame = None

    frame_number = 0

    rows = []

    while True:
        success, frame = (
            capture.read()
        )

        if not success:
            break

        frame_number += 1

        timestamp = (
            frame_number / fps
        )

        results = model.predict(
            frame,
            classes=[
                PLAYER_CLASS_ID,
                BALL_CLASS_ID,
            ],
            conf=INFERENCE_CONFIDENCE,
            imgsz=IMAGE_SIZE,
            verbose=False,
        )

        result = results[0]

        persons = []
        balls = []

        if result.boxes is not None:

            for box in result.boxes:

                class_id = int(
                    box.cls[0]
                )

                confidence = float(
                    box.conf[0]
                )

                xyxy = (
                    box.xyxy[0]
                    .cpu()
                    .tolist()
                )

                if (
                    class_id
                    == PLAYER_CLASS_ID
                    and confidence
                    >= PERSON_MIN_CONFIDENCE
                ):
                    persons.append(
                        {
                            "bbox": xyxy,
                            "confidence":
                                confidence,
                        }
                    )

                if (
                    class_id
                    == BALL_CLASS_ID
                    and confidence
                    >= BALL_MIN_CONFIDENCE
                ):
                    balls.append(
                        {
                            "bbox": xyxy,
                            "confidence":
                                confidence,
                        }
                    )

        # -------------------------------------------------
        # Select best ball detection
        # -------------------------------------------------

        best_ball = None

        if balls:
            best_ball = max(
                balls,
                key=lambda item:
                    item[
                        "confidence"
                    ],
            )

        ball_x = None
        ball_y = None
        ball_confidence = None

        speed_pixels_per_second = None

        nearest_player_distance = None
        nearest_player_bbox = None

        if best_ball is not None:

            ball_confidence = (
                best_ball[
                    "confidence"
                ]
            )

            ball_x, ball_y = (
                box_center(
                    best_ball[
                        "bbox"
                    ]
                )
            )

            current_ball = (
                ball_x,
                ball_y,
            )

            ball_trail.append(
                current_ball
            )

            # ---------------------------------------------
            # Ball speed
            # ---------------------------------------------

            if (
                previous_ball
                is not None
                and previous_ball_frame
                is not None
            ):

                frame_difference = (
                    frame_number
                    - previous_ball_frame
                )

                if frame_difference <= 10:

                    travelled = distance(
                        current_ball,
                        previous_ball,
                    )

                    seconds = (
                        frame_difference
                        / fps
                    )

                    speed_pixels_per_second = (
                        travelled
                        / seconds
                    )

            previous_ball = (
                current_ball
            )

            previous_ball_frame = (
                frame_number
            )

            # ---------------------------------------------
            # Player closest to ball
            # ---------------------------------------------

            player_candidates = []

            for person in persons:

                foot_point = (
                    player_foot_point(
                        person[
                            "bbox"
                        ]
                    )
                )

                player_distance = (
                    distance(
                        current_ball,
                        foot_point,
                    )
                )

                player_candidates.append(
                    (
                        player_distance,
                        person,
                    )
                )

            if player_candidates:

                (
                    nearest_player_distance,
                    nearest_player,
                ) = min(
                    player_candidates,
                    key=lambda item:
                        item[0],
                )

                nearest_player_bbox = (
                    nearest_player[
                        "bbox"
                    ]
                )

        # -------------------------------------------------
        # Draw players
        # -------------------------------------------------

        for person in persons:

            x1, y1, x2, y2 = [
                int(value)
                for value
                in person[
                    "bbox"
                ]
            ]

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (160, 160, 160),
                1,
            )

        # -------------------------------------------------
        # Highlight nearest player
        # -------------------------------------------------

        if nearest_player_bbox is not None:

            x1, y1, x2, y2 = [
                int(value)
                for value
                in nearest_player_bbox
            ]

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 255),
                3,
            )

            cv2.putText(
                frame,
                "Nearest player",
                (
                    x1,
                    max(
                        25,
                        y1 - 10,
                    ),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # -------------------------------------------------
        # Ball + trajectory
        # -------------------------------------------------

        if (
            ball_x is not None
            and ball_y is not None
        ):

            cv2.circle(
                frame,
                (
                    int(ball_x),
                    int(ball_y),
                ),
                10,
                (0, 0, 255),
                3,
            )

            cv2.putText(
                frame,
                (
                    f"BALL "
                    f"{ball_confidence:.2f}"
                ),
                (
                    int(ball_x) + 15,
                    int(ball_y) - 15,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        trail_points = list(
            ball_trail
        )

        for index in range(
            1,
            len(trail_points),
        ):

            start = trail_points[
                index - 1
            ]

            end = trail_points[
                index
            ]

            cv2.line(
                frame,
                (
                    int(start[0]),
                    int(start[1]),
                ),
                (
                    int(end[0]),
                    int(end[1]),
                ),
                (0, 0, 255),
                2,
            )

        # -------------------------------------------------
        # Information overlay
        # -------------------------------------------------

        cv2.putText(
            frame,
            f"Frame: {frame_number}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            f"Time: {timestamp:.2f}s",
            (30, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if (
            speed_pixels_per_second
            is not None
        ):

            cv2.putText(
                frame,
                (
                    "Ball speed: "
                    f"{speed_pixels_per_second:.0f} "
                    "px/s"
                ),
                (30, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        writer.write(
            frame
        )

        rows.append(
            {
                "frame":
                    frame_number,

                "time_seconds":
                    timestamp,

                "ball_detected":
                    best_ball
                    is not None,

                "ball_x":
                    ball_x,

                "ball_y":
                    ball_y,

                "ball_confidence":
                    ball_confidence,

                "ball_speed_px_s":
                    speed_pixels_per_second,

                "nearest_player_distance_px":
                    nearest_player_distance,

                "persons_detected":
                    len(
                        persons
                    ),
            }
        )

        if (
            frame_number
            % 100
            == 0
        ):
            print(
                f"Processed "
                f"{frame_number} frames"
            )

    capture.release()
    writer.release()

    # -----------------------------------------------------
    # Save structured tracking data
    # -----------------------------------------------------

    with CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer_csv = (
            csv.DictWriter(
                file,
                fieldnames=rows[
                    0
                ].keys(),
            )
        )

        writer_csv.writeheader()

        writer_csv.writerows(
            rows
        )

    print()
    print("Finished.")
    print(
        f"Video: {OUTPUT_PATH}"
    )
    print(
        f"Data:  {CSV_PATH}"
    )


if __name__ == "__main__":
    main()