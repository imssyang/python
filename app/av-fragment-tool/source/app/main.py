# -*- coding: utf-8 -*-
import sys
from app.config import DataConfig
from app.barrage import BarrageData
from app.fragment import FragmentData


def main():
    config = DataConfig()
    print(f"[CONFIG]: {config.__dict__}")

    room_id = config.target["room"]
    from_time = config.target["from"]
    to_time = config.target["to"]
    barrage_data = BarrageData(
        config.barrage,
        str(room_id),
        from_time.strftime("%Y-%m-%d %H:%M:%S"),
        to_time.strftime("%Y-%m-%d %H:%M:%S"),
        config.target["weight"],
    )
    from_time_text = from_time.strftime("%Y%m%d-%H%M%S")
    to_time_text = to_time.strftime("%H%M%S")
    image_path = f"out/{room_id}-{from_time_text}-{to_time_text}.png"
    barrage_data.save(image_path)
    print(f"[IMAGE]: {image_path}")

    fragments = barrage_data.fragments()
    fragment_data = FragmentData(
        str(room_id), config.target["extra"], fragments, config.dataset[room_id]
    )
    fragment_data.truncate_all()


if __name__ == "__main__":
    sys.exit(main())
