from pathlib import Path
import json
import shutil


def write_single_csv(df, output_file: str | Path) -> None:
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    tmp_dir = output_file.with_suffix("")

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    if output_file.exists():
        output_file.unlink()

    df.coalesce(1).write.mode("overwrite").option("header", True).csv(str(tmp_dir))

    part_file = next(tmp_dir.glob("part-*.csv"))
    shutil.move(str(part_file), str(output_file))
    shutil.rmtree(tmp_dir)


def save_json(data: dict, output_file: str | Path) -> None:
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)