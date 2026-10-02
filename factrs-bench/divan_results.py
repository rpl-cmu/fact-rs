import re


def parse_divan(text):
    """Read Divan's terminal summaries, converting times to milliseconds."""
    results = []
    method = None
    units = {"ns": 1e-6, "us": 1e-3, "\u00b5s": 1e-3, "\u03bcs": 1e-3, "ms": 1, "s": 1e3}
    for line in text.splitlines():
        columns = line.lstrip(" \u2502\u251c\u2570\u2500").split("\u2502")
        head = columns[0].strip()
        if head in {"factrs", "sophus", "tinysolver"}:
            method = head
            continue
        if ".g2o" not in head:
            continue
        if method is None or len(columns) < 4:
            raise ValueError(f"Invalid Divan result: {line}")
        filename, fastest = head.split(maxsplit=1)
        times = []
        for value in (fastest, columns[2], columns[1]):
            match = re.fullmatch(r"([\d.eE+-]+)\s*(ns|us|\u00b5s|\u03bcs|ms|s)", value.strip())
            if match is None:
                raise ValueError(f"Invalid Divan timing: {value}")
            times.append(float(match[1]) * units[match[2]])
        if not 0 <= times[0] <= times[1] <= times[2]:
            raise ValueError(f"Invalid Divan timing range: {line}")
        # Summary triplets preserve median bars and min/max whiskers, not sample distributions.
        results.extend(
            {"method": method, "filename": filename, "time": time}
            for time in times
        )
    if not results:
        raise ValueError("No Divan results found; capture benchmark output first.")
    return results


if __name__ == "__main__":
    rows = parse_divan(
        "g2o fastest | slowest | median\n"
        "\u251c\u2500 factrs \u2502 \u2502 \u2502\n"
        "  \u2570\u2500 M3500.g2o 1000 ns \u2502 2 ms \u2502 1 \u00b5s \u2502 1 ms\n"
        "\u2570\u2500 sophus \u2502 \u2502 \u2502\n"
        "  \u2570\u2500 sphere2500.g2o 500 us \u2502 1 s \u2502 2 ms \u2502 3 ms\n"
    )
    assert [row["time"] for row in rows] == [0.001, 0.001, 2, 0.5, 2, 1000]
    assert rows[0]["method"] == "factrs" and rows[3]["method"] == "sophus"
    for invalid in ("", "factrs\nM3500.g2o 1 ms \u2502 bad \u2502 2 ms \u2502 2 ms"):
        try:
            parse_divan(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid output was accepted")
