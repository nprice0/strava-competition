def format_time(seconds):
    mins, sec = divmod(seconds, 60)
    return f"{mins}m {sec}s"