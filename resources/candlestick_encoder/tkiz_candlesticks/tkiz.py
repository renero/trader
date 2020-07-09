from tkiz_candle import TkizCandle


def inc_pos(by=0.5):
    global pos
    rv = pos
    pos += by
    return rv


def paint():
    global pos

    TkizCandle(0.500, 0.01).draw(inc_pos())
    TkizCandle(0.750, 0.01).draw(inc_pos())
    TkizCandle(0.250, 0.01).draw(inc_pos())
    TkizCandle(1.000, 0.01).draw(inc_pos())
    TkizCandle(0.000, 0.01).draw(inc_pos())

    TkizCandle(0.500, 0.10).draw(inc_pos())
    TkizCandle(0.750, 0.10).draw(inc_pos())
    TkizCandle(0.250, 0.10).draw(inc_pos())
    TkizCandle(0.950, 0.10).draw(inc_pos())
    TkizCandle(0.050, 0.10).draw(inc_pos())

    TkizCandle(0.500, 0.25).draw(inc_pos())
    TkizCandle(0.750, 0.25).draw(inc_pos())
    TkizCandle(0.250, 0.25).draw(inc_pos())
    TkizCandle(0.875, 0.25).draw(inc_pos())
    TkizCandle(0.125, 0.25).draw(inc_pos())

    TkizCandle(0.500, 0.50).draw(inc_pos())
    TkizCandle(0.650, 0.50).draw(inc_pos())
    TkizCandle(0.350, 0.50).draw(inc_pos())
    TkizCandle(0.750, 0.50).draw(inc_pos())
    TkizCandle(0.250, 0.50).draw(inc_pos())

    TkizCandle(0.500, 0.75).draw(inc_pos())
    TkizCandle(0.575, 0.75).draw(inc_pos())
    TkizCandle(0.425, 0.75).draw(inc_pos())
    TkizCandle(0.625, 0.75).draw(inc_pos())
    TkizCandle(0.375, 0.75).draw(inc_pos())

    TkizCandle(0.500, 1.00).draw(inc_pos())


if __name__ == "__main__":
    pos = 0
    paint()