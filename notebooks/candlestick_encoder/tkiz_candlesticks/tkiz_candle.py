import string


class TkizCandle:

    def __init__(self, center_pos, size):
        self.center_pos = center_pos
        self.size = size

    def draw(self, xpos, max_size=3.0, width=0.2, sep=0.5):
        """
        Pinta un candlestick con TKIZ en la posición X dada por xpos.
        :param xpos: Posición en el eje X en el que queremos pintarlo
        :param max_size: El tamaño máximo (anchura Y) del área de dibujo.
        :param width: El ancho (X) del candlestick
        :param sep: La separación entre candlestick consecutivos.
        :return: None
        """
        center = max_size * self.center_pos
        shadow_pos = xpos+(width/2.0)
        rect_up = center + ((max_size * self.size) / 2.0)
        rect_down = center - ((max_size * self.size) / 2.0)
        index = int(xpos / sep)
        label = list(string.ascii_uppercase)[index]

        print("\\filldraw[thick, fill=white] ({:.3f}, {:.3f}) rectangle ({:.3f}, {:.3f});".
              format(xpos, rect_down, xpos+width, rect_up))
        print("\draw[thick] ({:.3f}, {:.3f}) -- ({:.3f}, 3);".
              format(shadow_pos, rect_up, shadow_pos))
        print("\draw[thick] ({:.3f}, {:.3f}) -- ({:.3f}, 0) node[align=center, below] {{\\tiny {:s}}};".
              format(shadow_pos, rect_down, shadow_pos, label))

        return None
