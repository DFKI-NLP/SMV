import imgkit
from pathlib import Path  # maybe needed; used in
from spacy import displacy


def color_str(string, clr):
    return clr + string + Color.END


class Color:
    """ Source: https://stackoverflow.com/a/17303428 """
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def save_heatmap_as_image(hm, filename):
    ents = []
    colors = {}
    ii = 0
    for color_token in hm:
        ff = ii + len(color_token.token)

        # One entity in displaCy contains start and end markers (character index) and optionally a label
        # The label can be added by setting "attribution_labels" to True
        ent = {
            'start': ii,
            'end': ff,
            'label': str(color_token.score),
        }

        ents.append(ent)
        # A "colors" dict takes care of the mapping between attribution labels and hex colors
        colors[str(color_token.score)] = color_token.hex()
        ii = ff

    to_render = {
        'text': ''.join([t.token for t in hm]),
        'ents': ents,
    }

    template = """
            <mark class="entity" style="background: {bg}; padding: 0.15em 0.3em; margin: 0 0.2em; line-height: 2.2;
            border-radius: 0.25em; box-decoration-break: clone; -webkit-box-decoration-break: clone">
                {text}
            </mark>
            """

    html = displacy.render(
        to_render,
        style='ent',
        manual=True,
        jupyter=False,
        page=True,
        options={'template': template,
                 'colors': colors,
                 }
    )

    #output_path = Path("./heatmap_renders/" + filename)
    # output_path.open("w", encoding="utf-8").write(html)
    image = imgkit.from_string(html, "heatmap_renders/" + filename)

