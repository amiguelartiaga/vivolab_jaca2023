import justext as jt

def limpiar_texto_html(text):
    stop = jt.get_stoplist("Spanish")
    out = [x.text for x in jt.justext(text,stop) if not x.is_boilerplate]
    return "\n".join(out)
