class ErrorDelProyecto(Exception):
    pass

class ErrorDeConfiguración(ErrorDelProyecto):
    pass

class ErrorDeConstrucción(ErrorDelProyecto):
    pass

class ErrorDeArtefacto(ErrorDelProyecto):
    pass

class CulturaPedidaFueraDelCorpus(ErrorDeConstrucción):
    def __init__(self, etiqueta: str, *, criterio: str, poblacion: str, sugerencias=()) -> None:
        pista = (
            f"\n  ¿Quisiste decir {', '.join(repr(s) for s in sugerencias)}?"
            if sugerencias
            else ""
        )
        super().__init__(
            f"cultura_pedida: la etiqueta {etiqueta!r} no existe en el vocabulario "
            f"normalizado de {poblacion} con el criterio {criterio!r}.\n"
            f"  No se acepta con n = 0: ese n se publica como cobertura cultural al lado "
            f"del resultado, y un cero se lee como falta de cobertura cuando acá es un "
            f"desajuste de vocabulario — que es justo la falla que la cobertura existe "
            f"para distinguir.{pista}\n"
            f"  Mirá el vocabulario real con `poetry run tp3-consultas culturas`."
        )
        self.etiqueta = etiqueta
        self.sugerencias = tuple(sugerencias)

class LicenciaNoDeclarada(ErrorDeConstrucción):
    def __init__(self, museo: str, section: str) -> None:
        super().__init__(
            f"el par (museo={museo!r}, section={section!r}) no tiene entrada en "
            f"`ingesta.tabla_de_licencias`. Un par sin entrada es error duro, no un "
            f"default: declaralo en la configuración."
        )
        self.museo = museo
        self.section = section
