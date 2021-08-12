from Connection import Connection


class Analysis(Connection):
    def __init__(self, from_db: bool = True, raw: bool = True):
        super().__init__(from_db, raw)
