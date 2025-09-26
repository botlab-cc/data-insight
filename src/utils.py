def dataframe_to_text(df, max_rows=10):
    """
    Convierte un DataFrame pequeño a un texto legible para el prompt del LLM.
    Cada fila se convierte en "col: valor, col: valor ...".
    Limita a max_rows filas y añade nota si hay más.
    """
    if df.empty:
        return ""
    rows = df.to_dict(orient="records")
    lines = []
    for i, row in enumerate(rows):
        if i >= max_rows:
            lines.append(f"... ({len(rows)-max_rows} filas omitidas)")
            break
        row_text = ", ".join(f"{k}: {v}" for k, v in row.items())
        lines.append(row_text)
    return "\n".join(lines)
