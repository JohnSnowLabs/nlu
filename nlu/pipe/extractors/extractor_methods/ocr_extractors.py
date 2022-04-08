def extract_table(df):
    import pyspark.sql.functions as f
    columns = df.select(f.size(f.col("ocr_table.chunks").getItem(0).chunkText)).collect()[0][0]
    exploded_results = df \
        .select("ocr_table") \
        .withColumn("cells", f.explode(f.col("ocr_table.chunks"))) \
        .select([f.col("cells")[i].getField("chunkText").alias(f"col{i}") for i in
                 range(0, columns)])
    return exploded_results.toPandas()


def extract_tables(df,rename_cols=True):
    df = df.withColumn("table_index", df.ocr_table.area.index)
    # pagennum
    pandas_tables = []
    for r in df.dropDuplicates(["path", "ocr_table", "table_index"]).collect():
        if r.table_index is not None:
            pandas_tables.append(extract_table(df.filter(
                (df.path == r.path) & (df.ocr_table.area.page == r.ocr_table.area.page) & (
                        df.table_index == r.table_index))))
    if rename_cols:
        return use_first_row_as_column_names_for_list_of_dfs(pandas_tables)
    return pandas_tables


def use_first_row_as_column_names(df):
    # Use first row of df to define columns and drop the row afterwars
    df.columns = df.iloc[0].values
    df = df.iloc[1:]
    return df


def use_first_row_as_column_names_for_list_of_dfs(pd_tables):
    new_tables = []
    for t in pd_tables:
        new_tables.append(use_first_row_as_column_names(t))
    return new_tables


