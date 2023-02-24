import pandas as pd

from gilda import ground_df

TEST_COLUMNS = ['idx', 'text']
TEST_ROWS = [
    (1, "kras"),
    (2, "apoptosis"),
]
TEST_SOLUTIONS = ['hgnc:6407', 'GO:0006915']


def test_pandas_grounding():
    df = pd.DataFrame(TEST_ROWS, columns=TEST_COLUMNS)
    ground_df(df, 'text')
    assert 'text_grounded' in df.columns
    assert list(df['text_grounded']) == TEST_SOLUTIONS

    ground_df(df, 'text', target_column="target_column")
    assert 'target_column' in df.columns
    assert list(df['target_column']) == TEST_SOLUTIONS
